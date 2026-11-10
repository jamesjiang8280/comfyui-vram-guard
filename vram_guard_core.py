"""Core logic for the ComfyUI VRAM Guard node.

The guard puts a ceiling on the memory ComfyUI may use for a workflow, and
reports which accelerator (GPU / NPU / Apple Silicon GPU / CPU) is actually
running the workflow.

Enforcement layers
------------------

* **CUDA / ROCm**: a hard PyTorch allocator cap
  (``torch.cuda.set_per_process_memory_fraction``) so the process cannot
  allocate past the ceiling, plus comfy-aimdo's "simple VRAM headroom" and
  ComfyUI's ``EXTRA_RESERVED_VRAM`` (the same effect as ``--reserve-vram``).

* **Apple Silicon (MPS)**: macOS has one unified memory pool shared by the
  CPU and the GPU, so the ceiling has to be applied in two places:

  1. ``torch.mps.set_per_process_memory_fraction`` caps every Metal
     allocation of the ComfyUI process.  The fraction is computed against
     Metal's *recommended working set size*
     (``torch.mps.recommended_max_memory()``), and it raises a hard MPS OOM
     once the budget is reached.
  2. ComfyUI's own planner (``comfy.model_management.get_free_memory``) is
     wrapped for MPS so it only ever sees the *remaining budget* instead of
     all free system RAM.  Without this ComfyUI would happily load another
     model and only hit the allocator wall halfway through.

  ``EXTRA_RESERVED_VRAM`` is deliberately *not* used on MPS: ComfyUI derives
  its MPS numbers from the whole system RAM, so a large reserve there would
  make it believe the machine has far less memory than it does and refuse to
  load anything.

* **XPU / NPU / MLU / CPU**: reported for monitoring.  A memory-fraction API
  is used when the backend provides one; otherwise only ComfyUI's
  ``EXTRA_RESERVED_VRAM`` hint can be applied.
"""

from __future__ import annotations

import collections
import ctypes
import logging
import os
import platform
import re
import subprocess
import sys
import threading
import time

LOGGER = logging.getLogger("VRAMGuard")

GIB = 1024 ** 3
DEFAULT_LIMIT_GB = 12.0

_IS_DARWIN = sys.platform == "darwin"
_IS_WINDOWS = os.name == "nt"

_SUPPORTED_KINDS = ("cuda", "mps", "xpu", "npu", "mlu")

_lock = threading.RLock()
_state = {
    "ok": False,
    "backend": None,
    "kind": None,
    "device_name": None,
    "unified": False,
    "limit_gb": None,
    "limit_bytes": None,
    "budget_bytes": 0,
    "fraction": None,
    "fraction_basis_bytes": None,
    "metal_cap_bytes": None,
    "reserve_bytes": None,
    "aimdo_headroom_bytes": None,
    "enforce": None,
    "applied_at": None,
    "source": None,
    "error": None,
    "peak_process_bytes": 0,
}


# ---------------------------------------------------------------------------
# Small cross-platform helpers
# ---------------------------------------------------------------------------

def _torch():
    import torch
    return torch


def _has_cuda() -> bool:
    try:
        return bool(_torch().cuda.is_available())
    except Exception:
        return False


def _has_mps() -> bool:
    try:
        return bool(_torch().backends.mps.is_available())
    except Exception:
        return False


def _backend_available(name: str) -> bool:
    """True when ``torch.<name>`` exists and reports an available device."""
    try:
        module = getattr(_torch(), name, None)
        return module is not None and bool(module.is_available())
    except Exception:
        return False


def _command_output(argv, timeout: float = 2.0) -> str:
    """Run a small read-only command and return its stdout ('' on failure)."""
    kwargs = {"capture_output": True, "text": True, "timeout": timeout}
    if _IS_WINDOWS:
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    try:
        proc = subprocess.run(argv, **kwargs)
        return proc.stdout or ""
    except Exception:
        return ""


def _sysctl(name: str) -> str:
    if not _IS_DARWIN:
        return ""
    return _command_output(["sysctl", "-n", name]).strip()


_chip_name_cache = None


def apple_chip_name() -> str:
    """Marketing name of the Apple Silicon chip, e.g. ``Apple M5``."""
    global _chip_name_cache
    if _chip_name_cache is not None:
        return _chip_name_cache
    name = _sysctl("machdep.cpu.brand_string")
    if not name and _IS_DARWIN:
        match = re.search(r"Chip:\s*(.+)", _command_output(["system_profiler", "SPHardwareDataType"], timeout=8))
        if match:
            name = match.group(1).strip()
    _chip_name_cache = name
    return name


def cpu_name() -> str:
    name = ""
    if _IS_DARWIN:
        name = _sysctl("machdep.cpu.brand_string")
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        name = line.split(":", 1)[1].strip()
                        break
        except Exception:
            name = ""
    return name or platform.processor() or platform.machine() or "CPU"


def system_memory_bytes() -> int:
    """Total physical (unified) memory of the machine, in bytes."""
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    if _IS_DARWIN:
        value = _sysctl("hw.memsize")
        if value.isdigit():
            return int(value)
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass
    return 0


def _virtual_memory() -> dict:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {
            "total": int(vm.total),
            "available": int(vm.available),
            "used": int(vm.used),
            "percent": float(vm.percent),
        }
    except Exception:
        total = system_memory_bytes()
        return {"total": total, "available": 0, "used": 0, "percent": 0.0}


def process_footprint_bytes(include_children: bool = True) -> int:
    """Resident memory of this process (plus its children, if asked)."""
    try:
        import psutil
        try:
            process = psutil.Process(os.getpid())
        except Exception:
            process = psutil.Process()
        total = int(process.memory_info().rss)
        if include_children:
            for child in process.children(recursive=True):
                try:
                    total += int(child.memory_info().rss)
                except Exception:
                    continue
        return total
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

_device_lock = threading.Lock()
_device_cache = {"at": 0.0, "info": None}
_DEVICE_CACHE_TTL = 5.0


def _comfy_device_type():
    """The device type ComfyUI itself selected, when available."""
    try:
        import comfy.model_management as mm
        device = mm.get_torch_device()
        kind = str(getattr(device, "type", "") or "")
        return kind if kind in _SUPPORTED_KINDS + ("cpu",) else None
    except Exception:
        return None


def _pick_kind() -> str:
    kind = _comfy_device_type()
    if kind:
        return kind
    if _has_cuda():
        return "cuda"
    for name in ("xpu", "npu", "mlu"):
        if _backend_available(name):
            return name
    if _has_mps():
        return "mps"
    return "cpu"


def _device_name(kind: str) -> str:
    torch = None
    try:
        torch = _torch()
    except Exception:
        pass
    if kind == "cuda":
        try:
            return str(torch.cuda.get_device_name(0))
        except Exception:
            return "NVIDIA GPU"
    if kind == "xpu":
        for getter in ("get_device_name", "get_device_properties"):
            try:
                value = getattr(torch.xpu, getter)(0)
                return str(getattr(value, "name", value))
            except Exception:
                continue
        return "Intel GPU"
    if kind == "npu":
        try:
            return str(torch.npu.get_device_name(0))
        except Exception:
            return "Ascend NPU"
    if kind == "mlu":
        try:
            return str(torch.mlu.get_device_name(0))
        except Exception:
            return "Cambricon MLU"
    if kind == "mps":
        return apple_chip_name() or "Apple Silicon GPU"
    return cpu_name()


def _device_total(kind: str) -> int:
    torch = None
    try:
        torch = _torch()
    except Exception:
        pass
    if kind == "cuda":
        try:
            return int(torch.cuda.get_device_properties(0).total_memory)
        except Exception:
            return 0
    if kind == "xpu":
        try:
            return int(torch.xpu.get_device_properties(0).total_memory)
        except Exception:
            return 0
    if kind == "npu":
        try:
            return int(torch.npu.get_device_properties(0).total_memory)
        except Exception:
            return 0
    if kind == "mlu":
        try:
            return int(torch.mlu.get_device_properties(0).total_memory)
        except Exception:
            return 0
    # MPS and CPU both live in the machine's unified/physical memory.
    return system_memory_bytes()


def mps_recommended_bytes() -> int:
    try:
        return int(_torch().mps.recommended_max_memory())
    except Exception:
        return 0


def mps_allocated_bytes() -> int:
    try:
        return int(_torch().mps.current_allocated_memory())
    except Exception:
        return 0


def mps_driver_bytes() -> int:
    """Metal allocations of this process, including cached blocks."""
    try:
        return int(_torch().mps.driver_allocated_memory())
    except Exception:
        return 0


def device_info(refresh: bool = False) -> dict:
    """Describe the accelerator ComfyUI is using (cached for a few seconds)."""
    now = time.time()
    with _device_lock:
        cached = _device_cache["info"]
        if cached and not refresh and now - _device_cache["at"] < _DEVICE_CACHE_TTL:
            return dict(cached)
        kind = _pick_kind()
        name = _device_name(kind)
        runtime = "CPU"
        vendor = platform.system() or "CPU"
        unified = kind in ("mps", "cpu")
        total = _device_total(kind)
        recommended = mps_recommended_bytes() if kind == "mps" else None
        if kind == "cuda":
            hip = ""
            try:
                hip = _torch().version.hip or ""
            except Exception:
                hip = ""
            if hip:
                vendor = "AMD"
                runtime = f"ROCm {hip}"
            else:
                cuda_version = ""
                try:
                    cuda_version = _torch().version.cuda or ""
                except Exception:
                    cuda_version = ""
                vendor = "NVIDIA"
                runtime = f"CUDA {cuda_version}".strip()
        elif kind == "xpu":
            vendor, runtime = "Intel", "XPU"
        elif kind == "npu":
            vendor, runtime = "Ascend", "NPU"
        elif kind == "mlu":
            vendor, runtime = "Cambricon", "MLU"
        elif kind == "mps":
            vendor, runtime = "Apple", "MPS"
        info = {
            "kind": kind,
            "name": name,
            "vendor": vendor,
            "runtime": runtime,
            "unified": unified,
            "total_bytes": int(total or 0),
            "recommended_bytes": recommended,
            "system_total_bytes": system_memory_bytes(),
            "os": platform.platform(),
        }
        _device_cache["info"] = dict(info)
        _device_cache["at"] = now
        return dict(info)


def device_badge(info: dict | None = None) -> str:
    info = info or device_info()
    kind = info.get("kind")
    return {
        "cuda": "ROCm" if str(info.get("runtime", "")).startswith("ROCm") else "CUDA",
        "mps": "MPS",
        "xpu": "XPU",
        "npu": "NPU",
        "mlu": "MLU",
        "cpu": "CPU",
    }.get(kind, str(kind or "?").upper())


def device_label(info: dict | None = None) -> str:
    """One-line human readable accelerator description."""
    info = info or device_info()
    name = info.get("name") or "未知设备"
    total = int(info.get("total_bytes") or 0)
    if info.get("kind") == "mps":
        memory = f"统一内存 {total / GIB:.2f} GiB" if total else "统一内存"
        return f"{name}（Apple Silicon · MPS · {memory}）"
    if info.get("kind") == "cpu":
        return f"{name}（CPU）"
    memory = f"显存 {total / GIB:.2f} GiB" if total else ""
    runtime = info.get("runtime") or badge
    suffix = f"{runtime} · {memory}" if memory else str(runtime)
    return f"{name}（{suffix}）"


# ---------------------------------------------------------------------------
# comfy-aimdo (DynamicVRAM) helpers
# ---------------------------------------------------------------------------

def _aimdo():
    """Return comfy_aimdo.control if this ComfyUI build ships it."""
    try:
        import comfy_aimdo.control as control
        return control
    except Exception:
        return None


def aimdo_ready() -> bool:
    control = _aimdo()
    return control is not None and getattr(control, "lib", None) is not None


def aimdo_usage_bytes() -> int | None:
    control = _aimdo()
    if control is None or getattr(control, "lib", None) is None:
        return None
    try:
        return int(control.get_total_vram_usage())
    except Exception:
        return None


def aimdo_headroom_bytes() -> int | None:
    control = _aimdo()
    if control is None or getattr(control, "lib", None) is None:
        return None
    try:
        return int(control.get_simple_vram_headroom())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Live monitoring
# ---------------------------------------------------------------------------

class _NvmlUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _GpuUtilization:
    """Device-wide utilisation: NVML (CUDA), ioreg (Apple), else ``None``."""

    def __init__(self):
        self._lock = threading.Lock()
        self._lib = None
        self._handle = None
        self._failed = False
        self._smi_value = None
        self._smi_at = 0.0
        self._ioreg_value = None
        self._ioreg_at = 0.0

    def _load_nvml(self):
        if self._lib is not None or self._failed:
            return
        try:
            lib = ctypes.CDLL("nvml.dll") if _IS_WINDOWS else ctypes.CDLL("libnvidia-ml.so.1")
            if lib.nvmlInit_v2() != 0:
                raise RuntimeError("nvmlInit_v2 failed")
            handle = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != 0:
                raise RuntimeError("nvmlDeviceGetHandleByIndex_v2 failed")
            lib.nvmlDeviceGetUtilizationRates.argtypes = [ctypes.c_void_p, ctypes.POINTER(_NvmlUtilization)]
            lib.nvmlDeviceGetUtilizationRates.restype = ctypes.c_int
            self._lib = lib
            self._handle = handle
        except Exception:
            self._failed = True
            LOGGER.debug("[VRAMGuard] NVML unavailable, falling back to nvidia-smi", exc_info=True)

    def _read_nvml(self):
        if self._lib is None:
            self._load_nvml()
        if self._lib is None:
            return None
        util = _NvmlUtilization()
        if self._lib.nvmlDeviceGetUtilizationRates(self._handle, ctypes.byref(util)) != 0:
            return None
        return int(util.gpu)

    def _read_smi(self):
        now = time.time()
        if self._smi_at and now - self._smi_at < 2.0:
            return self._smi_value
        value = None
        try:
            output = _command_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            )
            value = int(output.strip().splitlines()[0])
        except Exception:
            value = None
        self._smi_at = now
        self._smi_value = value
        return value

    def _read_ioreg(self):
        """Apple Silicon GPU utilisation from the IOAccelerator registry."""
        now = time.time()
        if self._ioreg_at and now - self._ioreg_at < 1.5:
            return self._ioreg_value
        value = None
        output = _command_output(
            ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator", "-k", "PerformanceStatistics", "-w0"],
            timeout=3.0,
        )
        if output:
            values = [int(v) for v in re.findall(r'"Device Utilization %"=(\d+)', output)]
            if values:
                value = max(values)
        self._ioreg_at = now
        self._ioreg_value = value
        return value

    def read(self, kind: str | None = None) -> int | None:
        with self._lock:
            kind = kind or device_info().get("kind")
            if kind == "cuda":
                value = self._read_nvml()
                if value is None:
                    value = self._read_smi()
                return value
            if kind == "mps":
                return self._read_ioreg()
            return None


_GPU_UTIL = _GpuUtilization()

_HISTORY_LEN = 120
_history = collections.deque(maxlen=_HISTORY_LEN)
_history_lock = threading.Lock()
_sampler_lock = threading.Lock()
_sampler_started = False


def _vram_pct(snap: dict) -> float:
    total = int(snap.get("total_bytes") or 0)
    used = int(snap.get("used_bytes") or 0)
    return round(used * 100.0 / total, 1) if total else 0.0


def _sample_once() -> dict:
    snap = snapshot()
    sample = {
        "ts": time.time(),
        "vram_pct": _vram_pct(snap),
        "gpu_util": _GPU_UTIL.read(snap.get("kind")),
    }
    with _history_lock:
        _history.append(sample)
    return sample


def _sampler_loop() -> None:
    while True:
        try:
            _sample_once()
        except Exception:
            LOGGER.debug("[VRAMGuard] live sample failed", exc_info=True)
        time.sleep(1.0)


def _ensure_sampler() -> None:
    global _sampler_started
    if _sampler_started:
        return
    with _sampler_lock:
        if _sampler_started:
            return
        threading.Thread(target=_sampler_loop, name="VRAMGuardSampler", daemon=True).start()
        _sampler_started = True


# ---------------------------------------------------------------------------
# ComfyUI planner integration (Apple Silicon unified memory)
# ---------------------------------------------------------------------------

def _active_budget_bytes() -> int:
    state = get_state()
    if not state.get("ok"):
        return 0
    return int(state.get("budget_bytes") or 0)


def install_mps_budget_patch() -> bool:
    """Teach ComfyUI that MPS has only ``budget - driver allocations`` left.

    ComfyUI computes MPS free memory from the *whole system* RAM.  With a
    ceiling in place we must report the remaining slice of the budget instead,
    otherwise the planner keeps loading models until the Metal allocator
    throws a hard OOM mid-graph.
    """
    try:
        import comfy.model_management as mm
    except Exception:
        return False
    with _lock:
        current = getattr(mm, "get_free_memory", None)
        if current is None or getattr(current, "_vram_guard_patched", False):
            return current is not None

        def get_free_memory(dev=None, torch_free_too=False):
            try:
                if dev is None:
                    dev = mm.get_torch_device()
            except Exception:
                dev = None
            if getattr(dev, "type", None) == "mps":
                budget = _active_budget_bytes()
                if budget > 0:
                    driver = mps_driver_bytes()
                    cached = max(0, driver - mps_allocated_bytes())
                    system_available = int(_virtual_memory().get("available") or 0)
                    budget_left = max(0, budget - driver)
                    total = min(system_available, budget_left) if system_available else budget_left
                    total += cached
                    return (total, cached) if torch_free_too else total
            return current(dev, torch_free_too)

        get_free_memory._vram_guard_patched = True
        get_free_memory._vram_guard_original = current
        try:
            get_free_memory.__name__ = getattr(current, "__name__", "get_free_memory")
            get_free_memory.__doc__ = getattr(current, "__doc__", None)
        except Exception:
            pass
        mm.get_free_memory = get_free_memory
        LOGGER.info("[VRAMGuard] ComfyUI MPS free-memory accounting patched")
        return True


def unload_models() -> None:
    """Drop ComfyUI's cached models and allocator caches."""
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception:
        LOGGER.exception("[VRAMGuard] failed to unload cached models")
    try:
        torch = _torch()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        elif _has_mps():
            torch.mps.synchronize()
            torch.mps.empty_cache()
    except Exception:
        LOGGER.exception("[VRAMGuard] failed to empty the accelerator cache")


# ---------------------------------------------------------------------------
# Applying a ceiling
# ---------------------------------------------------------------------------

def _set_cuda_limit(limit_gb: float, *, unload_cached_models: bool, info: dict) -> dict:
    torch = _torch()
    device_count = max(1, int(torch.cuda.device_count()))
    total = int(torch.cuda.get_device_properties(0).total_memory)
    limit_bytes = int(round(limit_gb * GIB))
    if limit_bytes >= total:
        limit_bytes = total
        fraction = 1.0
    else:
        fraction = limit_bytes / float(total)

    if unload_cached_models and limit_bytes < total:
        in_use = 0
        for index in range(device_count):
            try:
                in_use = max(in_use, int(torch.cuda.memory_reserved(index)))
            except Exception:
                pass
        in_use = max(in_use, int(aimdo_usage_bytes() or 0))
        if in_use > limit_bytes:
            LOGGER.info("[VRAMGuard] %d MiB in use > %d MiB limit, unloading cached models",
                        in_use // (1024 * 1024), limit_bytes // (1024 * 1024))
            unload_models()

    for index in range(device_count):
        torch.cuda.set_per_process_memory_fraction(fraction, index)

    reserve = max(0, total - limit_bytes)
    try:
        import comfy.model_management as mm
        mm.EXTRA_RESERVED_VRAM = reserve
    except Exception:
        LOGGER.debug("[VRAMGuard] comfy.model_management not importable", exc_info=True)

    headroom = None
    if aimdo_ready():
        control = _aimdo()
        try:
            if hasattr(control, "set_simple_vram_headroom"):
                control.set_simple_vram_headroom(reserve)
                headroom = int(control.get_simple_vram_headroom())
            else:
                LOGGER.warning("[VRAMGuard] comfy_aimdo has no set_simple_vram_headroom")
        except Exception:
            LOGGER.exception("[VRAMGuard] failed to set comfy-aimdo headroom")

    return {
        "limit_gb": limit_gb,
        "limit_bytes": limit_bytes,
        "total_bytes": total,
        "fraction": fraction,
        "fraction_basis_bytes": total,
        "budget_bytes": 0 if limit_bytes >= total else limit_bytes,
        "reserve_bytes": reserve,
        "aimdo_headroom_bytes": headroom,
        "enforce": "allocator",
    }


def _set_mps_limit(limit_gb: float, *, unload_cached_models: bool, info: dict) -> dict:
    torch = _torch()
    total = system_memory_bytes() or int(info.get("total_bytes") or 0)
    recommended = mps_recommended_bytes()
    limit_bytes = int(round(limit_gb * GIB))
    unlimited = total > 0 and limit_bytes >= total
    if unlimited:
        limit_bytes = total
        fraction = 1.0
    else:
        fraction = (limit_bytes / float(recommended)) if recommended else 1.0
        fraction = max(0.0, min(2.0, fraction))

    metal_cap = int(fraction * recommended) if recommended else 0

    if unload_cached_models and not unlimited:
        in_use = max(mps_driver_bytes(), process_footprint_bytes())
        if in_use > limit_bytes:
            LOGGER.info("[VRAMGuard] %d MiB in use > %d MiB unified-memory limit, unloading cached models",
                        in_use // (1024 * 1024), limit_bytes // (1024 * 1024))
            unload_models()

    # Hard cap on every Metal allocation made by this process.
    torch.mps.set_per_process_memory_fraction(float(fraction))
    install_mps_budget_patch()

    return {
        "limit_gb": limit_gb,
        "limit_bytes": limit_bytes,
        "total_bytes": total,
        "fraction": fraction,
        "fraction_basis_bytes": recommended,
        "metal_cap_bytes": metal_cap,
        "budget_bytes": 0 if unlimited else limit_bytes,
        "reserve_bytes": max(0, total - limit_bytes),
        "aimdo_headroom_bytes": None,
        "enforce": "allocator+planner" if not unlimited else "off",
    }


def _set_other_accelerator_limit(kind: str, limit_gb: float, *, unload_cached_models: bool, info: dict) -> dict:
    """Best-effort ceiling for XPU / NPU / MLU.

    Uses the backend's per-process memory fraction when it exists; otherwise
    only ComfyUI's ``EXTRA_RESERVED_VRAM`` hint can be set.
    """
    module = getattr(_torch(), kind, None)
    total = int(info.get("total_bytes") or 0)
    limit_bytes = int(round(limit_gb * GIB))
    if total and limit_bytes >= total:
        limit_bytes = total
        fraction = 1.0
    else:
        fraction = (limit_bytes / float(total)) if total else 1.0

    enforce = "reserve"
    setter = getattr(module, "set_per_process_memory_fraction", None) if module is not None else None
    if callable(setter):
        try:
            setter(float(fraction), 0)
            enforce = "allocator"
        except TypeError:
            try:
                setter(float(fraction))
                enforce = "allocator"
            except Exception:
                LOGGER.exception("[VRAMGuard] %s memory fraction failed", kind)
        except Exception:
            LOGGER.exception("[VRAMGuard] %s memory fraction failed", kind)

    reserve = max(0, total - limit_bytes)
    try:
        import comfy.model_management as mm
        mm.EXTRA_RESERVED_VRAM = reserve
    except Exception:
        LOGGER.debug("[VRAMGuard] comfy.model_management not importable", exc_info=True)

    return {
        "limit_gb": limit_gb,
        "limit_bytes": limit_bytes,
        "total_bytes": total,
        "fraction": fraction,
        "fraction_basis_bytes": total,
        "budget_bytes": 0 if limit_bytes >= total else limit_bytes,
        "reserve_bytes": reserve,
        "aimdo_headroom_bytes": None,
        "enforce": enforce,
    }


def apply_limit(limit_gb: float = DEFAULT_LIMIT_GB, *,
                unload_cached_models: bool = True,
                source: str = "node") -> dict:
    """Apply a memory ceiling to the current process.

    ``limit_gb`` is the total amount of accelerator memory ComfyUI may use, in
    GiB.  On Apple Silicon it is the part of the *unified memory* budget that
    belongs to the workflow.  Anything the system (or another application)
    already uses is *not* subtracted: the value is the ComfyUI budget.

    Never raises; the returned dict always describes what happened.
    """
    result = {"ok": False, "source": source}
    try:
        info = device_info(refresh=True)
        kind = info.get("kind")
        if kind not in _SUPPORTED_KINDS:
            raise RuntimeError(
                "未检测到可用的算力卡（CUDA / ROCm / MPS / XPU / NPU 均不可用），只能显示状态"
            )

        try:
            limit_gb = float(limit_gb)
        except (TypeError, ValueError):
            raise ValueError("limit_gb must be a number")
        if limit_gb <= 0:
            raise ValueError("limit_gb must be > 0")

        if kind == "mps":
            details = _set_mps_limit(limit_gb, unload_cached_models=unload_cached_models, info=info)
        elif kind == "cuda":
            details = _set_cuda_limit(limit_gb, unload_cached_models=unload_cached_models, info=info)
        else:
            details = _set_other_accelerator_limit(
                kind, limit_gb, unload_cached_models=unload_cached_models, info=info,
            )

        result.update(details)
        result.update({
            "ok": True,
            "kind": kind,
            "backend": device_badge(info),
            "device_name": info.get("name"),
            "unified": bool(info.get("unified")),
        })
        with _lock:
            _state.update({
                "ok": True,
                "kind": kind,
                "backend": device_badge(info),
                "device_name": info.get("name"),
                "unified": bool(info.get("unified")),
                "limit_gb": limit_gb,
                "limit_bytes": details["limit_bytes"],
                "budget_bytes": details["budget_bytes"],
                "fraction": details["fraction"],
                "fraction_basis_bytes": details.get("fraction_basis_bytes"),
                "metal_cap_bytes": details.get("metal_cap_bytes"),
                "reserve_bytes": details.get("reserve_bytes"),
                "aimdo_headroom_bytes": details.get("aimdo_headroom_bytes"),
                "enforce": details.get("enforce"),
                "applied_at": time.time(),
                "source": source,
                "error": None,
            })
        LOGGER.info(
            "[VRAMGuard] %s limit=%.2f GiB (%.1f%% of %s) reserve=%.2f GiB enforce=%s source=%s",
            device_badge(info), limit_gb, details["fraction"] * 100.0,
            "unified memory" if kind == "mps" else "device memory",
            details.get("reserve_bytes", 0) / GIB, details.get("enforce"), source,
        )
    except Exception as exc:
        LOGGER.exception("[VRAMGuard] failed to apply the memory limit")
        result["error"] = str(exc)
        with _lock:
            _state.update({
                "ok": False,
                "applied_at": time.time(),
                "source": source,
                "error": str(exc),
            })
    return result


# ---------------------------------------------------------------------------
# Snapshots / state / reports
# ---------------------------------------------------------------------------

def snapshot(device: int = 0) -> dict:
    """Current accelerator + allocator numbers.  Never raises."""
    out = {"kind": "cpu", "cuda": False, "device_index": device}
    try:
        info = device_info()
        out.update({
            "kind": info.get("kind"),
            "name": info.get("name"),
            "vendor": info.get("vendor"),
            "runtime": info.get("runtime"),
            "unified": bool(info.get("unified")),
            "recommended_bytes": info.get("recommended_bytes"),
            "total_bytes": int(info.get("total_bytes") or 0),
        })
        kind = info.get("kind")
        if kind == "cuda":
            torch = _torch()
            props = torch.cuda.get_device_properties(device)
            total = int(props.total_memory)
            free, _ = torch.cuda.mem_get_info(device)
            out.update({
                "cuda": True,
                "total_bytes": total,
                "free_bytes": int(free),
                "used_bytes": total - int(free),
                "torch_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "torch_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "aimdo_usage_bytes": aimdo_usage_bytes(),
                "aimdo_headroom_bytes": aimdo_headroom_bytes(),
            })
        elif kind == "mps":
            vm = _virtual_memory()
            total = int(vm.get("total") or system_memory_bytes())
            used = int(vm.get("used") or 0)
            footprint = process_footprint_bytes()
            torch_allocated = mps_allocated_bytes()
            torch_driver = mps_driver_bytes()
            out.update({
                "cuda": False,
                "total_bytes": total,
                "used_bytes": used,
                "free_bytes": int(vm.get("available") or max(0, total - used)),
                "system_used_bytes": used,
                "system_percent": vm.get("percent"),
                "process_bytes": footprint,
                "torch_allocated_bytes": torch_allocated,
                "torch_reserved_bytes": torch_driver,
                "mps_allocated_bytes": torch_allocated,
                "mps_driver_bytes": torch_driver,
                "aimdo_usage_bytes": None,
                "aimdo_headroom_bytes": None,
            })
        elif kind in ("xpu", "npu", "mlu"):
            module = getattr(_torch(), kind, None)
            total = int(info.get("total_bytes") or 0)
            free = 0
            try:
                free = int(module.mem_get_info(device)[0])
            except Exception:
                try:
                    free = int(module.mem_get_info()[0])
                except Exception:
                    free = 0
            out.update({
                "cuda": False,
                "free_bytes": free,
                "used_bytes": max(0, total - free),
                "torch_allocated_bytes": 0,
                "torch_reserved_bytes": 0,
                "aimdo_usage_bytes": None,
                "aimdo_headroom_bytes": None,
            })
        else:  # CPU only
            vm = _virtual_memory()
            total = int(vm.get("total") or 0)
            used = int(vm.get("used") or 0)
            out.update({
                "cuda": False,
                "total_bytes": total,
                "used_bytes": used,
                "free_bytes": int(vm.get("available") or 0),
                "system_percent": vm.get("percent"),
                "process_bytes": process_footprint_bytes(),
                "torch_allocated_bytes": 0,
                "torch_reserved_bytes": 0,
            })
    except Exception as exc:
        out["error"] = str(exc)
    return out


def get_state() -> dict:
    with _lock:
        return dict(_state)


def planner_patch_active() -> bool:
    """True once ComfyUI's MPS free-memory accounting is under our control."""
    try:
        import comfy.model_management as mm
        return bool(getattr(mm.get_free_memory, "_vram_guard_patched", False))
    except Exception:
        return False


def status() -> dict:
    return {
        "state": get_state(),
        "device": snapshot(),
        "device_info": device_info(),
        "planner_patch": planner_patch_active(),
    }


def _gb(value) -> str:
    try:
        return f"{int(value) / GIB:.2f} GiB"
    except Exception:
        return "n/a"


def _note_peak(process_bytes: int) -> None:
    if not process_bytes:
        return
    with _lock:
        if process_bytes > int(_state.get("peak_process_bytes") or 0):
            _state["peak_process_bytes"] = int(process_bytes)


def live_payload() -> dict:
    """One snapshot plus the rolling history used by the node's live panel."""
    _ensure_sampler()
    snap = snapshot()
    info = device_info()
    total = int(snap.get("total_bytes") or 0)
    used = int(snap.get("used_bytes") or 0)
    state = get_state()
    if snap.get("kind") == "mps":
        _note_peak(int(snap.get("process_bytes") or 0))
    gpu = _GPU_UTIL.read(snap.get("kind"))
    with _history_lock:
        history = list(_history)
    if not history:
        history = [{
            "ts": time.time(),
            "vram_pct": round(used * 100.0 / total, 1) if total else 0.0,
            "gpu_util": gpu,
        }]
    budget = int(state.get("budget_bytes") or 0)
    process_bytes = int(snap.get("process_bytes") or 0)
    return {
        "ts": time.time(),
        "kind": snap.get("kind"),
        "cuda": bool(snap.get("cuda")),
        "name": snap.get("name"),
        "vendor": snap.get("vendor"),
        "runtime": snap.get("runtime"),
        "unified": bool(snap.get("unified")),
        "device": {
            "kind": info.get("kind"),
            "name": info.get("name"),
            "vendor": info.get("vendor"),
            "runtime": info.get("runtime"),
            "unified": bool(info.get("unified")),
            "total_bytes": int(info.get("total_bytes") or 0),
            "recommended_bytes": info.get("recommended_bytes"),
            "badge": device_badge(info),
            "label": device_label(info),
        },
        "total_bytes": total,
        "used_bytes": used,
        "free_bytes": int(snap.get("free_bytes") or 0),
        "vram_pct": round(used * 100.0 / total, 1) if total else 0.0,
        "process_bytes": process_bytes,
        "torch_allocated_bytes": int(snap.get("torch_allocated_bytes") or 0),
        "torch_reserved_bytes": int(snap.get("torch_reserved_bytes") or 0),
        "mps_allocated_bytes": snap.get("mps_allocated_bytes"),
        "mps_driver_bytes": snap.get("mps_driver_bytes"),
        "system_percent": snap.get("system_percent"),
        "aimdo_usage_bytes": snap.get("aimdo_usage_bytes"),
        "aimdo_headroom_bytes": snap.get("aimdo_headroom_bytes"),
        "limit_gb": state.get("limit_gb"),
        "limit_bytes": int(state.get("limit_bytes") or 0),
        "budget_bytes": budget,
        "fraction": state.get("fraction"),
        "fraction_basis_bytes": state.get("fraction_basis_bytes"),
        "metal_cap_bytes": state.get("metal_cap_bytes"),
        "reserve_bytes": int(state.get("reserve_bytes") or 0),
        "enforce": state.get("enforce"),
        "planner_patch": planner_patch_active(),
        "over_limit": bool(budget and process_bytes and process_bytes > budget),
        "peak_process_bytes": int(state.get("peak_process_bytes") or 0),
        "gpu_util": gpu,
        "history": history,
    }


def format_report(result: dict, snap: dict) -> str:
    lines: list[str] = []
    info = device_info()
    lines.append(f"算力卡：{device_label(info)}")

    if result.get("ok"):
        kind = result.get("kind") or info.get("kind")
        lines.append(f"内存上限已生效：{result.get('limit_gb', 0):.2f} GiB"
                     f"（为系统/其他程序预留 {_gb(result.get('reserve_bytes'))}）")
        fraction = result.get("fraction") or 0.0
        if kind == "mps":
            basis = result.get("fraction_basis_bytes") or 0
            lines.append(f"MPS 分配器上限：{fraction * 100:.1f}%（fraction={fraction:.4f}，"
                         f"基准是 Metal 推荐工作集 {_gb(basis)}）")
            lines.append(f"ComfyUI 规划预算：{_gb(result.get('budget_bytes'))}"
                         f"（已接管 MPS 可用内存计算）")
        elif kind == "cuda":
            lines.append(f"PyTorch 分配器上限：{fraction * 100:.1f}%（fraction={fraction:.4f}）")
        else:
            lines.append(f"分配器上限：{fraction * 100:.1f}%（fraction={fraction:.4f}，"
                         f"enforce={result.get('enforce')}）")
    elif result.get("error"):
        lines.append(f"内存上限未生效：{result.get('error')}")
    else:
        lines.append("尚未设置内存上限（本进程还没有运行过 VRAM Guard 节点）")

    if snap.get("kind") == "mps":
        lines.append(f"统一内存：已用 {_gb(snap.get('used_bytes'))} / "
                     f"共 {_gb(snap.get('total_bytes'))}"
                     + (f"（系统 {float(snap.get('system_percent') or 0):.1f}%）"
                        if snap.get("system_percent") is not None else ""))
        lines.append(f"ComfyUI 进程：常驻 {_gb(snap.get('process_bytes'))}（含子进程）")
        lines.append(f"MPS 分配：{_gb(snap.get('mps_allocated_bytes'))} / "
                     f"driver {_gb(snap.get('mps_driver_bytes'))}")
    elif snap.get("cuda") or snap.get("kind") in ("xpu", "npu", "mlu"):
        lines.append(f"显存：已用 {_gb(snap.get('used_bytes'))} / "
                     f"共 {_gb(snap.get('total_bytes'))} / 空闲 {_gb(snap.get('free_bytes'))}")
        lines.append(f"ComfyUI torch：已分配 {_gb(snap.get('torch_allocated_bytes'))} / "
                     f"已预留 {_gb(snap.get('torch_reserved_bytes'))}")
        if snap.get("aimdo_usage_bytes") is not None:
            lines.append(f"aimdo 记账：{_gb(snap.get('aimdo_usage_bytes'))} / "
                         f"headroom {_gb(snap.get('aimdo_headroom_bytes'))}")
    else:
        lines.append("没有可用的算力卡，仅显示系统内存。")

    lines.append(f"来源：{result.get('source', '?')}")
    return "\n".join(lines)


def _coerce_limit(value, default: float = DEFAULT_LIMIT_GB) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(default)


def apply_from_prompt(prompt) -> dict | None:
    """Read every VRAMGuard node in a prompt and apply the strictest limit."""
    if not isinstance(prompt, dict):
        return None
    guards = [node for node in prompt.values()
              if isinstance(node, dict) and node.get("class_type") == "VRAMGuard"]
    if not guards:
        return None

    limit = min(_coerce_limit(node.get("inputs", {}).get("limit_gb")) for node in guards)
    unload = any(bool(node.get("inputs", {}).get("unload_cached_models", True)) for node in guards)
    return apply_limit(limit, unload_cached_models=unload, source="prompt")
