"""Core logic for the ComfyUI VRAM Guard node.

The guard has two layers:

* a hard PyTorch allocator cap (``set_per_process_memory_fraction``) so the
  ComfyUI process cannot allocate more than the requested amount, and
* a comfy-aimdo "simple VRAM headroom" so DynamicVRAM keeps the remaining
  amount free on the device and evicts its own pages instead of spilling into
  Windows shared memory.

It mirrors what ComfyUI's ``--reserve-vram`` startup option does, but can be
changed at runtime from a workflow.
"""

from __future__ import annotations

import collections
import ctypes
import logging
import os
import subprocess
import threading
import time

LOGGER = logging.getLogger("VRAMGuard")

GIB = 1024 ** 3
DEFAULT_LIMIT_GB = 12.0

_lock = threading.RLock()
_state = {
    "ok": False,
    "limit_gb": None,
    "limit_bytes": None,
    "fraction": None,
    "reserve_bytes": None,
    "aimdo_headroom_bytes": None,
    "applied_at": None,
    "source": None,
    "error": None,
}


def _torch():
    import torch
    return torch


def cuda_available() -> bool:
    try:
        return _torch().cuda.is_available()
    except Exception:
        return False


def device_total_bytes(device: int = 0) -> int:
    return int(_torch().cuda.get_device_properties(device).total_memory)


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


# --- live monitoring ---------------------------------------------------------

class _NvmlUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _GpuUtilization:
    """Read device-wide GPU utilisation from NVML, with an nvidia-smi fallback."""

    def __init__(self):
        self._lock = threading.Lock()
        self._lib = None
        self._handle = None
        self._failed = False
        self._smi_value = None
        self._smi_at = 0.0

    def _load(self):
        if self._lib is not None or self._failed:
            return
        try:
            lib = ctypes.CDLL("nvml.dll") if os.name == "nt" else ctypes.CDLL("libnvidia-ml.so.1")
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
            self._load()
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
            kwargs = {"capture_output": True, "text": True, "timeout": 2}
            if os.name == "nt":
                kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                **kwargs,
            )
            value = int(out.stdout.strip().splitlines()[0])
        except Exception:
            value = None
        self._smi_at = now
        self._smi_value = value
        return value

    def read(self) -> int | None:
        with self._lock:
            value = self._read_nvml()
            if value is None:
                value = self._read_smi()
            return value


_GPU_UTIL = _GpuUtilization()

_HISTORY_LEN = 120
_history = collections.deque(maxlen=_HISTORY_LEN)
_history_lock = threading.Lock()
_sampler_lock = threading.Lock()
_sampler_started = False


def _sample_once() -> dict:
    snap = snapshot()
    total = int(snap.get("total_bytes") or 0)
    used = int(snap.get("used_bytes") or 0)
    sample = {
        "ts": time.time(),
        "vram_pct": round(used * 100.0 / total, 1) if total else 0.0,
        "gpu_util": _GPU_UTIL.read(),
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


def live_payload() -> dict:
    """One snapshot plus the rolling history used by the node's live panel."""
    _ensure_sampler()
    snap = snapshot()
    total = int(snap.get("total_bytes") or 0)
    used = int(snap.get("used_bytes") or 0)
    gpu = _GPU_UTIL.read()
    with _history_lock:
        history = list(_history)
    if not history:
        history = [{
            "ts": time.time(),
            "vram_pct": round(used * 100.0 / total, 1) if total else 0.0,
            "gpu_util": gpu,
        }]
    state = get_state()
    return {
        "ts": time.time(),
        "cuda": bool(snap.get("cuda")),
        "name": snap.get("name"),
        "total_bytes": total,
        "used_bytes": used,
        "free_bytes": int(snap.get("free_bytes") or 0),
        "vram_pct": round(used * 100.0 / total, 1) if total else 0.0,
        "torch_allocated_bytes": int(snap.get("torch_allocated_bytes") or 0),
        "torch_reserved_bytes": int(snap.get("torch_reserved_bytes") or 0),
        "aimdo_usage_bytes": snap.get("aimdo_usage_bytes"),
        "aimdo_headroom_bytes": snap.get("aimdo_headroom_bytes"),
        "limit_gb": state.get("limit_gb"),
        "limit_bytes": int(state.get("limit_bytes") or 0),
        "fraction": state.get("fraction"),
        "gpu_util": gpu,
        "history": history,
    }


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
    except Exception:
        LOGGER.exception("[VRAMGuard] failed to empty the CUDA cache")


def apply_limit(limit_gb: float = DEFAULT_LIMIT_GB, *,
                unload_cached_models: bool = True,
                source: str = "node") -> dict:
    """Apply a VRAM ceiling to the current process.

    ``limit_gb`` is the total amount of VRAM ComfyUI may use, in GiB.  Anything
    the device already uses (the desktop, a game, a browser) is *not* subtracted
    from it: the value is the ComfyUI budget, just like ``--reserve-vram``.

    Never raises; the returned dict always describes what happened.
    """
    result = {"ok": False, "source": source}
    try:
        torch = _torch()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; VRAM Guard needs an NVIDIA CUDA device")

        limit_gb = float(limit_gb)
        if limit_gb <= 0:
            raise ValueError("limit_gb must be > 0")

        device_count = max(1, int(torch.cuda.device_count()))
        total = int(torch.cuda.get_device_properties(0).total_memory)
        limit_bytes = int(round(limit_gb * GIB))
        if limit_bytes >= total:
            # No ceiling requested: let PyTorch use the whole device again.
            limit_bytes = total
            fraction = 1.0
        else:
            fraction = limit_bytes / float(total)

        result.update({
            "limit_gb": limit_gb,
            "limit_bytes": limit_bytes,
            "total_bytes": total,
            "fraction": fraction,
        })

        # Free anything already above the new ceiling *before* installing it,
        # otherwise the very next allocation fails until the cache drains.
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

        result.update({
            "ok": True,
            "reserve_bytes": reserve,
            "aimdo_headroom_bytes": headroom,
        })
        with _lock:
            _state.update({
                "ok": True,
                "limit_gb": limit_gb,
                "limit_bytes": limit_bytes,
                "fraction": fraction,
                "reserve_bytes": reserve,
                "aimdo_headroom_bytes": headroom,
                "applied_at": time.time(),
                "source": source,
                "error": None,
            })
        LOGGER.info("[VRAMGuard] limit=%.2f GiB (%.1f%%) reserve=%.2f GiB source=%s",
                    limit_gb, fraction * 100.0, reserve / GIB, source)
    except Exception as exc:
        LOGGER.exception("[VRAMGuard] failed to apply VRAM limit")
        result["error"] = str(exc)
        with _lock:
            _state.update({
                "ok": False,
                "applied_at": time.time(),
                "source": source,
                "error": str(exc),
            })
    return result


def snapshot(device: int = 0) -> dict:
    """Current device + allocator numbers.  Never raises."""
    out = {"cuda": False, "device_index": device}
    try:
        torch = _torch()
        if not torch.cuda.is_available():
            return out
        props = torch.cuda.get_device_properties(device)
        total = int(props.total_memory)
        free, _ = torch.cuda.mem_get_info(device)
        out.update({
            "cuda": True,
            "name": str(props.name),
            "total_bytes": total,
            "free_bytes": int(free),
            "used_bytes": total - int(free),
            "torch_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "torch_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "aimdo_usage_bytes": aimdo_usage_bytes(),
            "aimdo_headroom_bytes": aimdo_headroom_bytes(),
        })
    except Exception as exc:
        out["error"] = str(exc)
    return out


def get_state() -> dict:
    with _lock:
        return dict(_state)


def status() -> dict:
    return {"state": get_state(), "device": snapshot()}


def _gb(value) -> str:
    try:
        return f"{int(value) / GIB:.2f} GiB"
    except Exception:
        return "n/a"


def format_report(result: dict, snap: dict) -> str:
    lines = []
    if result.get("ok"):
        lines.append(f"VRAM 上限已生效：{result.get('limit_gb', 0):.2f} GiB"
                     f"（为系统/游戏预留 {_gb(result.get('reserve_bytes'))}）")
        fraction = result.get("fraction") or 0.0
        lines.append(f"PyTorch 分配器上限：{fraction * 100:.1f}%（fraction={fraction:.4f}）")
    elif result.get("error"):
        lines.append(f"VRAM 上限未生效：{result.get('error')}")
    else:
        lines.append("尚未设置 VRAM 上限（本进程还没有运行过 VRAM Guard 节点）")

    if snap.get("cuda"):
        lines.append(f"设备：{snap.get('name', '?')}")
        lines.append(f"全系统显存：已用 {_gb(snap.get('used_bytes'))} / "
                     f"共 {_gb(snap.get('total_bytes'))} / 空闲 {_gb(snap.get('free_bytes'))}")
        lines.append(f"ComfyUI torch：已分配 {_gb(snap.get('torch_allocated_bytes'))} / "
                     f"已预留 {_gb(snap.get('torch_reserved_bytes'))}")
        if snap.get("aimdo_usage_bytes") is not None:
            lines.append(f"aimdo 记账：{_gb(snap.get('aimdo_usage_bytes'))} / "
                         f"headroom {_gb(snap.get('aimdo_headroom_bytes'))}")
    else:
        lines.append("CUDA 不可用，只能显示状态。")

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
