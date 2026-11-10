"""Node definitions for ComfyUI VRAM Guard."""

from __future__ import annotations

import logging

from . import vram_guard_core as core

LOGGER = logging.getLogger("VRAMGuard")

_CATEGORY = "VRAM"


class VRAMGuard:
    """Set a hard VRAM ceiling for this ComfyUI process / workflow."""

    DESCRIPTION = (
        "限制 ComfyUI 跑工作流可用的显存/统一内存上限（GiB），并在节点面板上显示当前算力卡型号"
        "（NVIDIA / AMD / Apple Silicon / Intel / Ascend）。在每次提交工作流时、任何节点执行前生效："
        "CUDA 上设置 PyTorch 分配器上限与 comfy-aimdo 预留显存；Apple Silicon 上同时限制 MPS "
        "分配器并接管 MPS 可用内存计算，从而在 macOS 统一内存里给 ComfyUI 划出预算。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "limit_gb": ("FLOAT", {
                    "default": 12.0,
                    "min": 1.0,
                    "max": 1024.0,
                    "step": 0.25,
                    "tooltip": "ComfyUI 最多可用的显存（GiB）。16GB 卡填 12 表示给系统/游戏留约 4GiB。"
                               "Apple Silicon 上表示统一内存预算（按 Metal 推荐工作集折算 fraction）。"
                               "填大于等于物理内存的值可取消限制。",
                }),
                "unload_cached_models": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "如果当前缓存/进程占用已经超过新上限，先卸载缓存模型再安装上限（会变慢一次）。"
                               "Apple Silicon 上会同时检查 MPS 分配与进程常驻内存。",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("report", "limit_mb", "free_mb")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = _CATEGORY

    def run(self, limit_gb: float, unload_cached_models: bool = True):
        result = core.apply_limit(limit_gb, unload_cached_models=unload_cached_models, source="node")
        snap = core.snapshot()
        report = core.format_report(result, snap)
        limit_mb = int(result.get("limit_bytes", 0) // (1024 * 1024))
        free_mb = int(snap.get("free_bytes", 0) // (1024 * 1024))
        return {
            "ui": {"text": [report], "device": [core.device_label()]},
            "result": (report, limit_mb, free_mb),
        }


class VRAMMonitor:
    """Read-only VRAM report node."""

    DESCRIPTION = ("只读：显示当前用于跑工作流的算力卡型号、显存/统一内存占用、"
                   "ComfyUI 进程占用和 aimdo 记账，不修改任何设置。")

    @classmethod
    def INPUT_TYPES(cls):
        return {}

    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("report", "total_gb", "used_gb", "free_gb")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = _CATEGORY

    def run(self):
        snap = core.snapshot()
        with_state = core.get_state()
        result = {
            "ok": with_state.get("ok", False),
            "limit_gb": with_state.get("limit_gb"),
            "reserve_bytes": with_state.get("reserve_bytes"),
            "fraction": with_state.get("fraction"),
            "error": with_state.get("error"),
            "source": with_state.get("source", "monitor"),
        }
        report = core.format_report(result, snap)
        gib = 1024 ** 3
        total_gb = snap.get("total_bytes", 0) / gib
        used_gb = snap.get("used_bytes", 0) / gib
        free_gb = snap.get("free_bytes", 0) / gib
        return {
            "ui": {"text": [report], "device": [core.device_label()]},
            "result": (report, total_gb, used_gb, free_gb),
        }


NODE_CLASS_MAPPINGS = {
    "VRAMGuard": VRAMGuard,
    "VRAMMonitor": VRAMMonitor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRAMGuard": "VRAM 上限 (VRAM Guard)",
    "VRAMMonitor": "VRAM 监视 (VRAM Monitor)",
}
