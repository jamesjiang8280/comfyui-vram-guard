"""Node definitions for ComfyUI VRAM Guard."""

from __future__ import annotations

import logging

from . import vram_guard_core as core

LOGGER = logging.getLogger("VRAMGuard")

_CATEGORY = "VRAM"


class VRAMGuard:
    """Set a hard VRAM ceiling for this ComfyUI process / workflow."""

    DESCRIPTION = (
        "限制 ComfyUI 可用的显存上限（GiB）。在每次提交工作流时、任何节点执行前生效，"
        "同时设置 PyTorch 分配器上限和 comfy-aimdo 的保留显存，避免 ComfyUI 把游戏/视频挤爆。"
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
                               "填大于等于总显存的值可取消限制。",
                }),
                "unload_cached_models": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "如果当前缓存占用已经超过新上限，先卸载缓存模型再安装上限（会变慢一次）。",
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
        return {"ui": {"text": [report]}, "result": (report, limit_mb, free_mb)}


class VRAMMonitor:
    """Read-only VRAM report node."""

    DESCRIPTION = "只读：显示当前设备显存、ComfyUI 进程显存和 aimdo 记账，不修改任何设置。"

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
        return {"ui": {"text": [report]}, "result": (report, total_gb, used_gb, free_gb)}


NODE_CLASS_MAPPINGS = {
    "VRAMGuard": VRAMGuard,
    "VRAMMonitor": VRAMMonitor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRAMGuard": "VRAM 上限 (VRAM Guard)",
    "VRAMMonitor": "VRAM 监视 (VRAM Monitor)",
}
