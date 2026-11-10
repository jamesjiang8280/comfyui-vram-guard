"""ComfyUI VRAM Guard: runtime memory ceiling for ComfyUI workflows.

Works on CUDA / ROCm, Apple Silicon (MPS, unified memory), Intel XPU,
Ascend NPU and Cambricon MLU; the node panel always shows which accelerator
is running the workflow and how much memory it is allowed to use.

Install: put this folder into ComfyUI/custom_nodes/ and restart ComfyUI.
"""

from __future__ import annotations

from .hook import install as _install_hook
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .routes import register as _register_routes

__version__ = "1.2.0"

WEB_DIRECTORY = "./web"

_install_hook()
_register_routes()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY", "__version__"]
