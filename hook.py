"""Apply VRAMGuard settings before any node of a prompt runs.

ComfyUI loads models lazily while executing nodes, so a guard node that merely
runs "somewhere" in the graph is usually too late: the checkpoint may already
be resident.  Hooking PromptExecutor.execute lets us read the guard node's
widget values from the submitted prompt and set the ceiling *before* the first
node executes.

This is intentionally defensive: if the hook cannot be installed (different
ComfyUI version, missing module) the node itself still applies the limit when
it executes, just later.
"""

from __future__ import annotations

import logging

from . import vram_guard_core as core

LOGGER = logging.getLogger("VRAMGuard")
_installed = False


def install() -> bool:
    global _installed
    if _installed:
        return True
    try:
        import execution

        executor_cls = execution.PromptExecutor
        original = executor_cls.execute
    except Exception:
        LOGGER.exception("[VRAMGuard] could not find ComfyUI's PromptExecutor; "
                         "the limit will only apply when the node itself executes")
        return False

    if getattr(original, "_vram_guard_patched", False):
        _installed = True
        return True

    def execute(self, *args, **kwargs):
        try:
            prompt = args[0] if args else kwargs.get("prompt")
            core.apply_from_prompt(prompt)
        except Exception:
            LOGGER.exception("[VRAMGuard] failed to apply the limit before execution")
        return original(self, *args, **kwargs)

    execute._vram_guard_patched = True
    try:
        execute.__name__ = getattr(original, "__name__", "execute")
        execute.__doc__ = getattr(original, "__doc__", None)
    except Exception:
        pass

    executor_cls.execute = execute
    _installed = True
    LOGGER.info("[VRAMGuard] pre-execution hook installed")
    return True
