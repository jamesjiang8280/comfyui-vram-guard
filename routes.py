"""Small HTTP surface for debugging and for launcher scripts.

GET  /vram_guard/status -> applied state + live device numbers + device info
GET  /vram_guard/live   -> one snapshot, device info and the 120 s history
POST /vram_guard/apply  -> {"limit_gb": 12.0, "unload_cached_models": true}

The device info block reports the accelerator that actually runs the
workflow (NVIDIA / AMD ROCm / Apple Silicon MPS / Intel XPU / Ascend NPU /
Cambricon MLU / CPU), so scripts can log it next to the memory numbers.
"""

from __future__ import annotations

import asyncio
import logging

from aiohttp import web

from . import vram_guard_core as core

LOGGER = logging.getLogger("VRAMGuard")


def register() -> bool:
    try:
        from server import PromptServer

        server = getattr(PromptServer, "instance", None)
        if server is None:
            LOGGER.warning("[VRAMGuard] PromptServer instance not ready; HTTP routes disabled")
            return False

        routes = server.routes

        @routes.get("/vram_guard/status")
        async def vram_guard_status(request):
            return web.json_response(core.status())

        @routes.get("/vram_guard/live")
        async def vram_guard_live(request):
            payload = await asyncio.to_thread(core.live_payload)
            return web.json_response(payload, headers={"Cache-Control": "no-store"})

        @routes.post("/vram_guard/apply")
        async def vram_guard_apply(request):
            try:
                data = await request.json()
            except Exception:
                data = {}
            try:
                limit_gb = float(data.get("limit_gb", core.DEFAULT_LIMIT_GB))
            except Exception:
                return web.json_response({"error": "limit_gb must be a number"}, status=400)
            unload = bool(data.get("unload_cached_models", True))
            result = await asyncio.to_thread(
                core.apply_limit, limit_gb,
                unload_cached_models=unload, source="http",
            )
            return web.json_response({"result": result, "status": core.status()})

        LOGGER.info("[VRAMGuard] HTTP routes registered: /vram_guard/status (device: %s)",
                    core.device_label())
        return True
    except Exception:
        LOGGER.exception("[VRAMGuard] failed to register HTTP routes")
        return False
