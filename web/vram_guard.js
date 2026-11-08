// ComfyUI frontend extension: live VRAM / GPU utilisation panel for VRAMGuard.
import { app } from "../../scripts/app.js";
import { drawLivePanel } from "./vram_panel.js";

const live = {
  data: null,
  nodes: new Set(),
  timer: null,
  inflight: false,
  warned: false,
};

async function refresh() {
  if (live.inflight) return;
  live.inflight = true;
  try {
    const res = await fetch("/vram_guard/live", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    live.data = await res.json();
    for (const node of live.nodes) node.setDirtyCanvas(true, true);
  } catch (err) {
    if (!live.warned) {
      live.warned = true;
      console.warn("[VRAMGuard] live stats unavailable:", err);
    }
  } finally {
    live.inflight = false;
  }
}

function subscribe(node) {
  live.nodes.add(node);
  if (!live.timer) {
    refresh();
    live.timer = setInterval(refresh, 1000);
  }
}

function unsubscribe(node) {
  live.nodes.delete(node);
  if (live.nodes.size === 0 && live.timer) {
    clearInterval(live.timer);
    live.timer = null;
  }
}

class VRAMLiveWidget {
  constructor() {
    this.type = "vram_live";
    this.name = "vram_live";
    this.value = null;
    this.options = { serialize: false };
    this.serialize = false;
    this.y = 0;
    this.last_y = 0;
    this.disabled = false;
  }

  computeSize(width) {
    return [width, 150];
  }

  draw(ctx, node, width, y, height) {
    // ComfyUI passes the standard 20px widget height here even though
    // computeSize() reserved the whole panel area; use the node geometry.
    const panelHeight = Math.max(150, (node?.size?.[1] || y + 158) - y - 8);
    drawLivePanel(ctx, width, y, panelHeight, live.data);
  }

  serializeValue() {
    return undefined;
  }

  mouse() {
    return false;
  }
}

function setupNodeType(nodeType) {
  if (nodeType.prototype.__vramGuardLivePatched) return;
  nodeType.prototype.__vramGuardLivePatched = true;

  nodeType.prototype.__resizeForLivePanel = function () {
    const required = typeof this.computeSize === "function" ? this.computeSize() : null;
    const width = Math.max(this.size?.[0] || 0, required?.[0] || 0, 380);
    const height = Math.max(this.size?.[1] || 0, required?.[1] || 0, 240);
    if (!this.size || this.size[0] !== width || this.size[1] !== height) {
      this.setSize([width, height]);
    }
  };

  const onNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const result = onNodeCreated?.apply(this, arguments);
    try {
      if (!this.__vramLiveWidget) {
        this.__vramLiveWidget = new VRAMLiveWidget();
        this.__vramLiveWidget.parent = this;
        if (typeof this.addCustomWidget === "function") {
          this.addCustomWidget(this.__vramLiveWidget);
        } else {
          this.widgets = this.widgets || [];
          this.widgets.push(this.__vramLiveWidget);
        }
        this.__resizeForLivePanel();
      }
      subscribe(this);
    } catch (err) {
      console.error("[VRAMGuard] failed to create the live panel", err);
    }
    return result;
  };

  const onConfigure = nodeType.prototype.onConfigure;
  nodeType.prototype.onConfigure = function () {
    const result = onConfigure?.apply(this, arguments);
    try {
      if (this.__vramLiveWidget) {
        this.__resizeForLivePanel();
        subscribe(this);
      }
    } catch (err) {
      console.error("[VRAMGuard] failed to restore the live panel", err);
    }
    return result;
  };

  const onRemoved = nodeType.prototype.onRemoved;
  nodeType.prototype.onRemoved = function () {
    unsubscribe(this);
    return onRemoved?.apply(this, arguments);
  };
}

app.registerExtension({
  name: "Comfy.VRAMGuard.LiveStats",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "VRAMGuard" || nodeData.name === "VRAMMonitor") {
      setupNodeType(nodeType);
    }
  },
});
