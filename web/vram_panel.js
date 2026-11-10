// Pure canvas drawing for the VRAM Guard live panel.
// Kept free of ComfyUI imports so it can also be tested in a plain browser.

const GIB = 1024 ** 3;

const C = {
  panel: "rgba(18, 21, 26, 0.92)",
  border: "rgba(255, 255, 255, 0.07)",
  text: "#e6e9ee",
  dim: "#9aa3ad",
  track: "rgba(255, 255, 255, 0.10)",
  vram: "#38bdf8",
  gpu: "#c084fc",
  good: "#4ade80",
  warn: "#fbbf24",
  bad: "#f87171",
};

export function fmtGiB(bytes, digits = 1) {
  return ((Number(bytes) || 0) / GIB).toFixed(digits);
}

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function vramColor(pct) {
  return pct >= 90 ? C.bad : pct >= 70 ? C.warn : C.good;
}

function roundRect(ctx, x, y, w, h, r) {
  if (typeof ctx.roundRect === "function") {
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, r);
    return;
  }
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

function fitText(ctx, text, maxWidth) {
  const value = String(text ?? "");
  if (!value || ctx.measureText(value).width <= maxWidth) return value;
  let clipped = value;
  while (clipped.length > 1 && ctx.measureText(`${clipped}…`).width > maxWidth) {
    clipped = clipped.slice(0, -1);
  }
  return `${clipped}…`;
}

function drawBar(ctx, x, y, w, h, pct, color) {
  ctx.fillStyle = C.track;
  roundRect(ctx, x, y, w, h, h / 2);
  ctx.fill();
  const fill = w * clamp01(pct / 100);
  if (fill > 0.5) {
    ctx.fillStyle = color;
    roundRect(ctx, x, y, fill, h, h / 2);
    ctx.fill();
  }
}

function drawBadge(ctx, right, cy, label) {
  if (!label) return 0;
  ctx.font = "bold 9px 'Segoe UI', sans-serif";
  const text = String(label).toUpperCase();
  const textWidth = ctx.measureText(text).width;
  const padX = 6;
  const w = textWidth + padX * 2;
  const h = 15;
  const x = right - w;
  ctx.fillStyle = "rgba(56, 189, 248, 0.16)";
  roundRect(ctx, x, cy - h / 2, w, h, 4);
  ctx.fill();
  ctx.strokeStyle = "rgba(56, 189, 248, 0.35)";
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.fillStyle = "#7dd3fc";
  ctx.textAlign = "center";
  ctx.fillText(text, x + w / 2, cy);
  ctx.textAlign = "left";
  return w;
}

function drawSparkline(ctx, x, y, w, h, history) {
  if (h < 18 || w < 30) return;
  ctx.fillStyle = "rgba(255, 255, 255, 0.035)";
  ctx.fillRect(x, y, w, h);

  ctx.strokeStyle = "rgba(255, 255, 255, 0.07)";
  ctx.lineWidth = 1;
  for (const pct of [25, 50, 75]) {
    const gy = y + h * (1 - pct / 100);
    ctx.beginPath();
    ctx.moveTo(x, gy);
    ctx.lineTo(x + w, gy);
    ctx.stroke();
  }

  const points = Array.isArray(history) ? history : [];
  if (points.length < 2) {
    ctx.fillStyle = "rgba(255, 255, 255, 0.35)";
    ctx.font = "10px 'Segoe UI', sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("正在采集数据…", x + 6, y + h / 2);
    return;
  }

  const MAX = 120;
  const n = points.length;
  const px = (i) => x + (w * (i + (MAX - n))) / (MAX - 1);
  const py = (v) => y + h * (1 - clamp01(v / 100));

  const drawLine = (key, color) => {
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < n; i++) {
      const value = points[i]?.[key];
      if (value === null || value === undefined || Number.isNaN(value)) {
        started = false;
        continue;
      }
      const cx = px(i);
      const cy = py(value);
      if (!started) {
        ctx.moveTo(cx, cy);
        started = true;
      } else {
        ctx.lineTo(cx, cy);
      }
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    ctx.stroke();
  };

  drawLine("vram_pct", C.vram);
  drawLine("gpu_util", C.gpu);
}

export function drawLivePanel(ctx, width, y, height, data) {
  const x = 12;
  const w = width - 24;
  if (w < 40 || height < 56) return;

  ctx.save();
  ctx.textBaseline = "middle";
  ctx.textAlign = "left";

  ctx.fillStyle = C.panel;
  roundRect(ctx, x - 4, y + 2, w + 8, height - 6, 7);
  ctx.fill();
  ctx.strokeStyle = C.border;
  ctx.lineWidth = 1;
  ctx.stroke();

  if (!data) {
    ctx.fillStyle = C.dim;
    ctx.font = "12px 'Segoe UI', sans-serif";
    ctx.fillText("正在连接显存监视…", x + 6, y + height / 2);
    ctx.restore();
    return;
  }

  const device = data.device || {};
  const kind = device.kind || data.kind || "cpu";
  const unified = Boolean(device.unified ?? data.unified);
  const deviceName = device.name || data.name || "未知设备";
  const badge = device.badge || String(kind).toUpperCase();

  // Row 0: which accelerator actually runs the workflow.
  const headerY = y + 16;
  ctx.font = "bold 12px 'Segoe UI', 'PingFang SC', sans-serif";
  ctx.fillStyle = C.text;
  const badgeWidth = ctx.measureText(String(badge).toUpperCase()).width + 12;
  const nameLimit = w - 12 - badgeWidth - 10;
  ctx.fillText(fitText(ctx, deviceName, nameLimit), x + 4, headerY);
  drawBadge(ctx, x + w - 4, headerY, badge);

  if (kind === "cpu") {
    ctx.font = "11px 'Segoe UI', 'PingFang SC', sans-serif";
    ctx.fillStyle = C.warn;
    ctx.fillText("未检测到可用的算力卡", x + 4, y + 42);
    ctx.fillStyle = C.dim;
    ctx.font = "10px 'Segoe UI', 'PingFang SC', sans-serif";
    ctx.fillText(`仅监控 · 系统内存 ${Number(data.system_percent || 0).toFixed(0)}%`, x + 4, y + 60);
    ctx.restore();
    return;
  }

  const total = Number(data.total_bytes) || 0;
  const used = Number(data.used_bytes) || 0;
  const pct = Number(data.vram_pct) || 0;
  const gpu = data.gpu_util;
  const hasGpu = gpu !== null && gpu !== undefined;

  // Row 1: memory values + percentage.
  const row1 = y + 37;
  ctx.font = "12px 'Segoe UI', 'PingFang SC', sans-serif";
  ctx.fillStyle = C.vram;
  const memoryLabel = unified ? "统一内存" : "显存";
  ctx.fillText(memoryLabel, x + 4, row1);
  const labelWidth = ctx.measureText(memoryLabel).width;
  ctx.fillStyle = C.text;
  ctx.font = "12px 'Consolas', 'Segoe UI', monospace";
  const valueText = `${fmtGiB(used)} / ${fmtGiB(total)} GiB`;
  const valueX = x + 4 + labelWidth + 8;
  ctx.fillText(fitText(ctx, valueText, x + w - 56 - valueX), valueX, row1);

  ctx.textAlign = "right";
  ctx.fillStyle = vramColor(pct);
  ctx.font = "bold 13px 'Segoe UI', sans-serif";
  ctx.fillText(`${pct.toFixed(1)}%`, x + w - 4, row1);
  ctx.textAlign = "left";

  // Memory bar with a tick at the configured ceiling.
  const barY = y + 48;
  const barH = 9;
  drawBar(ctx, x + 4, barY, w - 8, barH, pct, vramColor(pct));
  if (Number(data.limit_bytes) > 0 && Number(data.limit_bytes) < total) {
    const lx = x + 4 + (w - 8) * clamp01(Number(data.limit_bytes) / total);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(lx, barY - 3);
    ctx.lineTo(lx, barY + barH + 3);
    ctx.stroke();
  }

  // Row 2: ceiling / reserve.
  const row2 = y + 69;
  ctx.font = "10px 'Segoe UI', 'PingFang SC', sans-serif";
  ctx.fillStyle = C.dim;
  const limitText = data.limit_gb ? `上限 ${Number(data.limit_gb).toFixed(2)} GiB` : "未设置上限";
  const reserve = Number(data.reserve_bytes) || 0;
  const reserveText = data.limit_bytes && reserve > 0 ? ` · 预留 ${fmtGiB(reserve)} GiB` : "";
  const overText = data.over_limit ? " · 已超预算" : "";
  ctx.fillStyle = data.over_limit ? C.bad : C.dim;
  ctx.fillText(fitText(ctx, `${limitText}${reserveText}${overText}`, w - 8), x + 4, row2);

  // Row 3: backend specific detail (unified memory / torch allocator).
  const row3 = y + 86;
  ctx.font = "10px 'Segoe UI', 'PingFang SC', sans-serif";
  ctx.fillStyle = C.dim;
  let detail;
  if (unified) {
    const processGb = fmtGiB(data.process_bytes);
    const systemPct = data.system_percent;
    const mpsGb = fmtGiB(data.mps_allocated_bytes);
    const systemText = systemPct !== null && systemPct !== undefined ? ` · 系统 ${Number(systemPct).toFixed(0)}%` : "";
    detail = `ComfyUI 进程 ${processGb} GiB · MPS ${mpsGb} GiB${systemText}`;
  } else {
    const enforced = data.enforce === "allocator" ? "分配器硬上限" : "预留提示";
    detail = `ComfyUI ${fmtGiB(data.torch_allocated_bytes)} / ${fmtGiB(data.torch_reserved_bytes)} GiB · ${enforced}`;
  }
  ctx.fillText(fitText(ctx, detail, w - 8), x + 4, row3);

  // Row 4: GPU core utilisation.
  const row4 = y + 105;
  ctx.font = "12px 'Segoe UI', 'PingFang SC', sans-serif";
  ctx.fillStyle = C.gpu;
  ctx.fillText("GPU 核心", x + 4, row4);
  ctx.textAlign = "right";
  ctx.fillStyle = C.gpu;
  ctx.font = "bold 13px 'Segoe UI', sans-serif";
  ctx.fillText(hasGpu ? `${Math.round(gpu)}%` : "—", x + w - 4, row4);
  ctx.textAlign = "left";

  const gpuBarY = y + 116;
  drawBar(ctx, x + 4, gpuBarY, w - 8, 6, hasGpu ? gpu : 0, C.gpu);

  // History sparkline: memory % (cyan) and GPU % (purple) over the last 120 s.
  drawSparkline(ctx, x + 4, y + 129, w - 8, height - 137, data.history);

  ctx.restore();
}
