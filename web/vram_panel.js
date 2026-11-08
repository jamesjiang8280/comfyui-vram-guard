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

  if (!data || !data.total_bytes) {
    ctx.fillStyle = C.dim;
    ctx.font = "12px 'Segoe UI', sans-serif";
    const msg = data && data.cuda === false ? "CUDA 不可用" : "正在连接显存监视…";
    ctx.fillText(msg, x + 6, y + height / 2);
    ctx.restore();
    return;
  }

  const total = data.total_bytes;
  const used = data.used_bytes || 0;
  const pct = data.vram_pct || 0;
  const gpu = data.gpu_util;
  const hasGpu = gpu !== null && gpu !== undefined;

  // Row 1: VRAM absolute values + percentage.
  const row1 = y + 18;
  ctx.font = "12px 'Segoe UI', sans-serif";
  ctx.fillStyle = C.vram;
  ctx.fillText("显存", x + 4, row1);
  ctx.fillStyle = C.text;
  ctx.font = "12px 'Consolas', 'Segoe UI', monospace";
  ctx.fillText(`${fmtGiB(used)} / ${fmtGiB(total)} GiB`, x + 40, row1);

  ctx.textAlign = "right";
  ctx.fillStyle = vramColor(pct);
  ctx.font = "bold 13px 'Segoe UI', sans-serif";
  ctx.fillText(`${pct.toFixed(1)}%`, x + w - 4, row1);
  ctx.textAlign = "left";

  // VRAM bar with a tick at the configured limit.
  const barY = y + 29;
  const barH = 9;
  drawBar(ctx, x + 4, barY, w - 8, barH, pct, vramColor(pct));
  if (data.limit_bytes > 0 && data.limit_bytes < total) {
    const lx = x + 4 + (w - 8) * clamp01(data.limit_bytes / total);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(lx, barY - 3);
    ctx.lineTo(lx, barY + barH + 3);
    ctx.stroke();
  }

  // Row 2: limit / reserve.
  const row2 = y + 50;
  ctx.font = "10px 'Segoe UI', sans-serif";
  ctx.fillStyle = C.dim;
  const limitText = data.limit_gb ? `已生效 ${Number(data.limit_gb).toFixed(2)} GiB` : "未设置上限";
  const reserve = data.limit_bytes ? Math.max(0, total - data.limit_bytes) : 0;
  ctx.fillText(
    data.limit_bytes ? `${limitText} · 预留 ${fmtGiB(reserve)} GiB` : limitText,
    x + 4,
    row2,
  );

  // Row 3: GPU core utilisation.
  const row3 = y + 68;
  ctx.font = "12px 'Segoe UI', sans-serif";
  ctx.fillStyle = C.gpu;
  ctx.fillText("GPU 核心", x + 4, row3);
  ctx.textAlign = "right";
  ctx.fillStyle = C.gpu;
  ctx.font = "bold 13px 'Segoe UI', sans-serif";
  ctx.fillText(hasGpu ? `${Math.round(gpu)}%` : "—", x + w - 4, row3);
  ctx.textAlign = "left";

  const gpuBarY = y + 79;
  drawBar(ctx, x + 4, gpuBarY, w - 8, 6, hasGpu ? gpu : 0, C.gpu);

  // History sparkline: VRAM % (cyan) and GPU % (amber) over the last 120 s.
  drawSparkline(ctx, x + 4, y + 92, w - 8, height - 100, data.history);

  ctx.restore();
}
