# ComfyUI VRAM Guard

给 ComfyUI 工作流加一个**显存 / 统一内存天花板**：默认 12 GiB，16 GB 显卡上给系统 / 游戏 / 视频留出约 4 GiB；同时在节点上实时显示**当前跑工作流的算力卡型号**、显存占用与 GPU 利用率。

![VRAM Guard 实时面板](docs/preview.png)

## 功能

- **显示算力卡型号**：节点面板顶部直接显示当前用于跑 ComfyUI 工作流的加速卡，例如 `Apple M5`、`NVIDIA GeForce RTX 5060 Ti`、`AMD Radeon RX 7900 XTX`、`Intel Arc`、`Ascend 910B`，并标注 `CUDA / ROCm / MPS / XPU / NPU / MLU` 运行时。
- **提交前生效**：每次排队工作流时，在任何节点执行之前设置上限（此时模型还没加载），不需要改启动参数。
- **CUDA / ROCm 三层限制**：PyTorch 分配器硬上限 + comfy-aimdo（DynamicVRAM）显存预留 + `EXTRA_RESERVED_VRAM`，效果等价于启动参数 `--reserve-vram 4`。
- **Apple Silicon 统一内存限制**：限制 MPS/Metal 分配器，并接管 ComfyUI 的 MPS 可用内存计算，让工作流在 macOS 统一内存中也有明确预算（见下文）。
- **实时面板**：显存 `已用 / 总量 GiB`、百分比、上限刻度，GPU 核心利用率，以及最近 120 秒的双曲线；每秒刷新，不运行工作流也会更新。
- **只读监视节点**：`VRAM Monitor` 只显示数字，不修改任何设置。
- **HTTP 接口**：`/vram_guard/status`、`/vram_guard/live`、`/vram_guard/apply`，可以脚本化查询/切换上限。

## 安装

```bash
cd <ComfyUI>/custom_nodes
git clone https://github.com/jamesjiang8280/comfyui-vram-guard.git
```

也可以直接把 `comfyui-vram-guard` 目录复制到 `<ComfyUI>/custom_nodes/` 下。

重启 ComfyUI，启动日志里应出现（括号里是检测到的算力卡）：

```
[VRAMGuard] pre-execution hook installed
[VRAMGuard] HTTP routes registered: /vram_guard/status (device: Apple M5（Apple Silicon · MPS · 统一内存 24.00 GiB）)
```

然后在浏览器里按 **Ctrl+F5** 强制刷新一次，加载新的前端扩展。

兼容性：ComfyUI 0.39+；CUDA / ROCm（PyTorch 2.8+）、Apple Silicon（PyTorch 2.12+，MPS）、Intel XPU / Ascend NPU / Cambricon MLU（能拿到设备信息就显示，有 `set_per_process_memory_fraction` 时才会设硬上限）。Windows 上测试于 RTX 5060 Ti 16G + torch 2.12.1+cu130；macOS 上测试于 Apple M5 24G 统一内存 + torch 2.12.1（ComfyUI 0.42.1，PyTorch 2.12 的 `torch.mps.set_per_process_memory_fraction`）。没有 comfy-aimdo 的版本会自动退化为只有 PyTorch 硬上限。

## 使用

1. 在工作流里添加节点 **“VRAM 上限 (VRAM Guard)”**（分类 `VRAM`）。
2. `limit_gb` 填 **12**（表示 ComfyUI 最多用 12 GiB）。
3. 正常排队运行即可；节点不需要连到任何东西，它是输出节点，会自己执行。

提交工作流时，控制台会打印：

```
[VRAMGuard] MPS limit=12.00 GiB (67.6% of unified memory) reserve=12.00 GiB enforce=allocator+planner source=prompt
```

`source=prompt` 表示在第一个节点执行前就生效了（这是关键：模型还没加载）。

节点输出 `report` 可以接到 **Preview Any / Show Text** 查看文字报告，例如 Apple Silicon：

```
算力卡：Apple M5（Apple Silicon · MPS · 统一内存 24.00 GiB）
内存上限已生效：12.00 GiB（为系统/其他程序预留 12.00 GiB）
MPS 分配器上限：67.6%（fraction=0.6757，基准是 Metal 推荐工作集 17.76 GiB）
ComfyUI 规划预算：12.00 GiB（已接管 MPS 可用内存计算）
统一内存：已用 12.44 GiB / 共 24.00 GiB（系统 58.8%）
ComfyUI 进程：常驻 3.20 GiB（含子进程）
MPS 分配：2.10 GiB / driver 2.40 GiB
```

NVIDIA 上则是：

```
算力卡：NVIDIA GeForce RTX 5060 Ti（CUDA 13.0 · 显存 15.90 GiB）
内存上限已生效：12.00 GiB（为系统/其他程序预留 3.90 GiB）
PyTorch 分配器上限：75.5%（fraction=0.7546）
显存：已用 1.10 GiB / 共 15.90 GiB / 空闲 14.81 GiB
ComfyUI torch：已分配 0.00 GiB / 已预留 0.00 GiB
aimdo 记账：0.00 GiB / headroom 3.90 GiB
```

### 实时面板（数字 + 动态曲线）

节点下半部分自带一个每秒刷新的实时面板，**不需要运行工作流**：

- **算力卡**：设备型号 + 运行时标签（`MPS` / `CUDA` / `ROCm` / `NPU` …）；
- **显存 / 统一内存**：`已用 / 总量 GiB` 与百分比（按整卡或整机计算），进度条上的白色刻度是 `limit_gb` 的位置；
- **上限 / 预留**：当前**已生效**的上限，以及为系统/游戏预留的量；
- **进程占用**：Apple Silicon 显示 ComfyUI 进程常驻内存（含子进程）与 MPS 已分配量；CUDA 显示 torch 已分配 / 已预留；
- **GPU 核心**：NVIDIA 通过 NVML 读取；Apple Silicon 通过 IOKit `IOAccelerator` 的 `Device Utilization %` 读取；其他后端显示 `—`；
- **动态曲线**：最近 120 秒的显存占用率（青色）与 GPU 核心利用率（紫色）。

拖动节点右下角把节点拉高，曲线区域会自动变大。数据来自 `GET /vram_guard/live`，前端每秒轮询一次。

注意：面板里的“已生效上限”是当前进程实际生效的值；拖动 `limit_gb` 控件本身不会立刻改上限，排队运行一次后才会更新（避免运行中改上限导致 OOM）。

如果面板一直显示“正在连接显存监视…”，先强制刷新浏览器（Ctrl+F5），并确认 `http://127.0.0.1:8188/vram_guard/live` 能打开。

### Apple Silicon（MPS / 统一内存）

macOS 上 CPU 和 GPU 共用一块统一内存，没有独立的“显存”，所以上限分两层施加：

| 层 | 调用 | 效果 |
|---|---|---|
| MPS 分配器 | `torch.mps.set_per_process_memory_fraction(limit / recommended_max_memory)` | 进程内 Metal/torch 分配超过 `limit_gb` 直接 MPS OOM，不会继续吃内存 |
| ComfyUI 规划器 | 包装 `comfy.model_management.get_free_memory()` | 让 ComfyUI 只看到“预算 - 已分配”的剩余量，提前卸载模型而不是撞上分配器的墙 |

`recommended_max_memory` 是 Metal 的推荐工作集（24 GB 机器上约 17.76 GiB），因此 `limit_gb=12` 时 fraction ≈ 0.6757，分配器允许约 12 GiB。上限可以超过推荐工作集（fraction 允许到 2.0），但要小于物理内存才有意义。

**为什么不用 `EXTRA_RESERVED_VRAM`？** ComfyUI 的 MPS 内存数字来自整机 RAM，如果把 reserve 设成 `24-12=12 GiB`，ComfyUI 会以为机器只剩很少内存而拒绝加载任何模型。所以 MPS 路径改用上面的规划器包装，NVIDIA 路径仍保留 `EXTRA_RESERVED_VRAM` 语义。

注意：MPS 硬上限是**按进程**生效的。ComfyUI 工作流里调用外部 MLX / 其他 Python 子进程时，子进程有自己独立的内存预算（面板会把它计入“ComfyUI 进程（含子进程）”显示，但无法替它设限）。

### 取消限制

把 `limit_gb` 填成大于等于物理内存 / 总显存的值（16 GB 卡填 **16**，24 GB 统一内存填 **24**）：CUDA 上 PyTorch 上限恢复为 100%、aimdo 预留归零；Apple Silicon 上 MPS 分配器恢复到系统推荐工作集（fraction=1.0），规划器包装自动失效。

### 状态接口（可选）

```
GET  http://127.0.0.1:8188/vram_guard/status
GET  http://127.0.0.1:8188/vram_guard/live
POST http://127.0.0.1:8188/vram_guard/apply     {"limit_gb": 12}
```

`/vram_guard/status` 返回 `state`（已生效上限）、`device`（设备与当前内存数字）、`device_info`（算力卡型号 / 厂商 / 运行时 / 总内存）和 `planner_patch`（Apple Silicon 上 ComfyUI 规划器是否已被接管）。`/vram_guard/live` 返回面板所需的全部字段、`gpu_util` 和最近 120 秒的历史数组。可以在不运行工作流的情况下查看/修改当前上限（端口按你的 ComfyUI 实际端口改）。

## 它做了什么（CUDA / ROCm）

| 层 | 调用 | 效果 |
|---|---|---|
| PyTorch 分配器 | `torch.cuda.set_per_process_memory_fraction(limit/total)` | 进程内 torch 分配超过上限直接 OOM，不会继续吃显存 |
| comfy-aimdo（DynamicVRAM） | `set_simple_vram_headroom(total-limit)` | 让 DynamicVRAM 把多出来的模型页换出，优先保住“留给其他程序”的空间 |
| ComfyUI 旧估算路径 | `EXTRA_RESERVED_VRAM = total-limit` | 等价于启动参数 `--reserve-vram 4` 的效果 |

## 重要限制（请务必读）

1. **只限内存，不限算力。** 这个节点不会降低 GPU 核心占用率。ComfyUI 跑图时 GPU 仍可能接近 100%，游戏帧数仍然会掉；它保证的是“不会被 ComfyUI 挤爆显存 / 触发 Windows 共享显存卡顿”。
2. **`limit_gb` 是 ComfyUI 的预算，不是整卡读数。** CUDA 上下文、驱动等进程开销另算，`nvidia-smi` 里可能看到约 12.3–12.5 GB。想更严格就填 11.5。
3. **aimdo 的 headroom 是设备级保留。** 游戏已经占用 2 GB 时，ComfyUI 会更早让出显存，总占用不会顶满；这是有意的保守策略。
4. **上限过小会 OOM。** DynamicVRAM / ComfyUI 会尽量换出，但激活值 / VAE 解码仍可能超过上限。遇到 OOM 就把 `limit_gb` 调大 0.5–1。
5. **Apple Silicon 的上限按进程计算。** 外部子进程（MLX 等）不共享 MPS 分配器预算；面板会显示它们的内存，但限制不到它们内部。
6. **改上限会清缓存。** 如果当前已缓存占用超过新上限，节点会先卸载缓存模型（`unload_cached_models=true`），下一次运行会重新加载，首次会慢一些。
7. 没有输入的工作流节点可能被 ComfyUI 缓存而跳过 `run()`，但**预执行钩子每次提交都会执行**，所以上限仍然每次生效（只是节点上的报告文字可能不刷新）。

## 验证

NVIDIA 上跑一个会吃显存的工作流，同时开一个 PowerShell：

```powershell
while ($true) { nvidia-smi --query-gpu=memory.used --format=csv,noheader; Start-Sleep -Seconds 1 }
```

Apple Silicon 上可以用：

```bash
# 进程的 Metal 分配（ComfyUI 的 Python 进程 PID）
sudo powermetrics --samplers gpu_power -i 1000   # 需要 sudo，看 GPU 占用
ioreg -r -d 1 -c IOAccelerator -k PerformanceStatistics -w0 | grep -o '"Device Utilization %"=[0-9]*'
```

在 12 GiB 上限下，ComfyUI 的分配应停在 12 GiB 附近而不是把整卡 / 整机吃满；面板上的百分比、GPU 核心利用率和 120 秒曲线都是每秒刷新的。`GET /vram_guard/status` 里的 `state.fraction` 也能确认（CUDA：12 / 15.90 ≈ 0.7546；MPS：12 / 17.76 ≈ 0.6757）。

## 卸载

删除 `custom_nodes/comfyui-vram-guard`，重启 ComfyUI 即可。限制是进程内的，重启后自然消失。

## English

A ComfyUI custom node that caps the process memory budget (default 12 GiB) **before any node of a prompt executes**, leaving the rest of the GPU (or, on Apple Silicon, the unified memory) for games or video playback. The node panel also shows the accelerator that actually runs the workflow — NVIDIA / AMD ROCm / Apple Silicon MPS / Intel XPU / Ascend NPU / Cambricon MLU — together with live memory usage, GPU utilisation and a 120-second history graph.

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jamesjiang8280/comfyui-vram-guard.git
```

Restart ComfyUI, hard-refresh the browser (Ctrl+F5), then add **VRAM Guard** from the `VRAM` category.

On CUDA the ceiling is enforced with `torch.cuda.set_per_process_memory_fraction()` plus comfy-aimdo's VRAM headroom (equivalent to `--reserve-vram 4`). On Apple Silicon it is enforced with `torch.mps.set_per_process_memory_fraction()` and by wrapping ComfyUI's MPS free-memory accounting so the planner only sees the remaining budget. It limits **memory, not compute** — the GPU can still be saturated while sampling, so a game may still lose FPS; the cap prevents ComfyUI from exhausting device or unified memory.

## License

MIT
