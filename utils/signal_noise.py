"""
时间序列上「信号 / 噪声」的可解释分解（用于 APAVA 等 [B,T,C] 数据）

思路（可写进论文/实验说明）
---------------------------
- **假设**：慢变、可平滑的部分更接近「形态/信号」；残差为高频「噪声/伪迹」。
- **实现**：沿时间维对每通道做 **对称滑动平均** 得 s_t；残差 r_t = x_t - s_t。
- **检验**：能量比 η = E[||r||²] / E[||x||²]；η 过大说明核太小或数据极不平滑，η 过小说明过度平滑。

可视化不依赖训练；单独运行 `scripts/visualize_signal_noise.py`。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


def _ensure_odd_kernel(kernel: int) -> int:
    k = max(3, int(kernel))
    if k % 2 == 0:
        k += 1
    return k


def moving_average_decompose(
    x: torch.Tensor, kernel: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x: [B, T, C]
        kernel: 奇数滑动窗口长度（若为偶数则 +1）
    Returns:
        signal, noise 与 x 同形；signal 为滑动平均，noise = x - signal
    """
    k = _ensure_odd_kernel(kernel)
    pad = k // 2
    # [B, C, T]
    xt = x.transpose(1, 2).contiguous()
    b, c, t = xt.shape
    weight = torch.ones(c, 1, k, device=x.device, dtype=x.dtype) / float(k)
    x_pad = F.pad(xt, (pad, pad), mode="reflect")
    smooth = F.conv1d(x_pad, weight, groups=c)
    assert smooth.size(-1) == t, (smooth.size(-1), t, k)
    noise = xt - smooth
    return smooth.transpose(1, 2).contiguous(), noise.transpose(1, 2).contiguous()


def energy_statistics(
    x: torch.Tensor, signal: torch.Tensor, noise: torch.Tensor
) -> Dict[str, float]:
    """标量能量与噪声能量占比，便于日志与检查分离是否合理。"""
    ex = (x**2).mean().clamp_min(1e-12)
    es = (signal**2).mean()
    en = (noise**2).mean()
    return {
        "energy_x": float(ex.item()),
        "energy_signal": float(es.item()),
        "energy_noise": float(en.item()),
        "noise_energy_ratio": float((en / ex).item()),
    }


def numpy_moving_average_decompose(
    x: np.ndarray, kernel: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """numpy [T, C] 或 [T]；用于离线可视化。"""
    t = torch.from_numpy(np.asarray(x, dtype=np.float32))
    if t.dim() == 1:
        t = t.unsqueeze(-1)
    t = t.unsqueeze(0)
    s, n = moving_average_decompose(t, kernel=kernel)
    return s.squeeze(0).numpy(), n.squeeze(0).numpy()


def plot_decomposition(
    x_tc: np.ndarray,
    kernel: int = 5,
    save_path: str = "signal_noise_vis.png",
    channels: Tuple[int, ...] = (0, 1, 2),
    title: Optional[str] = None,
) -> None:
    """
    将原始 x、信号 s、噪声 r 画在同一张图的子图中（需 matplotlib）。
    x_tc: [T, C]
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T, C = x_tc.shape
    s, r = numpy_moving_average_decompose(x_tc, kernel=kernel)
    chs = [c for c in channels if c < C]
    if not chs:
        chs = list(range(min(3, C)))

    fig, axes = plt.subplots(len(chs), 1, figsize=(10, 2.5 * len(chs)), sharex=True)
    if len(chs) == 1:
        axes = [axes]
    t_axis = np.arange(T)
    for ax, c in zip(axes, chs):
        ax.plot(t_axis, x_tc[:, c], "k-", alpha=0.35, label="x (input)")
        ax.plot(t_axis, s[:, c], "b-", lw=1.2, label=f"signal (MA k={kernel})")
        ax.plot(t_axis, r[:, c], "r-", alpha=0.7, lw=0.8, label="noise (residual)")
        ax.set_ylabel(f"ch {c}")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time step")
    fig.suptitle(title or "Signal vs noise (moving average decomposition)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
