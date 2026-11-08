"""第①级：泊松-高斯噪声模型 + 运动补偿逐像素卡尔曼滤波（在提亮之前的暗光原始域）。

观测模型   z_t = x_t + n_t,         Var(n_t) = R(x) = a·x + b      （泊松散粒噪声 + 高斯读出噪声）
状态模型   x_t = W_t(x_{t-1}) + w_t, Var(w_t) = Q = R·(q0 + q1·(1-c))  （W_t 为光流变形，c 为光流置信度）

    预测  x⁻ = W(x̂),  P⁻ = W(P) + Q
    新息  d = z − x⁻；3×3 平均后的新息 d̄ 在模型正确时方差约 (P⁻+R)/9
    检验  ρ = exp(−d̄² / (τ²(P⁻+R)/9))  —— 新息过大说明遮挡/光流失败/镜头切换，ρ→0
    更新  K = ρ·P⁻/(P⁻+R) + (1−ρ),  x̂ = x⁻ + K·d,  P = K·R

静止且光流可靠的区域，K 逐帧减小，相当于沿运动轨迹做越来越长的平均；
运动出错的区域 K→1，自动退回当前帧，不产生拖影。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from lowlight import ROOT


def load_noise_model():
    p = ROOT / "records/noise_model.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    return d["a"], d["b"], d.get("kappa", 1 / 9)


@dataclass
class KalmanDenoiser:
    a: float
    b: float
    q0: float = 0.05
    q1: float = 1.0
    tau: float = 3.0
    use_test: bool = True              # 消融：关掉新息检验
    kappa: float = 1 / 9               # Var(3×3均值新息)/Var(新息)，由 calibrate_noise 实测
    x: torch.Tensor | None = field(default=None, repr=False)
    P: torch.Tensor | None = field(default=None, repr=False)
    last_gain: torch.Tensor | None = field(default=None, repr=False)

    def reset(self):
        self.x = self.P = self.last_gain = None

    def R(self, z):
        zs = F.avg_pool2d(z, 5, 1, 2, count_include_pad=False)
        return (self.a * zs.clamp_min(0) + self.b).clamp_min(1e-7)

    @torch.no_grad()
    def update(self, z: torch.Tensor, context=None) -> torch.Tensor:
        """z: [1,3,H,W] 暗光帧(0–1)。context = flow_context(cur, prev) 的 (grid, conf, cut)。"""
        R = self.R(z)
        if self.x is None or context is None or context[2]:
            self.x, self.P = z.clone(), R.clone()
            self.last_gain = torch.ones_like(z)
            return self.x
        grid, conf, _ = context
        g = torch.from_numpy(grid)[None].to(z.device)
        c = torch.from_numpy(conf).to(z.device)[None, None].clamp(0, 1)
        xp = F.grid_sample(self.x, g, align_corners=True, padding_mode="border")
        Pp = F.grid_sample(self.P, g, align_corners=True, padding_mode="border")
        Pp = Pp + R * (self.q0 + self.q1 * (1 - c))
        d = z - xp
        K = Pp / (Pp + R)
        if self.use_test:
            dbar = F.avg_pool2d(d, 3, 1, 1, count_include_pad=False)
            # 噪声存在空间相关性：3×3 平均后方差只降到 κ 倍(实测 κ≈0.33，而非独立同分布的 1/9)
            rho = torch.exp(-dbar.square() / (self.tau ** 2 * (Pp + R) * self.kappa))
            K = rho * K + (1 - rho)
        self.x = (xp + K * d).clamp(0, 1)
        self.P = K * R
        self.last_gain = K
        return self.x


@dataclass
class EMADenoiser:
    """消融对照：固定系数的运动补偿 EMA（无噪声模型、无方差传播）。"""
    beta: float = 0.7
    x: torch.Tensor | None = field(default=None, repr=False)

    def reset(self):
        self.x = None

    @torch.no_grad()
    def update(self, z, context=None):
        if self.x is None or context is None or context[2]:
            self.x = z.clone(); return self.x
        grid, conf, _ = context
        g = torch.from_numpy(grid)[None].to(z.device)
        c = torch.from_numpy(conf).to(z.device)[None, None]
        xp = F.grid_sample(self.x, g, align_corners=True, padding_mode="border")
        w = self.beta * c
        self.x = (1 - w) * z + w * xp
        return self.x
