"""第②级：可学习双边网格增强（HDRNet 思路的轻量版，Gharbi et al., SIGGRAPH 2017）。

一个小 CNN 只看 256×256 缩略图，输出双边网格 Γ ∈ R^{12×D×16×16}：
空间 16×16 个格子 × 亮度 D 个区间，每格存一个 3×4 颜色仿射矩阵 [A|b]。
全分辨率上用逐点 MLP 得到引导图 g(x,y)∈[0,1]，按 (x, y, g) 对网格做三线性插值（“切片”），
得到每个像素自己的仿射变换：  out(x,y) = A(x,y,g)·u(x,y) + b(x,y,g)，  u = r^0.4 为幂律提亮后的输入。

与课程的关系
- 每个亮度区间一个仿射变换 ⇒ 学出来的是“随位置缓变的分段灰度变换/颜色查找表”；
- 按引导图在亮度维插值 ⇒ 与双边滤波同理，亮度差大的像素不会共用同一变换，天然保边；
- 网格只有 16×16×D 个自由度 ⇒ 只能表示平滑的色调映射，不会“画”出纹理，也不放大噪声的空间结构。
第③级（算子域时域平滑）直接对 Γ 做指数平滑：平滑的是变换而不是像素，因此没有运动拖影。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LIFT = 0.4


def conv(i, o, s=1, act=True):
    layers = [nn.Conv2d(i, o, 3, s, 1)]
    if act: layers.append(nn.LeakyReLU(0.1, True))
    return nn.Sequential(*layers)


class BGNet(nn.Module):
    def __init__(self, depth: int = 8, nf: int = 16, lowres: int = 256):
        super().__init__()
        self.D, self.lowres = depth, lowres
        self.splat = nn.Sequential(conv(3, nf, 2), conv(nf, 2 * nf, 2), conv(2 * nf, 4 * nf, 2), conv(4 * nf, 8 * nf, 2))  # → 16×16
        c = 8 * nf
        self.local = nn.Sequential(conv(c, c), conv(c, c, act=False))
        self.glob_conv = nn.Sequential(conv(c, c, 2), conv(c, c, 2))                                # → 4×4
        self.glob_fc = nn.Sequential(nn.Linear(c, c), nn.LeakyReLU(0.1, True), nn.Linear(c, c))
        self.pred = nn.Conv2d(c, 12 * depth, 1)
        self.guide = nn.Sequential(nn.Conv2d(3, 16, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(16, 1, 1))
        # 初始化：网格每格都是 A = 1.5·I, b = 0，即从“传统幂律变换 1.5·r^0.4”出发
        nn.init.zeros_(self.pred.weight)
        ident = torch.tensor([1.5, 0, 0, 0, 0, 1.5, 0, 0, 0, 0, 1.5, 0])
        with torch.no_grad(): self.pred.bias.copy_(ident.repeat(depth))

    @staticmethod
    def lift(x):
        return x.clamp_min(0).pow(LIFT)

    def predict_grid(self, u):
        low = F.interpolate(u, size=(self.lowres, self.lowres), mode="bilinear", align_corners=False, antialias=True)
        f = self.splat(low)
        g = self.glob_fc(self.glob_conv(f).mean((2, 3)))[:, :, None, None]
        f = F.leaky_relu(self.local(f) + g, 0.1)
        grid = self.pred(f)                                        # [B, D*12, 16, 16]
        B, _, H, W = grid.shape
        return grid.view(B, self.D, 12, H, W).permute(0, 2, 1, 3, 4).contiguous()  # [B,12,D,16,16]

    def guide_map(self, u):
        return torch.sigmoid(self.guide(u))

    def apply(self, u, grid):
        B, _, H, W = u.shape
        g = self.guide_map(u)
        ys = torch.linspace(-1, 1, H, device=u.device); xs = torch.linspace(-1, 1, W, device=u.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([xx.expand(B, H, W), yy.expand(B, H, W), g[:, 0] * 2 - 1], -1)[:, None]  # [B,1,H,W,3]
        coef = F.grid_sample(grid, coords, mode="bilinear", align_corners=True, padding_mode="border")[:, :, 0]  # [B,12,H,W]
        A = coef.view(B, 3, 4, H, W)
        return (A[:, :, :3] * u[:, None]).sum(2) + A[:, :, 3]

    def forward(self, x):
        u = self.lift(x)
        return self.apply(u, self.predict_grid(u))


class GridSmoother:
    """第③级：算子域时域平滑 Γ_t = α·Γ_{t−1} + (1−α)·Γ̂_t，镜头切换时重置。"""

    def __init__(self, alpha: float):
        self.alpha, self.prev = alpha, None

    def reset(self):
        self.prev = None

    def update(self, grid, cut=False):
        if self.prev is None or cut or self.alpha <= 0:
            self.prev = grid
        else:
            self.prev = self.alpha * self.prev + (1 - self.alpha) * grid
        return self.prev
