"""TRNet: 运动补偿两帧输入的轻量低光增强网络（Temporal Retinex-style Net）。

设计思路（对应课程内容）
- 灰度变换：输入先做 x^0.4 的幂律提亮，并把同一个幂律提亮结果作为“基底”，网络只学习残差；
- 去噪/复原：把上一帧按光流对齐后与当前帧一起送入网络，多帧冗余让网络区分“噪声”与“真实纹理”；
- 全局上下文：瓶颈层带全局平均池化分支，用来估计整幅画面应当提亮到什么程度（曝光自适应）；
- 运动估计：对齐用 lowlight.flow_context 的 Farneback 光流 + 前后向一致性置信度，网络同时拿到置信度图。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LIFT = 0.4          # 幂律提亮指数，全部数据只用训练集设定，固定不再调
BASE_GAIN = 1.5


def base_map(x: torch.Tensor) -> torch.Tensor:
    """传统灰度变换基底：s = 1.5 * r^0.4（在训练场景上网格搜索得到的全局幂律）。"""
    return (BASE_GAIN * x.clamp_min(0).pow(LIFT)).clamp(0, 1)


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.LeakyReLU(0.1, True),
                         nn.Conv2d(o, o, 3, padding=1), nn.LeakyReLU(0.1, True))


class TRNet(nn.Module):
    def __init__(self, temporal: bool = True, ch=(32, 48, 64, 96)):
        super().__init__()
        self.temporal = temporal
        cin = 7 if temporal else 3
        self.enc = nn.ModuleList([block(cin, ch[0]), block(ch[0], ch[1]), block(ch[1], ch[2]), block(ch[2], ch[3])])
        self.glob = nn.Sequential(nn.Linear(ch[3], ch[3]), nn.LeakyReLU(0.1, True), nn.Linear(ch[3], ch[3]))
        self.dec = nn.ModuleList([block(ch[3] + ch[2], ch[2]), block(ch[2] + ch[1], ch[1]), block(ch[1] + ch[0], ch[0])])
        self.head = nn.Conv2d(ch[0], 3, 3, padding=1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, cur, prev=None, conf=None):
        """cur/prev: [B,3,H,W] in [0,1], prev 已按光流对齐；conf: [B,1,H,W]。"""
        h, w = cur.shape[-2:]
        ph, pw = (-h) % 8, (-w) % 8
        lift = cur.clamp_min(0).pow(LIFT)
        if self.temporal:
            inp = torch.cat([lift, prev.clamp_min(0).pow(LIFT), conf], 1)
        else:
            inp = lift
        inp = F.pad(inp, (0, pw, 0, ph), mode="replicate")
        feats, x = [], inp
        for i, enc in enumerate(self.enc):
            x = enc(x)
            if i < 3:
                feats.append(x); x = F.max_pool2d(x, 2)
        x = x + self.glob(x.mean((2, 3)))[:, :, None, None]
        for dec, skip in zip(self.dec, reversed(feats)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([x, skip], 1))
        res = self.head(x)[..., :h, :w]
        return base_map(cur) + res


def gaussian_window(device, size=11, sigma=1.5):
    g = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-g ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)


def ssim_torch(a, b, win):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    f = lambda t: F.conv2d(t, win, groups=3)
    ma, mb = f(a), f(b)
    va, vb, cov = f(a * a) - ma ** 2, f(b * b) - mb ** 2, f(a * b) - ma * mb
    s = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))
    return s.mean()


def align_prev(prev_rgb: np.ndarray, grid: np.ndarray, device) -> torch.Tensor:
    """用 flow_context 给出的反向采样网格把上一帧(RGB float)对齐到当前帧。"""
    p = torch.from_numpy(np.ascontiguousarray(prev_rgb.transpose(2, 0, 1)))[None].to(device)
    g = torch.from_numpy(grid)[None].to(device)
    return F.grid_sample(p, g, align_corners=True, padding_mode="border")
