"""课程内的传统图像处理基线（均为纯 numpy/OpenCV，无学习参数）。

对应课程章节：灰度变换(幂律/对数)、直方图均衡与 CLAHE、空间平滑与保边滤波、
Retinex 照度-反射分解、非局部均值去噪、引导滤波。
"""
from __future__ import annotations

import cv2
import numpy as np

from vnet import BASE_GAIN, LIFT


def u8(x):
    return np.clip(np.rint(x * 255), 0, 255).astype(np.uint8)


def power_law(x: np.ndarray) -> np.ndarray:
    """灰度变换：s = c·r^γ，c、γ 在训练场景上网格搜索 (1.5, 0.4)。"""
    return np.clip(BASE_GAIN * np.maximum(x, 0) ** LIFT, 0, 1)


def hist_equalize(x: np.ndarray) -> np.ndarray:
    """全局直方图均衡（在 YCrCb 的亮度通道上做，保持色度）。"""
    ycc = cv2.cvtColor(u8(x), cv2.COLOR_RGB2YCrCb)
    ycc[..., 0] = cv2.equalizeHist(ycc[..., 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB).astype(np.float32) / 255


def clahe(x: np.ndarray, clip=3.0, tiles=8) -> np.ndarray:
    """限制对比度自适应直方图均衡 (CLAHE)，在 Lab 亮度通道上做。"""
    lab = cv2.cvtColor(u8(x), cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255


def msr(x: np.ndarray, sigmas=(15, 80, 250)) -> np.ndarray:
    """多尺度 Retinex：log(I) − log(G_σ * I) 取平均后线性拉伸。"""
    x = np.maximum(x, 1e-3)
    r = sum(np.log(x) - np.log(cv2.GaussianBlur(x, (0, 0), s) + 1e-3) for s in sigmas) / len(sigmas)
    lo, hi = np.percentile(r, 1), np.percentile(r, 99)
    return np.clip((r - lo) / max(hi - lo, 1e-6), 0, 1)


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """He 等人的引导滤波（灰度引导图），盒式滤波实现，O(N)。"""
    k = (2 * r + 1, 2 * r + 1)
    mean = lambda t: cv2.boxFilter(t, -1, k, borderType=cv2.BORDER_REFLECT)
    mg, ms = mean(guide), mean(src)
    cov = mean(guide * src) - mg * ms
    var = mean(guide * guide) - mg * mg
    a = cov / (var + eps)
    b = ms - a * mg
    return mean(a) * guide + mean(b)


def retinex_guided(x: np.ndarray, nlm_h: float = 6.0, gain_cap: float = 12.0) -> np.ndarray:
    """传统完整流水线：NLM 去噪 → max-RGB 照度估计 → 引导滤波保边平滑 →
    照度幂律拉伸 → 以增益图(反射分量不变)重建 → 对高增益区加大平滑。

        I = R · L,   L ≈ GuidedFilter(max_c I),   Ŝ = R · L'  = I · (L'/L)
    """
    den = cv2.fastNlMeansDenoisingColored(u8(np.power(x, LIFT)), None, nlm_h, nlm_h, 5, 15)
    xd = np.power(den.astype(np.float32) / 255, 1 / LIFT)           # 回到线性域再做 Retinex
    h, w = x.shape[:2]
    l0 = xd.max(-1)
    small = cv2.resize(l0, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    illum = guided_filter(small, small, r=16, eps=1e-3)
    illum = np.maximum(cv2.resize(illum, (w, h), interpolation=cv2.INTER_LINEAR), 1e-3)
    illum2 = np.clip(BASE_GAIN * illum ** LIFT, 0, 1)
    gain = np.minimum(illum2 / illum, gain_cap)[..., None]
    return np.clip(xd * gain, 0, 1)
