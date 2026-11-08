"""KBG-Video：卡尔曼去噪(①) + 双边网格增强(②) + 算子域时域平滑(③) + 曝光回归(④)，逐帧在线、因果。

    暗光帧 z_t ──光流(与上一帧)──┐
        │                         ▼
        ├──> ① 运动补偿卡尔曼滤波(泊松-高斯噪声模型, 新息检验) ──> x̂_t（去噪后的暗光帧）
        │                                                            │
        │            u_t = x̂_t^0.4 ──> ② BGNet 预测双边网格 Γ̂_t ──> ③ Γ_t = αΓ_{t-1} + (1-α)Γ̂_t
        │                                                            │
        │                               切片：y_t(p) = A_Γt(p, g(p))·u_t(p) + b_Γt(p, g(p))
        │                                                            │
        └──> ④ 曝光回归 log μ*_t = c0 + c1·log mean(z_t)，同样以 α 做时域平滑 ──> out_t = y_t · μ*_t / mean(y_t)
    镜头切换（直方图+帧差联合判据）时 ①③④ 同时重置。
    ② 只负责“色调形状/颜色/局部对比”，④ 负责“整体多亮”，二者解耦（见 tools/fit_exposure.py 的数据分析）。
"""
from __future__ import annotations

import json

import numpy as np
import torch

from bgnet import BGNet, GridSmoother
from kalman import KalmanDenoiser, load_noise_model
from lowlight import ROOT, ImageState, flow_context, image_numpy, tensor_image


def kalman_from_records(use_test=True):
    sel = json.loads((ROOT / "records/kalman_selection.json").read_text(encoding="utf-8"))["chosen"]
    a, b, k = load_noise_model()
    return KalmanDenoiser(a, b, sel["q0"], sel["q1"], sel["tau"], use_test=use_test, kappa=k)


def exposure_model():
    e = json.loads((ROOT / "records/exposure_model.json").read_text(encoding="utf-8"))
    return e["c0"], e["c1"]


def expose(y, target):
    return (y * (target / y.mean().clamp_min(1e-4))).clamp(0, 1)


def load_bgnet(name="bgnet_kf", device="auto"):
    if device == "auto": device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.loads((ROOT / f"records/train_{name}.json").read_text(encoding="utf-8"))
    net = BGNet(depth=cfg["depth"]).to(device)
    net.load_state_dict(torch.load(ROOT / f"weights/{name}.pth", map_location=device, weights_only=True))
    return net.eval(), device, cfg


class KBGVideo:
    @classmethod
    def final(cls, device="auto"):
        """按 records/v2_config.json 冻结的最终配置构建（验证集选出）。"""
        p = json.loads((ROOT / "records/v2_config.json").read_text(encoding="utf-8"))["final_pipeline"]
        return cls(p["grid_alpha"], p["net"], device, p["denoiser"], p["exposure"], beta=p["pixel_beta"])

    def __init__(self, alpha=0.0, net="bgnet_kf", device="auto", denoiser="kalman", exposure=None, beta=0.0):
        self.beta = beta                 # ③' 像素域运动补偿置信度融合（第一版 ImageState 模块复用）
        self.net, self.dev, cfg = load_bgnet(net, device)
        # 默认：解耦训练的网络配合曝光回归；直接监督的网络自带曝光，不再缩放
        self.exposure = (cfg.get("loss", "direct") == "decoupled") if exposure is None else exposure
        self.c0, self.c1 = exposure_model()
        self.denoiser_kind = denoiser
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.prev = None
        self.den = None if self.denoiser_kind == "none" else kalman_from_records(use_test=self.denoiser_kind != "kalman_no_test")
        self.smoother = GridSmoother(self.alpha)
        self.pix = ImageState(self.beta, True)
        self.log_target = None

    @torch.inference_mode()
    def step(self, rgb: np.ndarray, context="auto"):
        """返回 dict(denoised, raw, out, cut, target)。context 可由调用方传入以复用光流。"""
        if isinstance(context, str):
            context = flow_context(rgb, self.prev) if self.prev is not None else None
        cut = bool(context and context[2])
        z = tensor_image(rgb, self.dev)
        x = self.den.update(z, context) if self.den is not None else z
        u = self.net.lift(x)
        grid = self.net.predict_grid(u)
        lt = self.c0 + self.c1 * float(np.log(max(float(rgb.mean()), 1e-4)))
        raw = self.net.apply(u, grid).clamp(0, 1)
        if self.exposure: raw = expose(raw, float(np.exp(lt)))
        g = self.smoother.update(grid, cut)
        self.log_target = lt if (self.log_target is None or cut) else self.alpha * self.log_target + (1 - self.alpha) * lt
        out = self.net.apply(u, g).clamp(0, 1)
        if self.exposure: out = expose(out, float(np.exp(self.log_target)))
        if self.beta > 0: out = self.pix.update(out, context)
        self.prev = rgb
        return dict(denoised=x, raw=raw, out=out, cut=cut, target=float(np.exp(self.log_target)))

    def step_numpy(self, rgb):
        r = self.step(rgb)
        return image_numpy(r["out"]), r["cut"]
