"""标定泊松-高斯噪声模型 Var(n) = a·x + b，只用训练场景。

做法：相邻两帧按光流对齐，在光流置信度 > 0.8 的像素上取差分 d = z_t − W(z_{t−1})，
理想情况下 Var(d) = 2·R(x)。按亮度分箱，用 MAD 稳健估计方差(抑制光流残差/纹理的影响)，
再在 x<0.25（暗光帧绝大多数像素所在区间）做加权最小二乘直线拟合。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lowlight import ROOT, flow_context          # noqa: E402
from train_trnet import TRAIN, load_arrays       # noqa: E402
from vnet import align_prev                      # noqa: E402

import cv2                                        # noqa: E402

edges = np.linspace(0, 0.4, 41)
samples = [[] for _ in range(40)]
kappas = []
mad_var = lambda v: (1.4826 * np.median(np.abs(v - np.median(v)))) ** 2
rng = np.random.default_rng(0)
for scene, n in TRAIN.items():
    lq, _ = load_arrays(scene, n)
    for i in range(1, n, 2):
        c = lq[i].astype(np.float32) / 255; p = lq[i - 1].astype(np.float32) / 255
        grid, conf, cut = flow_context(c, p)
        if cut: continue
        a = align_prev(p, grid, "cuda")[0].permute(1, 2, 0).cpu().numpy()
        ok = np.repeat((conf > 0.8)[..., None], 3, -1)
        full = c - a
        dark = ok & (c < 0.2)
        # 空间相关性：3×3 均值后的方差与原方差之比（独立同分布时应为 1/9）
        kappas.append(mad_var(cv2.blur(full, (3, 3))[dark]) / mad_var(full[dark]))
        d, mu = full[ok], ((c + a) / 2)[ok]
        idx = np.digitize(mu, edges) - 1
        for k in range(40):
            v = d[idx == k]
            if len(v) > 4000: v = rng.choice(v, 4000, replace=False)
            samples[k].append(v)
    print("calibrated", scene, flush=True)
mid, var, cnt = [], [], []
for k in range(40):
    if not samples[k]: continue
    v = np.concatenate(samples[k])
    if len(v) < 2000: continue
    mad = np.median(np.abs(v - np.median(v)))
    mid.append((edges[k] + edges[k + 1]) / 2); var.append((1.4826 * mad) ** 2 / 2); cnt.append(len(v))
mid, var, cnt = map(np.array, (mid, var, cnt))
sel = mid < 0.25
A = np.stack([mid[sel], np.ones(sel.sum())], 1); wts = np.sqrt(cnt[sel])
a, b = np.linalg.lstsq(A * wts[:, None], var[sel] * wts, rcond=None)[0]
a, b = float(max(a, 0)), float(max(b, 1e-7))
out = dict(a=a, b=b, kappa=float(np.mean(kappas)), kappa_std=float(np.std(kappas)), model="Var = a*x + b (x in [0,1])", fit_range="x<0.25", estimator="MAD of aligned temporal differences / 2",
           scenes=list(TRAIN), bins=dict(x=mid.tolist(), var=var.tolist(), count=cnt.tolist()))
(ROOT / "records/noise_model.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"kappa={out['kappa']:.3f}±{out['kappa_std']:.3f}")
print(f"a={a:.3e} b={b:.3e}  -> std at x=0.05: {np.sqrt(a*.05+b)*255:.2f}/255, x=0.2: {np.sqrt(a*.2+b)*255:.2f}/255")
