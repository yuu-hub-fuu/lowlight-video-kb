"""只在训练场景上选择卡尔曼参数（代理指标：去噪后经固定幂律提亮的 PSNR / SSIM）。

规则（事先声明）：在 PSNR 距最优不超过 0.01 dB 的配置中取 SSIM 最高者。
同时记录固定系数 EMA、无新息检验两个对照，供报告引用。
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import classical as C                                                  # noqa: E402
from kalman import EMADenoiser, KalmanDenoiser, load_noise_model       # noqa: E402
from lowlight import ROOT, flow_context, image_numpy, quality, tensor_image  # noqa: E402
from train_trnet import load_arrays                                    # noqa: E402

SCENES = ["pair2", "pair4", "pair14", "pair17", "pair20", "pair24"]   # 训练场景子集
a, b, k = load_noise_model()
data = [load_arrays(s, 30) for s in SCENES]
ctxs = [[None] + [flow_context(lq[i].astype(np.float32) / 255, lq[i - 1].astype(np.float32) / 255) for i in range(1, 30)]
        for lq, _ in data]


def score(make):
    P, S, G = [], [], []
    for (lq, gt), cs in zip(data, ctxs):
        d = make()
        for i in range(30):
            x = d.update(tensor_image(lq[i].astype(np.float32) / 255, "cuda"), cs[i])
            if i % 3 == 2:
                p, s = quality(C.power_law(image_numpy(x)), gt[i].astype(np.float32) / 255)
                P.append(p); S.append(s)
                if hasattr(d, "last_gain"): G.append(float(d.last_gain.mean()))
    return dict(psnr=float(np.mean(P)), ssim=float(np.mean(S)), mean_gain=float(np.mean(G)) if G else None)


rows = []
for q0, q1, tau in itertools.product((0.02, 0.05), (0.0, 1.0), (3, 4, 6, 8, 12)):
    r = score(lambda: KalmanDenoiser(a, b, q0, q1, tau, kappa=k)); r.update(kind="kalman", q0=q0, q1=q1, tau=tau)
    rows.append(r); print(r, flush=True)
controls = []
for beta in (0.5, 0.7, 0.85):
    r = score(lambda: EMADenoiser(beta)); r.update(kind="ema", beta=beta); controls.append(r); print(r, flush=True)
r = score(lambda: KalmanDenoiser(a, b, 0.02, 0.0, 6, use_test=False, kappa=k)); r.update(kind="kalman_no_test", q0=0.02, q1=0.0)
controls.append(r); print(r, flush=True)
r = score(lambda: KalmanDenoiser(a, b, 0.02, 0.0, 6, kappa=1 / 9)); r.update(kind="kalman_iid_kappa", q0=0.02, q1=0.0, tau=6)
controls.append(r); print(r, flush=True)
best_p = max(r["psnr"] for r in rows)
chosen = max([r for r in rows if r["psnr"] >= best_p - 0.01], key=lambda r: r["ssim"])
out = dict(rule="among Kalman configs within 0.01 dB of the best proxy PSNR, take the highest proxy SSIM",
           scenes=SCENES, noise_model=dict(a=a, b=b, kappa=k), chosen=chosen, grid=rows, controls=controls)
(ROOT / "records/kalman_selection.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("CHOSEN", chosen)
