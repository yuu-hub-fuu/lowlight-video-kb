"""曝光估计：从暗光帧的全局统计量预测“正常光参考”的平均亮度。只用训练场景。

发现：训练场景中，参考图平均亮度与输入平均亮度的相关系数只有约 0.5，
亮度比 4.8±1.2 —— 参考曝光有相当一部分无法由输入决定。端到端网络会借助画面内容
“记住”每个训练场景的曝光，导致验证集越训越差。因此把曝光单独交给一个参数极少的回归模型。

候选模型（留一场景交叉验证选择，不看验证集）
  M0 常数目标     log μ_gt = c0
  M1 固定增益     log μ_gt = c0 + log μ_in
  M2 对数线性     log μ_gt = c0 + c1·log μ_in
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lowlight import ROOT                    # noqa: E402
from splits import TRAIN, VAL                # noqa: E402
from tools.build_cache import load           # noqa: E402


def scene_stats(scene):
    c = load(scene)
    xin = np.log(np.asarray(c["lq"], np.float32).mean((1, 2, 3)) / 255)
    xgt = np.log(np.asarray(c["gt"], np.float32).mean((1, 2, 3)) / 255)
    return xin, xgt


def fit(model, X, Y):
    if model == "M0": return np.array([Y.mean(), 0.0])
    if model == "M1": return np.array([(Y - X).mean(), 1.0])
    A = np.stack([np.ones_like(X), X], 1); return np.linalg.lstsq(A, Y, rcond=None)[0]


def predict(p, X): return p[0] + p[1] * X


data = {s: scene_stats(s) for s in TRAIN}
cv = {}
for m in ("M0", "M1", "M2"):
    errs = []
    for held in data:
        # 场景等权：每个场景先取均值，再拟合
        X = np.array([data[s][0].mean() for s in data if s != held]); Y = np.array([data[s][1].mean() for s in data if s != held])
        p = fit(m, X, Y); errs.append(float(np.abs(predict(p, data[held][0]) - data[held][1]).mean()))
    cv[m] = float(np.mean(errs))
best = min(cv, key=cv.get)
X = np.array([v[0].mean() for v in data.values()]); Y = np.array([v[1].mean() for v in data.values()])
params = fit(best, X, Y)
val_err = {}
for s in VAL:
    xin, xgt = scene_stats(s); val_err[s] = float(np.abs(predict(params, xin) - xgt).mean())
out = dict(model=best, c0=float(params[0]), c1=float(params[1]), loso_cv_mean_abs_log_error=cv,
           note="mean |log error|; 0.10 ≈ ±10% brightness. Validation error recorded for reference only, not used for selection.",
           val_mean_abs_log_error=float(np.mean(list(val_err.values()))), val_by_scene=val_err,
           train_corr_scene_means=float(np.corrcoef(X, Y)[0, 1]))
(ROOT / "records/exposure_model.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
