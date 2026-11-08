"""预计算每个场景的中间结果并存盘（内存映射读取，16 GB 内存的笔记本也能训练 34 个场景）。

data/cache/<scene>/
  lq.npy  gt.npy   uint8 RGB 原始帧
  ap.npy  cf.npy   uint8 光流对齐后的上一帧 / 光流置信度（TRNet 输入）
  kf.npy           float16 卡尔曼去噪后的暗光帧（BGNet 输入；保留 8 位以下精度）
卡尔曼参数读取 records/kalman_selection.json（只用训练场景选出）。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kalman import KalmanDenoiser, load_noise_model        # noqa: E402
from lowlight import ROOT, flow_context, quantize, tensor_image  # noqa: E402
from splits import TEST, TRAIN, VAL                         # noqa: E402
from train_trnet import load_arrays                         # noqa: E402
from vnet import align_prev                                 # noqa: E402

CACHE = ROOT / "data/cache"


def kalman_params():
    sel = json.loads((ROOT / "records/kalman_selection.json").read_text(encoding="utf-8"))["chosen"]
    a, b, k = load_noise_model()
    return dict(a=a, b=b, q0=sel["q0"], q1=sel["q1"], tau=sel["tau"], kappa=k)


def build(scene, frames, force=False):
    d = CACHE / scene
    if (d / "kf.npy").exists() and not force:
        return
    d.mkdir(parents=True, exist_ok=True)
    lq, gt = load_arrays(scene, frames)
    kf = KalmanDenoiser(**kalman_params())
    ap, cf, out = [], [], []
    prev = None
    for x8 in lq:
        x = x8.astype(np.float32) / 255
        ctx = flow_context(x, prev) if prev is not None else None
        if ctx is None:
            ap.append(x8); cf.append(np.zeros(x8.shape[:2], np.uint8))
        else:
            grid, mask, cut = ctx
            ap.append(quantize(align_prev(prev, grid, "cuda")[0].permute(1, 2, 0).cpu().numpy()))
            cf.append(np.clip(np.rint((0 if cut else 1) * mask * 255), 0, 255).astype(np.uint8))
        out.append(kf.update(tensor_image(x, "cuda"), ctx)[0].permute(1, 2, 0).cpu().numpy().astype(np.float16))
        prev = x
    for name, arr in (("lq", lq), ("gt", gt), ("ap", np.stack(ap)), ("cf", np.stack(cf)), ("kf", np.stack(out))):
        np.save(d / f"{name}.npy", arr)
    print("cached", scene, len(lq), flush=True)


def load(scene):
    d = CACHE / scene
    return {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in ("lq", "gt", "ap", "cf", "kf")}


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser(); ap_.add_argument("--force", action="store_true"); a = ap_.parse_args()
    for s, n in TRAIN.items(): build(s, n, a.force)
    for s in VAL: build(s, 30, a.force)
    # 测试场景的缓存只供最终评测脚本读取，训练/调参代码从不访问
    for s in TEST: build(s, 30, a.force)
    (CACHE / "params.json").write_text(json.dumps(kalman_params(), indent=2), encoding="utf-8")
