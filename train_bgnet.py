"""训练双边网格网络 BGNet（第②级）。

--input kf  输入为卡尔曼去噪后的暗光帧（完整方案）
--input lq  输入为原始暗光帧（消融：去掉第①级）
训练 34 个训练池场景，验证 pair7–12 早停；官方测试场景不参与。
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from bgnet import BGNet
from lowlight import ROOT, quality
from splits import TRAIN, VAL
from tools.build_cache import load
from vnet import gaussian_window, ssim_torch


def exposure_target(lq_mean: float) -> float:
    """曝光回归（tools/fit_exposure.py，只用训练场景拟合）：μ_gt = exp(c0 + c1·log μ_in)。"""
    e = json.loads((ROOT / "records/exposure_model.json").read_text(encoding="utf-8"))
    return float(np.exp(e["c0"] + e["c1"] * np.log(max(lq_mean, 1e-4))))


def expose(out, target):
    """把网络输出整体缩放到目标平均亮度（单个标量，不改变色调形状与颜色比例）。"""
    return (out * (target / out.mean().clamp_min(1e-4))).clamp(0, 1)


def frame_tensor(arr, idx, device):
    a = np.asarray(arr[idx], dtype=np.float32)
    if arr.dtype == np.uint8: a = a / 255
    return torch.from_numpy(a).to(device).permute(2, 0, 1)[None]


@torch.no_grad()
def validate(net, val, key, device, decoupled, stride=3):
    """按真实推理流程验证：解耦模式下输出先经曝光回归缩放。"""
    net.eval(); ps, gms, ss = [], [], []
    for c in val:
        for i in range(0, len(c["lq"]), stride):
            out = net(frame_tensor(c[key], i, device)).clamp(0, 1)
            if decoupled:
                out = expose(out, exposure_target(float(np.asarray(c["lq"][i]).mean()) / 255))
            out = out[0].permute(1, 2, 0).cpu().numpy(); gt = np.asarray(c["gt"][i], np.float32) / 255
            p, s = quality(out, gt); ps.append(p); ss.append(s)
            gms.append(quality(np.clip(out * gt.mean() / max(out.mean(), 1e-4), 0, 1), gt)[0])
    net.train()
    return float(np.mean(ps)), float(np.mean(ss)), float(np.mean(gms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", choices=["kf", "lq"], default="kf")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", choices=["decoupled", "direct"], default="decoupled")
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    name = a.name or f"bgnet_{a.input}" + ("" if a.loss == "decoupled" else "_direct")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda"; t0 = time.time()
    train = [load(s) for s in TRAIN]; val = [load(s) for s in VAL]
    index = [(si, fi) for si, c in enumerate(train) for fi in range(len(c["lq"]))]
    # 场景等权采样：每个场景被抽到的概率相同，避免帧多的场景主导曝光
    scene_of = np.array([si for si, _ in index]); counts = np.bincount(scene_of)
    prob = 1 / counts[scene_of]; prob /= prob.sum()
    net = BGNet(depth=a.depth).to(dev)
    nparams = sum(p.numel() for p in net.parameters())
    print("frames", len(index), "scenes", len(train), "params", nparams, flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.iters, pct_start=0.05)
    win = gaussian_window(dev); best = (-1.0, 0); log = []
    for it in range(1, a.iters + 1):
        xs, ys = [], []
        for j in np.random.choice(len(index), a.batch, p=prob):
            si, fi = index[j]
            x, y = frame_tensor(train[si][a.input], fi, dev), frame_tensor(train[si]["gt"], fi, dev)
            if np.random.rand() < .5: x, y = x.flip(-1), y.flip(-1)
            if np.random.rand() < .5: x, y = x.flip(-2), y.flip(-2)
            xs.append(x); ys.append(y)
        x, y = torch.cat(xs), torch.cat(ys)
        out = net(x)
        if a.loss == "decoupled":
            # 曝光归一化损失：每个样本先按全局均值对齐到参考图（缩放系数不回传梯度），
            # 网络只需学习色调形状、颜色与局部对比；小权重的原始 L1 防止整体亮度漂移
            s = (y.mean((1, 2, 3), keepdim=True) / out.mean((1, 2, 3), keepdim=True).clamp_min(1e-4)).detach()
            on = (out * s).clamp(0, 1)
            loss = F.l1_loss(on, y) + 0.2 * (1 - ssim_torch(on, y, win)) + 0.05 * F.l1_loss(out, y)
        else:
            loss = F.l1_loss(out, y) + 0.2 * (1 - ssim_torch(out.clamp(0, 1), y, win))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if it % 100 == 0: print(f"it {it} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
        if it % a.val_every == 0 or it == a.iters:
            v, vs, vg = validate(net, val, a.input, dev, a.loss == "decoupled")
            log.append(dict(iter=it, val_psnr=v, val_ssim=vs, val_psnr_gtmean=vg, loss=float(loss.item())))
            print(f"== val PSNR {v:.3f} SSIM {vs:.4f} GT-mean PSNR {vg:.3f} (best {best[0]:.3f})", flush=True)
            torch.save(net.state_dict(), ROOT / f"weights/{name}_last.pth")
            if v > best[0]:
                best = (v, it); torch.save(net.state_dict(), ROOT / f"weights/{name}.pth")
    (ROOT / f"records/train_{name}.json").write_text(json.dumps(dict(
        name=name, input=a.input, loss=a.loss, depth=a.depth, iters=a.iters, val_every=a.val_every, batch=a.batch, lr=a.lr, seed=a.seed, params=nparams,
        train_scenes=TRAIN, val_scenes=VAL, best_val_psnr=best[0], best_iter=best[1], log=log,
        seconds=time.time() - t0), indent=2), encoding="utf-8")
    print("DONE best", best, flush=True)


if __name__ == "__main__":
    main()
