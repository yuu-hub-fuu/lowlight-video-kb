"""训练 TRNet（两帧/单帧两种设置），严格的场景级划分。

划分见 splits.py：训练 34 个训练池场景；验证 pair7–12；测试为作者官方 12 个室内测试场景，训练期间从不读取。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lowlight import ROOT, flow_context, quality, quantize
from vnet import TRNet, align_prev, gaussian_window, ssim_torch

from splits import TEST, TRAIN, VAL  # noqa: E402,F401  场景划分统一放在 splits.py


def scene_files(scene, frames):
    root = ROOT / "data/raw/indoor_np"
    lq, gt = sorted((root / "input" / scene).glob("*.npy")), sorted((root / "GT" / scene).glob("*.npy"))
    if len(lq) < frames or len(gt) < frames:
        raise RuntimeError(f"{scene}: need {frames} frames, have {len(lq)}/{len(gt)}")
    return lq[:frames], gt[:frames]      # 与作者代码一致：各自排序后按位置配对


def load_arrays(scene, frames):
    lq, gt = scene_files(scene, frames)
    f = lambda p: np.load(p)[..., ::-1].copy()          # BGR -> RGB uint8
    return np.stack([f(p) for p in lq]), np.stack([f(p) for p in gt])


def align_sequence(lq_u8, device):
    """逐帧计算 (对齐后的上一帧, 置信度)。第0帧用自身、置信度0。"""
    aligned, conf = [], []
    prev = None
    for x8 in lq_u8:
        x = x8.astype(np.float32) / 255
        if prev is None:
            aligned.append(x8); conf.append(np.zeros(x8.shape[:2], np.uint8))
        else:
            grid, mask, cut = flow_context(x, prev)
            a = align_prev(prev, grid, device)[0].permute(1, 2, 0).cpu().numpy()
            aligned.append(quantize(a)); conf.append(np.clip(np.rint((0 if cut else 1) * mask * 255), 0, 255).astype(np.uint8))
        prev = x
    return np.stack(aligned), np.stack(conf)


def to_t(a, device):
    return torch.from_numpy(a).to(device).permute(0, 3, 1, 2).float() / 255


def evaluate_val(net, val_data, device, stride=3):
    net.eval(); ps = []
    with torch.inference_mode():
        for lq, gt, ap, cf in val_data:
            for i in range(0, len(lq), stride):
                cur, prev = to_t(np.asarray(lq[i:i+1]), device), to_t(np.asarray(ap[i:i+1]), device)
                conf = torch.from_numpy(np.asarray(cf[i:i+1])).to(device)[:, None].float() / 255
                out = net(cur, prev, conf).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
                ps.append(quality(out, gt[i].astype(np.float32) / 255)[0])
    net.train()
    return float(np.mean(ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temporal", type=int, default=1)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--crop", type=int, nargs=2, default=[256, 448])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=250)
    a = ap.parse_args()
    name = a.name or ("trnet_2frame" if a.temporal else "trnet_1frame")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda"
    t0 = time.time()
    from tools.build_cache import load            # 内存映射读取预计算的对齐帧/置信度
    as_tuple = lambda c: (c["lq"], c["gt"], c["ap"], c["cf"])
    train = [as_tuple(load(s)) for s in TRAIN]
    val = [as_tuple(load(s)) for s in VAL]
    frames = [(si, fi) for si, d in enumerate(train) for fi in range(len(d[0]))]
    scene_of = np.array([si for si, _ in frames]); prob = 1 / np.bincount(scene_of)[scene_of]; prob /= prob.sum()
    net = TRNet(temporal=bool(a.temporal)).to(dev)
    print("params", sum(p.numel() for p in net.parameters()), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.iters, pct_start=0.05)
    win = gaussian_window(dev)
    ch, cw = a.crop; best = (-1, 0); log = []
    for it in range(1, a.iters + 1):
        xs, ps, cs, ys = [], [], [], []
        for _ in range(a.batch):
            si, fi = frames[np.random.choice(len(frames), p=prob)]   # 场景等权采样，与 BGNet 一致
            lq, gt, ap_, cf = train[si]
            y0, x0 = np.random.randint(0, 512 - ch + 1), np.random.randint(0, 960 - cw + 1)
            sl = (fi, slice(y0, y0 + ch), slice(x0, x0 + cw))
            tile = [lq[sl], ap_[sl], cf[sl][..., None], gt[sl]]
            if np.random.rand() < .5: tile = [t[:, ::-1] for t in tile]
            if np.random.rand() < .5: tile = [t[::-1] for t in tile]
            xs.append(tile[0]); ps.append(tile[1]); cs.append(tile[2]); ys.append(tile[3])
        x, p, c, y = (to_t(np.ascontiguousarray(np.stack(v)), dev) for v in (xs, ps, cs, ys))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(x, p, c)
        out = out.float()
        loss = F.l1_loss(out, y) + 0.2 * (1 - ssim_torch(out.clamp(0, 1), y, win))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if it % 100 == 0: print(f"it {it} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
        if it % a.val_every == 0 or it == a.iters:
            v = evaluate_val(net, val, dev); log.append(dict(iter=it, val_psnr=v, loss=float(loss.item())))
            print(f"== val PSNR {v:.3f} (best {best[0]:.3f})", flush=True)
            if v > best[0]:
                best = (v, it); torch.save(net.state_dict(), ROOT / f"weights/{name}.pth")
    (ROOT / f"records/train_{name}.json").write_text(json.dumps(dict(
        name=name, temporal=bool(a.temporal), iters=a.iters, batch=a.batch, crop=a.crop, lr=a.lr, seed=a.seed,
        train_scenes=TRAIN, val_scenes=VAL, best_val_psnr=best[0], best_iter=best[1], log=log,
        params=sum(p.numel() for p in net.parameters()), seconds=time.time() - t0), indent=2), encoding="utf-8")
    print("DONE best", best, flush=True)


if __name__ == "__main__":
    main()
