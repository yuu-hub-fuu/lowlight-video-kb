"""第二版统一评测：传统方法 / 深度基线 / KBG-Video 及其消融。

  --stage val   验证场景 pair7–12：比较全部方法，并按“验证 PSNR 最高”选择
                网格平滑系数 α 与（消融用的）像素融合系数 β，候选均为 {0.5,0.7,0.85,0.95}
  --stage test  读取冻结的 records/v2_config.json，在作者官方 12 个测试场景评测一次，保存示例帧
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import classical as C
from kalman import EMADenoiser
from kbg import KBGVideo, expose, exposure_model
from lowlight import (ROOT, ImageState, flow_context, image_numpy, load_model, quality, quantize,
                      reference_stats, save_rgb, tensor_image)
from splits import TEST, VAL
from train_trnet import load_arrays

CANDIDATES = [0.5, 0.7, 0.85, 0.95]
NLM_H = 15.0


def warp_error(prev_out, cur_out, gt_ctx):
    """用相邻“正常光参考帧”估计的光流把上一帧输出对齐到当前帧，
    在前后向一致的区域统计 |当前输出 − 对齐后的上一输出|。越小说明闪烁/抖动越少。"""
    grid, mask, _ = gt_ctx
    p = torch.from_numpy(np.ascontiguousarray(prev_out.transpose(2, 0, 1)))[None]
    w = F.grid_sample(p, torch.from_numpy(grid)[None], align_corners=True, padding_mode="border")[0].permute(1, 2, 0).numpy()
    m = (mask > 0.5).astype(np.float32)[..., None]
    return float((np.abs(cur_out - w) * m).sum() / max(m.sum() * 3, 1))


class Timer:
    def __init__(self, dev): self.dev = dev
    def __enter__(self): self.t = time.perf_counter(); return self
    def __exit__(self, *a):
        if str(self.dev).startswith("cuda"): torch.cuda.synchronize()
        self.ms = (time.perf_counter() - self.t) * 1000


def run(scenes, out_dir, alphas, betas, save_examples, device="auto"):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); cv2.setNumThreads(4); torch.set_grad_enabled(False)
    dce, dev = load_model(device)
    bg_raw = KBGVideo(0.0, "bgnet_lq", device, denoiser="none")
    kbg = {a: KBGVideo(a, "bgnet_kf", device) for a in alphas}
    kbg0 = KBGVideo(0.0, "bgnet_kf", device)
    kbg_nt = KBGVideo(0.0, "bgnet_kf", device, denoiser="kalman_no_test")
    c0, c1 = exposure_model()
    target_of = lambda img: float(np.exp(c0 + c1 * np.log(max(float(img.mean()), 1e-4))))
    rows = []
    for scene in scenes:
        lq8, gt8 = load_arrays(scene, 30)
        for e in (bg_raw, kbg0, kbg_nt, *kbg.values()): e.reset()
        ema = EMADenoiser(0.85)
        pix = {b: ImageState(b, True) for b in betas}
        old = ImageState(0.85, True)
        prev_rgb = gt_prev = None; last_out, last_err = {}, {}
        for i in range(len(lq8)):
            rgb = lq8[i].astype(np.float32) / 255; gt = gt8[i].astype(np.float32) / 255
            with Timer("cpu") as tf:
                ctx = flow_context(rgb, prev_rgb) if prev_rgb is not None else None
            flow_ms = tf.ms if ctx is not None else 0.0
            gt_ctx = flow_context(gt, gt_prev) if gt_prev is not None else None
            outs, times = {"暗光输入": rgb}, {"暗光输入": 0.0}

            def put(name, fn, dev_=dev, extra=0.0):
                with Timer(dev_) as t: r = fn()
                outs[name] = r; times[name] = t.ms + extra

            # —— 传统方法 ——
            put("幂律变换", lambda: C.power_law(rgb), "cpu")
            put("幂律变换+曝光回归", lambda: (lambda y: np.clip(y * target_of(rgb) / max(y.mean(), 1e-4), 0, 1))(C.power_law(rgb)), "cpu")
            put("直方图均衡", lambda: C.hist_equalize(rgb), "cpu")
            put("CLAHE", lambda: C.clahe(rgb), "cpu")
            put("多尺度Retinex", lambda: C.msr(rgb), "cpu")
            put("引导滤波Retinex+NLM", lambda: C.retinex_guided(rgb, NLM_H), "cpu")
            # —— 深度基线 ——
            x = tensor_image(rgb, dev)
            with torch.inference_mode():
                with Timer(dev) as t: base = dce(x)[1].clamp(0, 1)
                outs["Zero-DCE"], times["Zero-DCE"] = image_numpy(base), t.ms
                put("Zero-DCE+运动滤波(v1)", lambda: image_numpy(old.update(base, ctx)), extra=times["Zero-DCE"] + flow_ms)
            # —— KBG-Video 消融 ——
            put("BGNet(无去噪)", lambda: image_numpy(bg_raw.step(rgb, ctx)["out"]))
            def ema_bg():
                u = kbg0.net.lift(ema.update(x, ctx))
                return image_numpy(expose(kbg0.net.apply(u, kbg0.net.predict_grid(u)).clamp(0, 1), target_of(rgb)))
            put("EMA去噪+BGNet", ema_bg, extra=flow_ms)
            put("卡尔曼(无新息检验)+BGNet", lambda: image_numpy(kbg_nt.step(rgb, ctx)["out"]), extra=flow_ms)
            with Timer(dev) as t: r0 = kbg0.step(rgb, ctx)
            outs["卡尔曼+BGNet"], times["卡尔曼+BGNet"] = image_numpy(r0["out"]), t.ms + flow_ms
            with torch.inference_mode():
                for b, st in pix.items():
                    put(f"卡尔曼+BGNet+像素融合(β={b:g})", lambda: image_numpy(st.update(r0["out"], ctx)),
                        extra=times["卡尔曼+BGNet"])
            for a_, e in kbg.items():
                put(f"KBG-Video(α={a_:g})", lambda: image_numpy(e.step(rgb, ctx)["out"]), extra=flow_ms)
            stats = reference_stats(gt)
            for name, pred in outs.items():
                psnr, ssim = quality(pred, gt, stats)
                # GT-mean 指标：整体亮度按参考图均值对齐后再评，分离“曝光猜得准不准”与“增强质量”
                gm = np.clip(pred * gt.mean() / max(float(pred.mean()), 1e-4), 0, 1)
                psnr_gm, ssim_gm = quality(gm, gt, stats)
                q = quantize(pred).astype(np.float32) / 255; err = q - gt
                tce = float(np.mean(np.abs(err - last_err[name]))) if name in last_err else None
                we = warp_error(last_out[name], q, gt_ctx) if (name in last_out and gt_ctx is not None) else None
                last_err[name], last_out[name] = err, q
                rows.append(dict(scene=scene, index=i, method=name, psnr=psnr, ssim=ssim, psnr_gm=psnr_gm, ssim_gm=ssim_gm,
                                 tce=tce, warp_error=we,
                                 brightness_error=float(err.mean()), runtime_ms=times[name]))
                if save_examples and i in (0, 15, 29):
                    save_rgb(out_dir / "examples" / scene / f"{i:03d}_{name}.png", pred)
            if save_examples and i in (0, 15, 29):
                save_rgb(out_dir / "examples" / scene / f"{i:03d}_正常光参考.png", gt)
            prev_rgb, gt_prev = rgb, gt
        print("finished", scene, flush=True)
    with (out_dir / "per_frame.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows


def read_rows(path):
    rows = []
    for r in csv.DictReader(path.open(encoding="utf-8-sig")):
        for k in ("psnr", "ssim", "psnr_gm", "ssim_gm", "tce", "warp_error", "brightness_error", "runtime_ms"):
            r[k] = float(r[k]) if r[k] not in ("", "None") else None
        r["index"] = int(r["index"]); rows.append(r)
    return rows


def summarize(rows, scenes, out_dir):
    methods = list(dict.fromkeys(r["method"] for r in rows))
    seq = []
    for scene in scenes:
        for m in methods:
            rr = [r for r in rows if r["scene"] == scene and r["method"] == m]
            v = dict(scene=scene, method=m, frames=len(rr))
            for k in ("psnr", "ssim", "psnr_gm", "ssim_gm", "tce", "warp_error", "runtime_ms"):
                vals = [r[k] for r in rr if r[k] is not None]
                v[k] = float(np.mean(vals)) if vals else None
            v["brightness_flicker"] = float(np.std(np.diff([r["brightness_error"] for r in rr])))
            seq.append(v)
    keys = ["psnr", "ssim", "psnr_gm", "ssim_gm", "tce", "warp_error", "brightness_flicker", "runtime_ms"]
    summary = {m: {k: float(np.mean([r[k] for r in seq if r["method"] == m])) for k in keys} for m in methods}
    for fname, recs in (("per_frame.csv", rows), ("per_scene.csv", seq)):
        with (out_dir / fname).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
    (out_dir / "summary.json").write_text(json.dumps(dict(
        scenes=scenes, frames_per_scene=30, methods=summary,
        timing="单帧墙钟时间(ms)：含光流，不含指标与读写；RTX 4060 Laptop + CPU；传统方法在 CPU；仅供量级参考"),
        indent=2, ensure_ascii=False), encoding="utf-8")
    for m, v in summary.items():
        print(f"{m:34s} PSNR {v['psnr']:.3f} SSIM {v['ssim']:.4f} | GM {v['psnr_gm']:.3f}/{v['ssim_gm']:.4f} | WE {v['warp_error']:.5f} TCE {v['tce']:.5f} {v['runtime_ms']:.0f}ms", flush=True)
    return summary


def merged(stage, scenes, k):
    """--part 并行的各分片按场景顺序合并后统一汇总。"""
    out = ROOT / f"results_v2/{stage}"
    rows = [r for i in range(k) for r in read_rows(out / f"part{i}" / "per_frame.csv")]
    rows.sort(key=lambda r: (scenes.index(r["scene"]), r["index"]))
    return summarize(rows, scenes, out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["val", "test"], required=True)
    ap.add_argument("--part", default=None, help="i/k：只评测第 i 个分片（共 k 片），用于多进程并行")
    ap.add_argument("--merge", type=int, default=0, help="k：合并 k 个分片并汇总（val 阶段同时冻结配置）")
    a = ap.parse_args()
    cfg_path = ROOT / "records/v2_config.json"
    scenes = VAL if a.stage == "val" else TEST
    if a.part:
        i, k = map(int, a.part.split("/"))
        if a.stage == "val":
            run(scenes[i::k], ROOT / f"results_v2/val/part{i}", CANDIDATES, CANDIDATES, save_examples=False)
        else:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            run(scenes[i::k], ROOT / f"results_v2/test/part{i}", [cfg["alpha"]], [cfg["beta_pixel"]], save_examples=True)
        return
    if a.stage == "val":
        s = merged("val", VAL, a.merge) if a.merge else summarize(run(VAL, ROOT / "results_v2/val", CANDIDATES, CANDIDATES, False), VAL, ROOT / "results_v2/val")
        alpha_p = {c: s[f"KBG-Video(α={c:g})"]["psnr"] for c in CANDIDATES}
        beta_p = {c: s[f"卡尔曼+BGNet+像素融合(β={c:g})"]["psnr"] for c in CANDIDATES}
        cfg = dict(alpha=max(alpha_p, key=alpha_p.get), beta_pixel=max(beta_p, key=beta_p.get),
                   val_psnr_alpha=alpha_p, val_psnr_beta=beta_p, net="bgnet_kf",
                   kalman=json.loads((ROOT / "records/kalman_selection.json").read_text(encoding="utf-8"))["chosen"],
                   rule="alpha/beta = argmax validation scene-mean PSNR over {0.5,0.7,0.85,0.95}; frozen before reading test scenes",
                   val_scenes=VAL, test_scenes=TEST)
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        print("FROZEN", cfg)
    else:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        (ROOT / "records/v2_config_sha256.txt").write_text(hashlib.sha256(cfg_path.read_bytes()).hexdigest())
        if a.merge:
            merged("test", TEST, a.merge)
        else:
            summarize(run(TEST, ROOT / "results_v2/test", [cfg["alpha"]], [cfg["beta_pixel"]], True), TEST, ROOT / "results_v2/test")


if __name__ == "__main__":
    main()
