"""根据实际运行记录生成报告插图（所有数字都从 records/ 与 results_v2/ 读取，不手填）。"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lowlight import ROOT                               # noqa: E402

OUT = ROOT / "output/figures"; OUT.mkdir(parents=True, exist_ok=True)
OURS, BASE, OTHER, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#9a9a9a", "#1f1f1f", "#6b6b6b", "#e6e6e6"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False,
                     "font.size": 9, "axes.edgecolor": "#bdbdbd", "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200, "savefig.bbox": "tight"})


def jload(p): return json.loads((ROOT / p).read_text(encoding="utf-8"))


def fig_noise():
    d = jload("records/noise_model.json"); x = np.array(d["bins"]["x"]); v = np.array(d["bins"]["var"])
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.plot(x, v * 1e5, "o", ms=4, color=OTHER, label="实测（对齐帧差分的 MAD 估计）")
    xs = np.linspace(0, 0.4, 50); ax.plot(xs, (d["a"] * xs + d["b"]) * 1e5, color=OURS, lw=2, label=f"拟合 σ²=a·x+b（x<0.25）")
    ax.axvspan(0.25, 0.4, color="#f4f4f4", lw=0)
    ax.set_xlabel("暗光帧像素亮度 x"); ax.set_ylabel("噪声方差（×1e-5）"); ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"泊松-高斯噪声标定：a={d['a']:.2e}, b={d['b']:.2e}, κ={d['kappa']:.2f}", fontsize=9, color=INK)
    fig.savefig(OUT / "fig_noise.png"); plt.close(fig)


def fig_kalman():
    d = jload("records/kalman_selection.json")
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    g = d["grid"]; ax.scatter([r["ssim"] for r in g], [r["psnr"] for r in g], s=14, color=OTHER, label="卡尔曼参数网格")
    ema = [r for r in d["controls"] if r["kind"] == "ema"]
    ax.scatter([r["ssim"] for r in ema], [r["psnr"] for r in ema], s=22, marker="s", color=BASE, label="固定系数 EMA")
    for r in ema: ax.annotate(f"β={r['beta']}", (r["ssim"], r["psnr"]), fontsize=7, color=MUTED, xytext=(3, -8), textcoords="offset points")
    for r in d["controls"]:
        if r["kind"] == "kalman_iid_kappa":
            ax.scatter(r["ssim"], r["psnr"], s=26, marker="x", color=INK, label="κ=1/9（假设噪声独立）")
    c = d["chosen"]; ax.scatter(c["ssim"], c["psnr"], s=60, color=OURS, edgecolor="white", lw=1.5, zorder=5, label="选定配置")
    lo, hi = ax.get_ylim(); ax.set_ylim(lo - 0.004, hi)
    ax.set_xlabel("代理 SSIM（训练场景）"); ax.set_ylabel("代理 PSNR / dB"); ax.legend(frameon=False, fontsize=7)
    fig.savefig(OUT / "fig_kalman_selection.png"); plt.close(fig)


def fig_training():
    """长训练（4000/6000 步）的验证曲线：说明“训练越久验证越差”，以及曝光解耦的作用。"""
    runs = [("longrun_bgnet_kf_4000iters", "BGNet 曝光解耦损失", OURS, "-"),
            ("longrun_bgnet_kf_direct_4000iters", "BGNet 直接 L1 监督", BASE, "-"),
            ("train_trnet_2frame_14scenes", "TRNet 直接监督（14场景）", OTHER, "--")]
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    for f, lab, col, ls in runs:
        p = ROOT / f"records/{f}.json"
        if not p.exists(): continue
        log = json.loads(p.read_text(encoding="utf-8"))["log"]
        it = np.array([r["iter"] for r in log]); v = np.array([r["val_psnr"] for r in log])
        ax.plot(it, v, ls, color=col, lw=2, marker="o", ms=3, label=lab)
    ax.set_xlabel("训练步数"); ax.set_ylabel("验证集 PSNR / dB"); ax.legend(frameon=False, fontsize=7.5)
    fig.savefig(OUT / "fig_training.png"); plt.close(fig)


def fig_tone_curves():
    """把学到的双边网格“读出来”：同一张测试帧，三个位置上灰度输入 r → 输出亮度的映射曲线。"""
    from bgnet import BGNet
    from tools.build_cache import load
    cfg = jload("records/train_bgnet_kf.json")
    net = BGNet(depth=cfg["depth"]); net.load_state_dict(torch.load(ROOT / "weights/bgnet_kf.pth", map_location="cpu", weights_only=True)); net.eval()
    scene = jload("records/v2_config.json")["test_scenes"][0]
    c = load(scene); x = torch.from_numpy(np.asarray(c["kf"][15], np.float32)).permute(2, 0, 1)[None]
    with torch.no_grad():
        u = net.lift(x); grid = net.predict_grid(u)
        r = torch.linspace(0, 0.3, 64)
        fig, ax = plt.subplots(figsize=(4.2, 2.6))
        ax.plot(r, np.clip(1.5 * r.numpy() ** 0.4, 0, 1), "--", color=OTHER, lw=1.5, label="传统幂律 1.5·r^0.4")
        for (py, px), lab, col in [((0.25, 0.25), "左上区域", OURS), ((0.75, 0.75), "右下区域", BASE)]:
            # 在固定空间位置 (px,py) 依次放入 64 个灰度值，读取网格在该位置给出的变换
            patch = r.view(1, 1, 64, 1).repeat(1, 3, 1, 1).pow(0.4)
            g = net.guide_map(patch)                                                 # [1,1,64,1]
            coords = torch.stack([torch.full((64,), 2 * px - 1), torch.full((64,), 2 * py - 1), g[0, 0, :, 0] * 2 - 1], -1).view(1, 1, 64, 1, 3)
            coef = torch.nn.functional.grid_sample(grid, coords, align_corners=True, padding_mode="border")[0, :, 0, :, 0]  # [12,64]
            A = coef.view(3, 4, 64); uu = patch[0, :, :, 0]
            out = (A[:, :3] * uu[None]).sum(1) + A[:, 3]
            ax.plot(r, out.mean(0).clamp(0, 1).numpy(), color=col, lw=2, label=f"学到的映射：{lab}")
    ax.set_xlabel("暗光输入亮度 r"); ax.set_ylabel("输出亮度"); ax.legend(frameon=False, fontsize=7.5)
    ax.set_title(f"双边网格在不同位置学到的色调曲线（{scene} 第16帧）", fontsize=9, color=INK)
    fig.savefig(OUT / "fig_tone_curves.png"); plt.close(fig)


def per_scene(split):
    rows = list(csv.DictReader((ROOT / f"results_v2/{split}/per_scene.csv").open(encoding="utf-8-sig")))
    return rows


def fig_test_bars():
    s = jload("results_v2/test/summary.json")["methods"]
    ours = jload("records/v2_config.json")["final_method"]
    names = ["幂律变换", "CLAHE", "多尺度Retinex", "引导滤波Retinex+NLM", "Zero-DCE", "Zero-DCE+运动滤波(v1)", ours]
    labels = ["幂律变换", "CLAHE", "多尺度Retinex", "Retinex+NLM", "Zero-DCE", "v1(旧版)", "KBG-Video"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, key, title, fmt in zip(axes, ["psnr", "ssim", "warp_error"], ["PSNR↑ / dB", "SSIM↑", "扭曲误差↓"], ["{:.2f}", "{:.3f}", "{:.4f}"]):
        vals = [s[n][key] for n in names]
        cols = [OURS if n == ours else BASE if n == "Zero-DCE" else OTHER for n in names]
        ax.barh(range(len(names)), vals, color=cols, height=0.62)
        for i, v in enumerate(vals): ax.text(v, i, " " + fmt.format(v), va="center", fontsize=7, color=INK)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(labels if ax is axes[0] else [""] * len(names), fontsize=7.5)
        ax.set_title(title, fontsize=9, color=INK); ax.grid(axis="y", visible=False); ax.invert_yaxis()
        ax.set_xlim(0, max(vals) * 1.28)
    fig.savefig(OUT / "fig_test_bars.png"); plt.close(fig)


def fig_per_scene():
    ours = jload("records/v2_config.json")["final_method"]
    rows = per_scene("test"); scenes = list(dict.fromkeys(r["scene"] for r in rows))
    get = lambda m: [float(next(r["psnr"] for r in rows if r["scene"] == s and r["method"] == m)) for s in scenes]
    a, b = get("Zero-DCE"), get(ours)
    fig, ax = plt.subplots(figsize=(6.4, 2.4)); x = np.arange(len(scenes))
    ax.vlines(x, a, b, color=GRID, lw=3)
    ax.scatter(x, a, color=BASE, s=22, label="Zero-DCE", zorder=3); ax.scatter(x, b, color=OURS, s=22, label="KBG-Video", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(scenes, fontsize=7, rotation=0); ax.set_ylabel("PSNR / dB")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left"); ax.grid(axis="x", visible=False)
    fig.savefig(OUT / "fig_per_scene.png"); plt.close(fig)


if __name__ == "__main__":
    fig_noise(); fig_kalman()
    if (ROOT / "records/train_bgnet_kf.json").exists(): fig_training()
    if (ROOT / "records/v2_config.json").exists(): fig_tone_curves()
    if (ROOT / "results_v2/test/summary.json").exists(): fig_test_bars(); fig_per_scene()
    print(sorted(p.name for p in OUT.iterdir()))
