"""讲解版演示视频：分镜 + 中文配音(Windows 本地语音 Huihui) + 烧录字幕。

所有画面来自真实 SDSD 帧与本项目实际输出，所有数字从 results_v2/ 与 records/ 读取并在生成前核对。
用法：python make_presentation.py [--data <含 data/raw/indoor_np 的目录>]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import tempfile
import wave
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from kbg import KBGVideo
from lowlight import ROOT, image_numpy, load_model, quantize, tensor_image

W, H, FPS = 1280, 720, 25
BG, INK, MUTED, CARD = (17, 20, 24), (236, 238, 241), (150, 156, 165), (30, 34, 40)
BLUE, ORANGE, AQUA = (57, 135, 229), (217, 89, 38), (25, 158, 112)
FONT = "C:/Windows/Fonts/msyh.ttc"; FONT_B = "C:/Windows/Fonts/msyhbd.ttc"
F = lambda s, b=False: ImageFont.truetype(FONT_B if b else FONT, s)
VOICE = "Microsoft Huihui Desktop"


# ———————————————————— 数据 ————————————————————
def load_scene(data, scene, n=30):
    # 样例帧留在交付目录、其余帧在工作目录：两处合并后按文件名排序
    files = lambda kind: sorted({q.name: q for r_ in (ROOT, data) for q in (r_ / "data/raw/indoor_np" / kind / scene).glob("*.npy")}.items())
    lq = [q for _, q in files("input")][:n]; gt = [q for _, q in files("GT")][:n]
    assert len(lq) == n and len(gt) == n, (scene, len(lq), len(gt))
    f = lambda p: np.load(p)[..., ::-1].astype(np.float32) / 255
    return [f(p) for p in lq], [f(p) for p in gt]


@torch.inference_mode()
def run_scene(data, scene, dce, kbg, dev):
    lq, gt = load_scene(data, scene); kbg.reset(); rec = {k: [] for k in ("inp", "dce", "ours", "gt", "den", "K", "single")}
    for x, g in zip(lq, gt):
        r = kbg.step(x)
        rec["inp"].append(quantize(x)); rec["gt"].append(quantize(g))
        rec["dce"].append(quantize(image_numpy(dce(tensor_image(x, dev))[1])))
        rec["ours"].append(quantize(image_numpy(r["out"]))); rec["single"].append(quantize(image_numpy(r["raw"])))
        rec["den"].append(quantize(np.clip(image_numpy(r["denoised"]) * 4, 0, 1)))
        K = kbg.den.last_gain.mean(1)[0].cpu().numpy()
        rec["K"].append(cv2.applyColorMap(quantize(np.clip(K, 0, 1)), cv2.COLORMAP_VIRIDIS)[..., ::-1].copy())
    return rec


def pingpong(i, n):
    p = i % (2 * n - 2); return p if p < n else 2 * n - 2 - p


# ———————————————————— 绘制工具 ————————————————————
def canvas(): return Image.new("RGB", (W, H), BG)


def text(d, xy, s, size=26, fill=INK, bold=False, anchor="la"):
    d.text(xy, s, font=F(size, bold), fill=fill, anchor=anchor)


def fit(img, w, h):
    im = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    r = min(w / im.width, h / im.height)
    return im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))), Image.Resampling.LANCZOS)


def panel(cv, img, x, y, w, h, label=None, color=MUTED):
    d = ImageDraw.Draw(cv)
    top = y + (26 if label else 0)
    im = fit(img, w, h - (26 if label else 0)); cv.paste(im, (x + (w - im.width) // 2, top))
    if label: text(d, (x + w // 2, y + 2), label, 18, color, anchor="ma")
    return (x + (w - im.width) // 2, top, im.width, im.height)


def header(cv, sec, title):
    d = ImageDraw.Draw(cv)
    text(d, (48, 30), sec, 18, BLUE, True); text(d, (48, 54), title, 30, INK, True)


def subtitle(cv, s):
    if not s: return
    d = ImageDraw.Draw(cv, "RGBA")
    f = F(26); lines, cur = [], ""
    for ch in s:
        if d.textlength(cur + ch, font=f) > W - 160: lines.append(cur); cur = ch
        else: cur += ch
    lines.append(cur)
    hh = 40 * len(lines) + 16
    d.rectangle([0, H - hh - 8, W, H], fill=(0, 0, 0, 175))
    for k, ln in enumerate(lines): d.text((W // 2, H - hh + 2 + 40 * k), ln, font=f, fill=(255, 255, 255), anchor="ma")


def fade(img, a):
    return Image.blend(Image.new("RGB", img.size, BG), img, max(0.0, min(1.0, a)))


def box(d, xy, label, on, color=BLUE, sub=None):
    x0, y0, x1, y1 = xy
    d.rounded_rectangle(xy, 14, fill=CARD if on else BG, outline=color if on else (60, 64, 70), width=3)
    text(d, ((x0 + x1) // 2, (y0 + y1) // 2 - (12 if sub else 0)), label, 24, INK if on else (90, 94, 100), True, "mm")
    if sub: text(d, ((x0 + x1) // 2, (y0 + y1) // 2 + 20), sub, 16, MUTED if on else (70, 74, 80), anchor="mm")


def arrow(d, a, b, on):
    c = (150, 156, 165) if on else (60, 64, 70)
    d.line([a, b], fill=c, width=3)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    for s in (2.6, -2.6):
        d.line([b, (b[0] + 12 * math.cos(ang + s), b[1] + 12 * math.sin(ang + s))], fill=c, width=3)


def figure(path, w, h):
    im = Image.open(path).convert("RGB"); im = fit(im, w - 24, h - 24)
    card = Image.new("RGB", (im.width + 24, im.height + 24), (255, 255, 255)); card.paste(im, (12, 12)); return card


# ———————————————————— 配音 ————————————————————
def tts(text_, wav):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as t: t.write(text_)
    ps = (f"Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"$s.SelectVoice('{VOICE}'); $s.Rate=1; "
          f"$fmt=New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050,[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,[System.Speech.AudioFormat.AudioChannel]::Mono); "
          f"$s.SetOutputToWaveFile('{wav}',$fmt); $s.Speak([IO.File]::ReadAllText('{t.name}',[Text.Encoding]::UTF8)); $s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    with wave.open(str(wav)) as w_: return w_.readframes(w_.getnframes()), w_.getnframes() / w_.getframerate()


def sentences(s):
    return [x for x in re.split(r"(?<=[。！？；])", s) if x.strip()]


# ———————————————————— 主程序 ————————————————————
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT.parents[2] / "_工作数据/lowlight_video_project")
    ap.add_argument("--out", type=Path, default=ROOT / "output/KBG_讲解演示视频.mp4")
    a = ap.parse_args()
    data = a.data if (a.data / "data/raw/indoor_np/input/pair21").exists() else ROOT
    T = json.loads((ROOT / "results_v2/test/summary.json").read_text(encoding="utf-8"))["methods"]
    cfg = json.loads((ROOT / "records/v2_config.json").read_text(encoding="utf-8")); OURS = cfg["final_method"]
    o, z, v1 = T[OURS], T["Zero-DCE"], T["Zero-DCE+运动滤波(v1)"]
    we_drop = round((1 - o["warp_error"] / z["warp_error"]) * 100); dps = z["psnr"] - o["psnr"]
    # 旁白中的数字与实验记录逐一核对
    assert f"{z['ssim']:.2f}" == "0.69" and f"{o['ssim']:.2f}" == "0.77" and we_drop == 84 and f"{dps:.2f}" == "0.56", (z, o)
    assert o["psnr"] < v1["psnr"] and o["ssim"] < v1["ssim"]

    torch.set_grad_enabled(False)
    dce, dev = load_model(); kbg = KBGVideo.final(dev)
    S1, S2 = "pair13", "pair21"
    clip = {s: run_scene(data, s, dce, kbg, dev) for s in (S1, S2)}
    print("clips ready", flush=True)
    thumbs = []
    for s in ["pair13", "pair21", "pair31", "pair50", "pair58", "pair68"]:
        lq, gt = load_scene(data, s, 1); thumbs.append((s, quantize(lq[0]), quantize(gt[0])))

    # 局部放大区域：Zero-DCE 与本文高频能量差最大的 150×150 窗口（自动确定，不手挑）
    c = clip[S1]; i0 = 15
    hp = lambda im: np.abs(im.astype(np.float32) - cv2.GaussianBlur(im.astype(np.float32), (0, 0), 2)).mean(-1)
    diff = cv2.blur(hp(c["dce"][i0]) - hp(c["ours"][i0]), (150, 150))
    yy, xx = np.unravel_index(np.argmax(diff[75:-75, 75:-75]), diff[75:-75, 75:-75].shape); zy, zx = yy, xx
    # 闪烁曲线区域：参考帧在时间上最稳定且亮度适中的 64×64 块
    g = np.stack(c["gt"]).astype(np.float32).mean(-1); stdm = cv2.blur(g.std(0), (64, 64)); mean_ = cv2.blur(g.mean(0), (64, 64))
    score = np.where((mean_ > 60) & (mean_ < 180), stdm, 1e9)[32:-32, 32:-32]; py, px = np.unravel_index(np.argmin(score), score.shape)
    curve = {k: np.array([float(f[py:py + 64, px:px + 64].mean()) for f in c[k]]) for k in ("dce", "ours", "gt")}
    curve = {k: list((v / v.mean() - 1) * 100) for k, v in curve.items()}          # 相对自身均值的波动(%)，三条线同一尺度
    print("patch flicker std%", {k: round(float(np.std(v)), 3) for k, v in curve.items()}, flush=True)

    def play(rec, key, t, fps=10): return rec[key][pingpong(int(t * fps), len(rec[key]))]

    # —— 分镜 ——
    def s_title(cv, t, D):
        d = ImageDraw.Draw(cv); bgimg = fit(play(clip[S2], "ours", t), W, H)
        cv.paste(Image.blend(Image.new("RGB", bgimg.size, BG), bgimg, 0.25), (0, (H - bgimg.height) // 2))
        text(d, (W // 2, 250), "低光照视频增强", 58, INK, True, "mm")
        text(d, (W // 2, 330), "噪声建模卡尔曼去噪 · 可学习双边网格 · 运动补偿融合", 28, MUTED, anchor="mm")
        text(d, (W // 2, 400), "KBG-Video", 40, BLUE, True, "mm")
        text(d, (W // 2, 470), "计算机图像处理课程项目", 22, MUTED, anchor="mm")

    def s_problem(cv, t, D):
        header(cv, "问题", "暗光视频难在哪里"); d = ImageDraw.Draw(cv)
        panel(cv, play(clip[S1], "inp", t), 48, 110, 580, 330, "原始暗光帧（几乎全黑）")
        x = play(clip[S1], "inp", t).astype(np.float32) * 5
        panel(cv, np.clip(x, 0, 255).astype(np.uint8), 652, 110, 580, 330, "直接提亮 5 倍：噪声被一起放大", ORANGE)
        items = ["① 噪声放大：提亮 5 倍，噪声也放大 5 倍", "② 时域闪烁 / 拖影：逐帧处理会闪，直接平均会拖影", "③ 曝光不确定：参考图“该多亮”无法完全由输入决定"]
        for k, s in enumerate(items):
            if t > D * (0.25 + 0.22 * k): text(d, (90, 470 + 42 * k), s, 24, INK)

    def s_data(cv, t, D):
        header(cv, "数据", "SDSD：机械臂重复运动采集的真实配对视频"); d = ImageDraw.Draw(cv)
        for k, (s, lq, gt) in enumerate(thumbs):
            x, y = 48 + (k % 3) * 300, 112 + (k // 3) * 200
            if t < D * 0.12 * (k + 1): continue
            cv.paste(fit(gt, 290, 155), (x, y)); cv.paste(fit(np.clip(lq.astype(np.float32) * 4, 0, 255).astype(np.uint8), 110, 60), (x + 4, y + 4))
            text(d, (x, y + 160), f"{s}  左上角：暗光×4", 15, MUTED)
        rows = [("训练", "34 个场景，740 帧", AQUA), ("验证", "6 个场景，早停与选参", ORANGE), ("测试", "作者官方 12 个场景，只评一次", BLUE)]
        for k, (a_, b_, col) in enumerate(rows):
            y = 140 + 92 * k; d.rounded_rectangle([970, y, 1235, y + 74], 12, fill=CARD, outline=col, width=2)
            text(d, (990, y + 10), a_, 22, col, True); text(d, (990, y + 42), b_, 16, INK)

    def s_pipeline(cv, t, D):
        header(cv, "方法", "四步在线流水线：只用当前帧与过去帧"); d = ImageDraw.Draw(cv)
        names = [("暗光帧", ""), ("① 卡尔曼去噪", "噪声模型 + 光流"), ("② 双边网格", "学习的分段色调变换"),
                 ("③ 曝光回归", "两参数，决定多亮"), ("④ 运动补偿融合", "置信度门控"), ("输出", "")]
        xs = [40, 245, 450, 655, 860, 1065]; k_on = int(t / D * 7)
        for k, ((nm, sb), x) in enumerate(zip(names, xs)):
            box(d, (x, 300, x + 175, 400), nm, k <= k_on, BLUE if 0 < k < 5 else MUTED, sb or None)
            if k: arrow(d, (xs[k - 1] + 178, 350), (x - 4, 350), k <= k_on)
        d.line([(132, 400), (132, 470), (947, 470), (947, 404)], fill=(150, 156, 165) if k_on >= 4 else (60, 64, 70), width=2)
        text(d, (540, 478), "光流（暗光帧之间，Farneback + 前后向一致性）", 18, MUTED, anchor="ma")
        if k_on >= 6: text(d, (W // 2, 560), "课程知识点：噪声模型 · 卡尔曼/维纳滤波 · 灰度变换 · 双边滤波 · 运动估计 · 深度学习增强", 20, AQUA, anchor="mm")

    def s_kalman(cv, t, D):
        header(cv, "① 卡尔曼去噪", "在提亮之前、沿运动轨迹做最优估计")
        cv.paste(figure(ROOT / "output/figures/fig_noise.png", 470, 340), (40, 120))
        d = ImageDraw.Draw(cv); text(d, (48, 480), "K = P' / (P' + R)，R = a·x + b", 22, INK)
        text(d, (48, 515), "新息过大（遮挡/运动）→ K→1，退回当前帧", 20, MUTED)
        r = clip[S1]
        panel(cv, np.clip(play(r, "inp", t).astype(np.float32) * 4, 0, 255).astype(np.uint8), 540, 110, 360, 230, "输入 ×4（噪声明显）")
        panel(cv, play(r, "den", t), 905, 110, 360, 230, "卡尔曼去噪后 ×4", BLUE)
        panel(cv, play(r, "K", t), 540, 350, 360, 230, "增益 K：越暗 = 越多用历史帧")
        panel(cv, play(r, "ours", t), 905, 350, 360, 230, "最终输出", BLUE)

    def s_grid(cv, t, D):
        header(cv, "② 双边网格网络", "看缩略图 → 预测 16×16×8 个颜色变换 → 按像素切片")
        cv.paste(figure(ROOT / "output/figures/fig_tone_curves.png", 520, 380), (40, 120))
        r = clip[S2]; x = play(r, "inp", t).astype(np.float32) / 255
        pw = np.clip(1.5 * x ** 0.4, 0, 1)
        panel(cv, quantize(pw), 600, 110, 640, 235, "传统幂律变换 s = 1.5·r^0.4（网络的初始化）")
        panel(cv, play(r, "single", t), 600, 360, 640, 235, "卡尔曼 + 双边网格 + 曝光回归（单帧）", BLUE)

    def s_expo(cv, t, D):
        header(cv, "③④ 曝光回归 + 运动补偿融合", "网络训练越久越会“死记”亮度 → 把曝光单独交给回归")
        cv.paste(figure(ROOT / "output/figures/fig_training.png", 600, 400), (40, 120)); d = ImageDraw.Draw(cv)
        text(d, (680, 150), "③ 曝光回归（训练场景留一交叉验证）", 22, BLUE, True)
        text(d, (680, 190), "log μ* = 0.258 + 0.455·log μ_in", 24, INK)
        text(d, (680, 228), "斜率≈0.45：参考亮度 ≈ 输入亮度的 γ=0.45 次幂", 18, MUTED)
        text(d, (680, 300), "④ 运动补偿置信度融合", 22, BLUE, True)
        text(d, (680, 340), "out = (1 − β·c)·y + β·c·W(out_prev)", 24, INK)
        text(d, (680, 378), "c：光流置信度；β = 0.85（验证集选定）", 18, MUTED)
        text(d, (680, 450), "对照：平滑网格系数（方案 A）几乎无效", 20, ORANGE)
        text(d, (680, 482), "→ 残余闪烁来自像素级噪声，按验证集选方案 B", 18, MUTED)

    def s_slider(cv, t, D):
        header(cv, "效果对比", "测试场景：左 Zero-DCE ｜ 右 本文 KBG-Video"); d = ImageDraw.Draw(cv)
        r = clip[S1]; i = pingpong(int(t * 8), 30); a_, b_ = r["dce"][i], r["ours"][i]
        sx = int(960 * (0.5 + 0.38 * math.sin(2 * math.pi * t / 7)))
        comp = b_.copy(); comp[:, :sx] = a_[:, :sx]
        x0, y0, w0, h0 = panel(cv, comp, 40, 110, 800, 440)
        lx = x0 + int(sx * w0 / 960); d.line([(lx, y0), (lx, y0 + h0)], fill=(255, 255, 255), width=3)
        text(d, (x0 + 10, y0 + 10), "Zero-DCE", 22, ORANGE, True); text(d, (x0 + w0 - 10, y0 + 10), "KBG-Video", 22, BLUE, True, "ra")
        s_ = w0 / 960; d.rectangle([x0 + zx * s_, y0 + zy * s_, x0 + (zx + 150) * s_, y0 + (zy + 150) * s_], outline=(255, 214, 0), width=2)
        for k, (key, lab, col) in enumerate([("dce", "Zero-DCE 放大", ORANGE), ("ours", "本文 放大", BLUE)]):
            crop = r[key][i][zy:zy + 150, zx:zx + 150]
            panel(cv, cv2.resize(crop, (360, 360), interpolation=cv2.INTER_NEAREST), 870, 100 + 232 * k, 370, 222, lab, col)

    def s_flicker(cv, t, D):
        header(cv, "时间稳定性", "同一块静止区域的亮度随时间变化"); d = ImageDraw.Draw(cv)
        r = clip[S1]; i = min(29, int(t / D * 1.15 * 30))
        x0, y0, w0, h0 = panel(cv, r["ours"][i], 40, 110, 560, 300, "本文输出（黄框为统计区域）", BLUE)
        s_ = w0 / 960; d.rectangle([x0 + px * s_, y0 + py * s_, x0 + (px + 64) * s_, y0 + (py + 64) * s_], outline=(255, 214, 0), width=2)
        gx, gy, gw, gh = 650, 130, 590, 360
        d.rectangle([gx, gy, gx + gw, gy + gh], outline=(60, 64, 70))
        allv = sum(curve.values(), []); m_ = max(1.0, max(abs(v) for v in allv) * 1.2); lo, hi = -m_, m_
        zy_ = gy + gh - 10 - (gh - 20) * (0 - lo) / (hi - lo); d.line([(gx, zy_), (gx + gw, zy_)], fill=(60, 64, 70), width=1)
        for key, lab, col in [("gt", "正常光参考", AQUA), ("dce", "Zero-DCE", ORANGE), ("ours", "本文", BLUE)]:
            pts = [(gx + 10 + (gw - 20) * k / 29, gy + gh - 10 - (gh - 20) * (v - lo) / (hi - lo)) for k, v in enumerate(curve[key][: i + 1])]
            if len(pts) > 1: d.line(pts, fill=col, width=3)
        for k, (lab, col) in enumerate([("正常光参考", AQUA), ("Zero-DCE", ORANGE), ("本文", BLUE)]):
            d.line([(gx + 20 + 170 * k, gy + gh + 30), (gx + 50 + 170 * k, gy + gh + 30)], fill=col, width=4)
            text(d, (gx + 58 + 170 * k, gy + gh + 18), lab, 18, INK)
        text(d, (gx, gy - 26), f"区域亮度相对自身均值的波动（±{m_:.1f}%） → 帧序号", 16, MUTED)
        if t > D * 0.6: text(d, (48, 450), f"全测试集扭曲误差：Zero-DCE {z['warp_error']:.4f} → 本文 {o['warp_error']:.4f}（降低 {we_drop}%）", 24, INK, True)

    def s_results(cv, t, D):
        header(cv, "定量结果", "作者官方 12 个测试场景（配置冻结后只评一次）")
        cv.paste(figure(ROOT / "output/figures/fig_test_bars.png", 1200, 330), (40, 110)); d = ImageDraw.Draw(cv)
        text(d, (48, 470), f"优势　SSIM：Zero-DCE {z['ssim']:.3f} → 本文 {o['ssim']:.3f}；闪烁（扭曲误差）降低 {we_drop}%", 22, AQUA)
        text(d, (48, 508), f"不足　PSNR：本文 {o['psnr']:.2f} dB，低于 Zero-DCE（{z['psnr']:.2f}）与第一版（{v1['psnr']:.2f}）", 22, ORANGE)
        text(d, (48, 546), "原因：自训练网络在更暗的测试场景上泛化不足（报告第 5.2 节如实分析）", 18, MUTED)

    def s_end(cv, t, D):
        d = ImageDraw.Draw(cv); text(d, (W // 2, 150), "总结", 44, INK, True, "mm")
        pts = ["噪声模型标定（泊松-高斯 + 空间相关性 κ）", "运动补偿卡尔曼滤波（新息检验防拖影）", "可学习双边网格 = 保边的分段灰度变换",
               "曝光解耦：两参数回归 + 曝光归一化损失", "9 种方法对比 + 7 组消融，测试集只评一次"]
        for k, s in enumerate(pts):
            if t > D * 0.1 * (k + 1): text(d, (330, 230 + 52 * k), "· " + s, 26, INK)
        if t > D * 0.7: text(d, (W // 2, 560), "谢谢观看", 34, BLUE, True, "mm")

    script = [
        (s_title, "大家好，这是我的图像处理课程项目：低光照视频增强。方法叫 KBG-Video，结合了噪声建模的卡尔曼去噪、可学习的双边网格和运动补偿融合。"),
        (s_problem, "暗光视频有三个难点。第一，画面太暗，需要提亮五倍左右，右边可以看到，噪声也被一起放大了。第二，逐帧处理会闪烁，而简单的帧间平均又会拖影。第三，正常光参考图应该多亮，并不能完全由输入决定。"),
        (s_data, "数据来自 SDSD 数据集，用机械臂重复同一段运动，分别拍摄暗光和正常光视频，得到逐帧对齐的真实配对。我们用三十四个场景训练，六个场景验证，作者官方的十二个测试场景只在最后评测一次。"),
        (s_pipeline, "整个方法分四步，逐帧在线运行，只用当前帧和过去帧。第一步卡尔曼去噪，第二步双边网格提亮，第三步曝光回归，第四步运动补偿融合。光流贯穿其中，用来对齐相邻帧。"),
        (s_kalman, "第一步在提亮之前去噪。我们先用相邻帧差分标定了泊松加高斯的噪声模型，还发现噪声有空间相关性。然后沿着光流对每个像素做卡尔曼滤波。增益图越暗，表示这个像素用了越多历史帧；遇到运动和遮挡，新息检验会让它退回当前帧，避免拖影。"),
        (s_grid, "第二步，一个只有七十多万参数的小网络看缩略图，预测一个十六乘十六乘八的双边网格，每一格是一个颜色变换。它本质上是随位置和亮度变化的灰度变换，又像双边滤波一样保边。网络从传统幂律变换初始化，只学习修正量。"),
        (s_expo, "训练中我们发现，参考图亮度在场景之间相差约百分之二十五，网络训练越久，越会死记训练场景的亮度。于是把曝光交给一个只有两个参数的回归模型。最后一步用光流置信度门控的融合抑制残余闪烁；我们也试过平滑网格系数，但验证集表明几乎无效。"),
        (s_slider, "下面是测试场景的对比，分割线左边是 Zero-DCE，右边是我们的方法。右侧放大框里可以看到，Zero-DCE 把暗部噪声一起放大了，我们的结果明显更干净。"),
        (s_flicker, f"再看时间稳定性。这是同一块静止区域的亮度随时间的变化。在全部测试集上，衡量闪烁的扭曲误差降低了百分之{we_drop}。"),
        (s_results, "定量结果如实来看：相比 Zero-DCE，我们的结构相似度从零点六九提高到零点七七，闪烁降低百分之八十四；但峰值信噪比低了零点五六分贝，也不如第一版方案。原因是自己训练的网络在更暗的测试场景上泛化不足，报告里做了详细分析。"),
        (s_end, "总结一下，这个项目用到了课程中的噪声模型、卡尔曼与维纳滤波、灰度变换、双边滤波、运动估计和深度学习增强。谢谢观看。"),
    ]

    tmp = Path(tempfile.mkdtemp()); audio = bytearray(); rate = 22050; segs = []
    for k, (fn, narr) in enumerate(script):
        pcm, dur = tts(narr, tmp / f"s{k}.wav")
        lead, tail = 0.5, 0.8; D = lead + dur + tail
        audio += bytes(int(lead * rate) * 2) + pcm + bytes(int(tail * rate) * 2)
        n_frames = int(round(D * FPS)); pad = n_frames / FPS - D
        if pad > 0: audio += bytes(int(pad * rate) * 2)
        else: audio = audio[: len(audio) + int(pad * rate) * 2 // 2 * 2]
        segs.append((fn, narr, lead, dur, n_frames)); print(f"tts {k}: {dur:.1f}s", flush=True)
    wav = tmp / "all.wav"
    with wave.open(str(wav), "wb") as w_: w_.setnchannels(1); w_.setsampwidth(2); w_.setframerate(rate); w_.writeframes(bytes(audio))

    silent = tmp / "video.mp4"
    wr = imageio_ffmpeg.write_frames(str(silent), (W, H), fps=FPS, codec="libx264", quality=8, macro_block_size=8,
                                     output_params=["-pix_fmt", "yuv420p"]); wr.send(None)
    total = 0
    for fn, narr, lead, dur, n_frames in segs:
        sents = sentences(narr); lens = np.cumsum([len(s) for s in sents]) / sum(len(s) for s in sents)
        D = n_frames / FPS
        for f_ in range(n_frames):
            t = f_ / FPS; cv = canvas(); fn(cv, t, D)
            if 0.3 < t:
                prog = (t - lead) / max(dur, 1e-3); idx = int(np.searchsorted(lens, max(0, min(prog, 0.999)), side="right"))
                subtitle(cv, sents[min(idx, len(sents) - 1)])
            a_ = min(1, t / 0.4, (D - t) / 0.4)
            wr.send(np.ascontiguousarray(np.asarray(fade(cv, a_) if a_ < 1 else cv)))
        total += n_frames
        print("segment", fn.__name__, f"{D:.1f}s", flush=True)
    wr.close()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(silent), "-i", str(wav), "-c:v", "copy", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "44100", "-c:a", "aac", "-b:a", "128k",
                    "-shortest", "-movflags", "+faststart", str(a.out)], check=True, capture_output=True)
    print("DONE", a.out, f"{total / FPS:.1f}s")


if __name__ == "__main__":
    main()
