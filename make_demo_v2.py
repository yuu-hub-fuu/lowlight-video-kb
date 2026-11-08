"""生成第二版演示视频（真实 SDSD 连续帧，每场景前 30 帧，固定顺序，不挑帧）。

output/KBG_四栏对比_18场景.mp4   暗光输入 | Zero-DCE | KBG-Video(本文) | 正常光参考
output/KBG_算法过程_6场景.mp4     暗光输入(×4显示) | ①卡尔曼去噪后(×4) | 卡尔曼增益K(越暗=历史帧占比越高)
                                   ②BGNet+曝光回归 单帧输出 | ③运动补偿融合后(最终) | 正常光参考
output/KBG_最终输出_18场景.mp4 / KBG_暗光输入_18场景.mp4
播放帧率 10 fps 为展示选择，不代表原始采集帧率；场景之间显式重置状态。
"""
import json
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from kbg import KBGVideo
from lowlight import ROOT, image_numpy, load_model, quantize, tensor_image
from splits import TEST, VAL
from train_trnet import load_arrays

FONT = next((p for p in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/arial.ttf"] if Path(p).exists()), None)


def writer(path, size, fps=10):
    w = imageio_ffmpeg.write_frames(str(path), size, fps=fps, codec="libx264", quality=8, macro_block_size=2,
                                    output_params=["-movflags", "+faststart"])
    w.send(None); return w


def mosaic(panels, labels, cols, tile=(480, 256), head=34):
    font = ImageFont.truetype(FONT, 18) if FONT else ImageFont.load_default()
    rows = (len(panels) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile[0], rows * (tile[1] + head)), "#ffffff"); d = ImageDraw.Draw(canvas)
    for k, (p, lab) in enumerate(zip(panels, labels)):
        x, y = (k % cols) * tile[0], (k // cols) * (tile[1] + head)
        canvas.paste(Image.fromarray(quantize(p)).resize(tile, Image.Resampling.LANCZOS), (x, y + head))
        d.text((x + 8, y + 7), lab, font=font, fill="#111111")
    return np.ascontiguousarray(canvas)


def main():
    cfg = json.loads((ROOT / "records/v2_config.json").read_text(encoding="utf-8"))
    out = ROOT / "output"; out.mkdir(exist_ok=True)
    dce, dev = load_model(); kbg = KBGVideo.final(dev)
    scenes = TEST + VAL; process_scenes = TEST[:3] + VAL[:3]
    W = dict(cmp=writer(out / "KBG_四栏对比_18场景.mp4", (960, 580)), src=writer(out / "KBG_暗光输入_18场景.mp4", (960, 512)),
             enh=writer(out / "KBG_最终输出_18场景.mp4", (960, 512)), proc=writer(out / "KBG_算法过程_6场景.mp4", (1440, 580)))
    count = 0
    try:
        for scene in scenes:
            kbg.reset(); lq, gt = load_arrays(scene, 30)
            for i in range(30):
                rgb = lq[i].astype(np.float32) / 255; ref = gt[i].astype(np.float32) / 255
                with torch.inference_mode():
                    base = image_numpy(dce(tensor_image(rgb, dev))[1])
                r = kbg.step(rgb); final = image_numpy(r["out"])
                tag = f"{scene} #{i:02d}" + ("（测试）" if scene in TEST else "（验证）")
                frame = mosaic([rgb, base, final, ref], [f"暗光输入 {tag}", "Zero-DCE", "KBG-Video（本文）", "正常光参考"], 2)
                W["cmp"].send(frame); W["src"].send(np.ascontiguousarray(quantize(rgb))); W["enh"].send(np.ascontiguousarray(quantize(final)))
                if i == 15:
                    Image.fromarray(frame).save(out / f"{scene}_KBG四栏对比.png")
                if scene in process_scenes:
                    den = image_numpy(r["denoised"]); K = kbg.den.last_gain.mean(1, keepdim=True)[0, 0].cpu().numpy()
                    kmap = cv2.applyColorMap(quantize(np.clip(K, 0, 1)), cv2.COLORMAP_VIRIDIS)[..., ::-1].astype(np.float32) / 255
                    proc = mosaic([np.clip(rgb * 4, 0, 1), np.clip(den * 4, 0, 1), kmap, image_numpy(r["raw"]), final, ref],
                                  [f"暗光输入×4 {tag}", "①卡尔曼去噪后×4", "卡尔曼增益K（暗=多用历史帧）",
                                   "②BGNet+曝光回归 单帧输出", "③运动补偿融合后（最终输出）", "正常光参考"], 3)
                    W["proc"].send(proc)
                    if i == 15: Image.fromarray(proc).save(out / f"{scene}_算法过程.png")
                count += 1
            print("video", scene, flush=True)
    finally:
        for w in W.values(): w.close()
    (out / "video_metadata_v2.json").write_text(json.dumps(dict(
        frames=count, fps=10, duration_seconds=count / 10, scenes=scenes, process_scenes=process_scenes, frames_per_scene=30,
        source="SDSD indoor_np, original paired chronological frames", playback="10 fps chosen for display",
        audio="None: source data are image sequences", config=cfg), indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
