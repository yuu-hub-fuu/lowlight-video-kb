"""处理一个真实视频文件（保持帧率，默认保留音轨）。

模式
  improved  KBG-Video：卡尔曼去噪 + 双边网格 + 曝光回归 + 运动补偿融合（默认，v2 冻结配置）
  v1        Zero-DCE + 光流对齐的 RGB 时域滤波（第一版方案）
  baseline  原始 Zero-DCE 逐帧推理
"""
import argparse
import json
from pathlib import Path
import time

import imageio_ffmpeg
import numpy as np
import torch

from lowlight import ROOT, ImageState, flow_context, image_numpy, load_model, quantize, tensor_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mode", choices=["baseline", "v1", "improved"], default="improved")
    ap.add_argument("--config", type=Path, default=ROOT / "records/v2_config.json")
    ap.add_argument("--device", default="auto"); ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()
    if a.input.resolve() == a.output.resolve(): raise ValueError("Input and output must be different files")
    torch.set_num_threads(6)
    if a.mode == "improved":
        from kbg import KBGVideo
        enhancer = KBGVideo.final(a.device); dev = enhancer.dev
    else:
        net, dev = load_model(a.device)
        v1 = json.loads((ROOT / "records/final_config.json").read_text(encoding="utf-8"))
        state = ImageState(v1["beta"])
    reader = imageio_ffmpeg.read_frames(str(a.input)); meta = next(reader)
    w, h = meta["size"]; a.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(a.output), (w, h), fps=meta["fps"], codec="libx264", quality=8,
        macro_block_size=2, audio_path=str(a.input) if not a.no_audio and meta.get("audio_codec") else None,
        audio_codec="aac", output_params=["-movflags", "+faststart"])
    writer.send(None)
    prev = None; n = 0; cuts = 0; started = time.perf_counter()
    try:
        for raw in reader:
            rgb = np.frombuffer(raw, np.uint8).reshape(h, w, 3).astype(np.float32) / 255
            if a.mode == "improved":
                out, cut = enhancer.step_numpy(rgb); cuts += int(cut); frame = quantize(out)
            else:
                with torch.inference_mode():
                    _, pred, _ = net(tensor_image(rgb, dev))
                    if a.mode == "v1":
                        context = flow_context(rgb, prev) if prev is not None else None
                        cuts += int(bool(context and context[2]))
                        pred = state.update(pred.clamp(0, 1).pow(v1["gamma"]), context)
                    frame = quantize(image_numpy(pred))
            writer.send(np.ascontiguousarray(frame)); prev = rgb; n += 1
            if n % 30 == 0: print(f"{n} frames", flush=True)
            if a.max_frames and n >= a.max_frames: break
    finally:
        reader.close(); writer.close()
    stats = dict(frames=n, width=w, height=h, fps=meta["fps"], mode=a.mode, cuts=cuts, seconds=time.perf_counter() - started)
    a.output.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
