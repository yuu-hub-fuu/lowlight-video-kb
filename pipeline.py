"""完整视频增强流水线 TRNet-Video（逐帧在线、因果，不使用未来帧）。

    LQ帧 ─┬─> 光流(Farneback,1/4分辨率) ──> 对齐上一帧 + 置信度
          │                                         │
          └───────────────> TRNet(当前帧, 对齐帧, 置信度) ──> 单帧增强结果
                                                    │
                    上一帧输出 --(同一光流)--> 置信度门控的时域融合 ──> 输出
    镜头切换检测触发时，两个时域通路都被重置。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from lowlight import ROOT, ImageState, flow_context, image_numpy, quantize, tensor_image
from vnet import TRNet, align_prev


def load_trnet(name: str, device: str = "auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.loads((ROOT / f"records/train_{name}.json").read_text(encoding="utf-8"))
    net = TRNet(temporal=cfg["temporal"]).to(device)
    net.load_state_dict(torch.load(ROOT / f"weights/{name}.pth", map_location=device, weights_only=True))
    return net.eval(), device


class VideoEnhancer:
    def __init__(self, name="trnet_2frame", beta=0.0, device="auto"):
        self.net, self.dev = load_trnet(name, device)
        self.temporal = self.net.temporal
        self.beta = beta
        self.reset()

    def reset(self):
        self.prev = None
        self.state = ImageState(self.beta, True)

    @torch.inference_mode()
    def step(self, rgb: np.ndarray):
        """rgb: float32 HxWx3 in [0,1]。返回 (网络输出, 融合后输出, 本帧是否判为镜头切换)。"""
        context = flow_context(rgb, self.prev) if self.prev is not None else None
        cut = bool(context and context[2])
        cur = tensor_image(rgb, self.dev)
        if self.temporal:
            if context is None or cut:
                prev, conf = cur, torch.zeros_like(cur[:, :1])
            else:
                grid, mask, _ = context
                # 与训练一致：对齐帧、置信度都按 8 位存储精度取整
                prev = torch.round(align_prev(self.prev, grid, self.dev).clamp(0, 1) * 255) / 255
                conf = torch.round(torch.from_numpy(mask).to(self.dev)[None, None].clamp(0, 1) * 255) / 255
            out = self.net(cur, prev, conf).clamp(0, 1)
        else:
            out = self.net(cur).clamp(0, 1)
        fused = self.state.update(out, context) if self.beta > 0 else out
        self.prev = rgb
        return image_numpy(out), image_numpy(fused), cut
