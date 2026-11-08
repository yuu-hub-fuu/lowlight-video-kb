"""Video extension of Zero-DCE. Upstream network remains unmodified.

Our additions: persistent model loading, video IO, flow-aware curve smoothing,
cut resets, evaluation, and validation-only exposure calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def load_model(device: str = "auto"):
    if device == "auto": device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = importlib.util.spec_from_file_location("upstream_zero_dce", ROOT / "vendor/zero_dce/model.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    net = mod.enhance_net_nopool().to(device)
    state = torch.load(ROOT / "weights/Epoch99.pth", map_location=device, weights_only=True)
    net.load_state_dict(state); net.eval()
    return net, device


def tensor_image(rgb: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(device)


def image_numpy(t: torch.Tensor) -> np.ndarray:
    return t[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def apply_curve(x: torch.Tensor, curves: torch.Tensor) -> torch.Tensor:
    for r in curves.split(3, dim=1): x = x + r * (x.square() - x)
    return x.clamp(0, 1)


def quantize(rgb: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)


def save_rgb(path: Path, rgb: np.ndarray):
    """OpenCV imwrite can silently fail for Chinese Windows paths."""
    ok, encoded = cv2.imencode('.png', quantize(rgb)[..., ::-1])
    if not ok: raise IOError(f'Could not encode {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def flow_context(current_rgb, previous_rgb, scale=0.25):
    """Backward flow and consistency mask; only low-light inputs are used.

    Moving pixels are aligned before temporal filtering. A conservative
    scene-cut detector resets the entire state.
    """
    h, w = current_rgb.shape[:2]
    shape = (max(16, round(w*scale)), max(16, round(h*scale)))
    cur = cv2.resize(current_rgb, shape, interpolation=cv2.INTER_AREA)
    prev = cv2.resize(previous_rgb, shape, interpolation=cv2.INTER_AREA)
    gc = cv2.cvtColor(cur, cv2.COLOR_RGB2GRAY)
    gp = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    # Absolute and histogram changes together avoid treating camera panning as a cut.
    hist_c = cv2.calcHist([quantize(cur)], [0, 1, 2], None, [8, 8, 8], [0,256]*3)
    hist_p = cv2.calcHist([quantize(prev)], [0, 1, 2], None, [8, 8, 8], [0,256]*3)
    cv2.normalize(hist_c, hist_c, norm_type=cv2.NORM_L1)
    cv2.normalize(hist_p, hist_p, norm_type=cv2.NORM_L1)
    hist_dist = cv2.compareHist(hist_c, hist_p, cv2.HISTCMP_BHATTACHARYYA)
    change = float(np.mean(np.abs(cur-prev)))
    cut = bool(change > 0.12 and hist_dist > 0.55)
    gc8 = quantize(np.sqrt(np.maximum(gc, 0)))
    gp8 = quantize(np.sqrt(np.maximum(gp, 0)))
    back = cv2.calcOpticalFlowFarneback(gc8, gp8, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    forward = cv2.calcOpticalFlowFarneback(gp8, gc8, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    sy, sx = np.mgrid[:shape[1], :shape[0]].astype(np.float32)
    mx, my = sx + back[..., 0], sy + back[..., 1]
    warped_prev = cv2.remap(prev, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    warped_forward = cv2.remap(forward, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    fb = np.linalg.norm(back + warped_forward, axis=-1)
    photo = np.mean(np.abs(cur - warped_prev), axis=-1)
    valid = ((mx >= 0) & (my >= 0) & (mx < shape[0]-1) & (my < shape[1]-1))
    confidence = np.exp(-photo / 0.08) * np.exp(-fb / 1.5) * valid
    flow = cv2.resize(back, (w, h), interpolation=cv2.INTER_LINEAR)
    flow[..., 0] *= w / shape[0]; flow[..., 1] *= h / shape[1]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    grid = np.stack([2*(xx+flow[...,0])/max(w-1,1)-1,
                     2*(yy+flow[...,1])/max(h-1,1)-1], axis=-1)
    mask = cv2.resize(confidence.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return grid, mask, cut


@dataclass
class CurveState:
    beta: float = 0.6
    align: bool = True
    previous: torch.Tensor | None = None

    def update(self, curves: torch.Tensor, context=None) -> torch.Tensor:
        if self.previous is None or (context is not None and context[2]):
            result = curves
        elif self.align and context is not None:
            grid, confidence, _ = context
            g = torch.from_numpy(grid).unsqueeze(0).to(curves.device)
            aligned = torch.nn.functional.grid_sample(self.previous, g, align_corners=True,
                                                       padding_mode="border")
            weight = torch.from_numpy(confidence).to(curves.device)[None,None] * self.beta
            result = (1-weight)*curves + weight*aligned
        else:
            result = (1-self.beta)*curves + self.beta*self.previous
        self.previous = result.detach()
        return result


@dataclass
class ImageState:
    """Confidence-gated temporal filter on enhanced RGB, separate ablation."""
    beta: float = 0.6
    align: bool = True
    previous: torch.Tensor | None = None

    def update(self, current: torch.Tensor, context=None) -> torch.Tensor:
        if self.previous is None or (context is not None and context[2]):
            result = current
        elif self.align and context is not None:
            grid, confidence, _ = context
            g=torch.from_numpy(grid).unsqueeze(0).to(current.device)
            prev=torch.nn.functional.grid_sample(self.previous,g,align_corners=True,padding_mode='border')
            weight=torch.from_numpy(confidence).to(current.device)[None,None]*self.beta
            result=current*(1-weight)+prev*weight
        else:
            result=current*(1-self.beta)+self.previous*self.beta
        self.previous=result.detach()
        return result.clamp(0,1)


def reference_stats(gt):
    mu = cv2.GaussianBlur(gt, (11,11), 1.5)
    var = cv2.GaussianBlur(gt*gt, (11,11), 1.5) - mu*mu
    return mu, var


def quality(pred, gt, stats=None):
    """RGB PSNR and Gaussian-window SSIM on 8-bit-rounded images, no crop for PSNR."""
    pred = quantize(pred).astype(np.float32)/255
    gt = quantize(gt).astype(np.float32)/255
    mse = float(np.mean((pred-gt)**2))
    psnr = float(-10*np.log10(max(mse, 1e-12)))
    mg, vg = reference_stats(gt) if stats is None else stats
    mp = cv2.GaussianBlur(pred,(11,11),1.5)
    vp = cv2.GaussianBlur(pred*pred,(11,11),1.5) - mp*mp
    cov = cv2.GaussianBlur(pred*gt,(11,11),1.5) - mp*mg
    s = ((2*mp*mg+0.01**2)*(2*cov+0.03**2))/((mp*mp+mg*mg+0.01**2)*(vp+vg+0.03**2))
    ssim = float(np.mean(s[5:-5,5:-5]))
    return psnr, ssim


def timed_sync(device):
    if str(device).startswith("cuda"): torch.cuda.synchronize()
    return time.perf_counter()
