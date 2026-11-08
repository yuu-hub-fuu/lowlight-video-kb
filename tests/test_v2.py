"""第二版模块的单元测试：卡尔曼滤波性质、双边网格初始化、算子平滑、数据划分。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from bgnet import BGNet, GridSmoother
from kalman import KalmanDenoiser
import classical as C


def identity_context(h, w, conf=1.0):
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    grid = np.stack([2 * xx / (w - 1) - 1, 2 * yy / (h - 1) - 1], -1)
    return grid, np.full((h, w), conf, np.float32), False


class KalmanTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b = 1.36e-4, 1.58e-5

    def test_static_scene_noise_is_reduced(self):
        """静止场景：卡尔曼输出的误差方差应远小于单帧噪声方差。"""
        rng = np.random.default_rng(0); h, w = 48, 64
        clean = torch.full((1, 3, h, w), 0.08)
        kf = KalmanDenoiser(self.a, self.b, q0=0.02, q1=0.0, tau=6, kappa=0.345)
        ctx = identity_context(h, w); sd = np.sqrt(self.a * 0.08 + self.b)
        for t in range(20):
            z = clean + torch.from_numpy(rng.normal(0, sd, (1, 3, h, w)).astype(np.float32))
            out = kf.update(z, None if t == 0 else ctx)
        single = float(((z - clean) ** 2).mean()); filt = float(((out - clean) ** 2).mean())
        self.assertLess(filt, single / 3)

    def test_innovation_test_rejects_sudden_change(self):
        """画面突变（遮挡/光流失败）时，新息检验应让输出几乎等于当前帧，不留拖影。"""
        h, w = 32, 32
        kf = KalmanDenoiser(self.a, self.b, q0=0.02, q1=0.0, tau=6, kappa=0.345)
        ctx = identity_context(h, w)
        for t in range(10): kf.update(torch.full((1, 3, h, w), 0.05), None if t == 0 else ctx)
        out = kf.update(torch.full((1, 3, h, w), 0.30), ctx)
        self.assertGreater(float(out.mean()), 0.29)
        no_test = KalmanDenoiser(self.a, self.b, q0=0.02, q1=0.0, tau=6, use_test=False)
        for t in range(10): no_test.update(torch.full((1, 3, h, w), 0.05), None if t == 0 else ctx)
        self.assertLess(float(no_test.update(torch.full((1, 3, h, w), 0.30), ctx).mean()), 0.2)

    def test_cut_resets_state(self):
        h, w = 16, 16
        kf = KalmanDenoiser(self.a, self.b)
        kf.update(torch.zeros(1, 3, h, w))
        z = torch.rand(1, 3, h, w)
        self.assertTrue(torch.equal(kf.update(z, (None, None, True)), z))


class GridTests(unittest.TestCase):
    def test_initial_network_equals_power_law(self):
        """初始化时双边网格每格 A=1.5I, b=0，网络应精确等于传统幂律变换 1.5·r^0.4。"""
        net = BGNet().eval(); x = torch.rand(1, 3, 64, 96) * 0.3
        with torch.no_grad(): out = net(x).clamp(0, 1)[0].permute(1, 2, 0).numpy()
        ref = C.power_law(x[0].permute(1, 2, 0).numpy())
        self.assertLess(float(np.abs(out - ref).max()), 1e-5)

    def test_grid_shape_resolution_independent(self):
        net = BGNet().eval()
        with torch.no_grad():
            g1 = net.predict_grid(torch.rand(1, 3, 128, 240)); g2 = net.predict_grid(torch.rand(1, 3, 512, 960))
        self.assertEqual(tuple(g1.shape), (1, 12, 8, 16, 16)); self.assertEqual(g1.shape, g2.shape)

    def test_smoother(self):
        s = GridSmoother(0.8); a = torch.zeros(2); b = torch.ones(2)
        self.assertTrue(torch.equal(s.update(a), a))
        self.assertTrue(torch.allclose(s.update(b), torch.full((2,), 0.2)))
        self.assertTrue(torch.equal(s.update(b, cut=True), b))


class SplitTests(unittest.TestCase):
    def test_scene_disjoint(self):
        from splits import TEST, TRAIN, VAL
        self.assertFalse(set(TRAIN) & set(TEST)); self.assertFalse(set(VAL) & set(TEST)); self.assertFalse(set(TRAIN) & set(VAL))
        self.assertEqual(len(TEST), 12)


class ClassicalTests(unittest.TestCase):
    def test_guided_filter_preserves_constant_and_edges(self):
        img = np.zeros((40, 40), np.float32); img[:, 20:] = 1
        out = C.guided_filter(img, img, 4, 1e-4)
        self.assertLess(abs(float(out[:, :15].mean())), 1e-3); self.assertGreater(float(out[:, 25:].mean()), 0.999)


if __name__ == "__main__":
    unittest.main()
