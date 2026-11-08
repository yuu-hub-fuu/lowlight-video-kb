import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from lowlight import CurveState,ImageState,apply_curve,flow_context,quality


class CoreTests(unittest.TestCase):
    def test_curve_identity_and_bounds(self):
        x=torch.rand(1,3,24,32)
        self.assertTrue(torch.equal(apply_curve(x,torch.zeros(1,24,24,32)),x))
        y=apply_curve(x,torch.rand(1,24,24,32)*2-1)
        self.assertTrue(bool(((y>=0)&(y<=1)).all()))

    def test_state_reset_and_first_frame(self):
        state=CurveState(.8)
        a=torch.rand(1,24,24,32);b=torch.rand_like(a)
        self.assertTrue(torch.equal(state.update(a),a))
        self.assertTrue(torch.equal(state.update(b,(None,None,True)),b))

    def test_cut_detector(self):
        a=np.zeros((64,96,3),np.float32);b=np.ones_like(a)
        self.assertFalse(flow_context(a,a)[2]);self.assertTrue(flow_context(a,b)[2])

    def test_motion_alignment_preserves_translated_structure(self):
        # Known one-pixel translation must be corrected by the supplied backward
        # grid rather than blurred as an unaligned temporal average would be.
        h,w=24,32
        a=torch.zeros(1,3,h,w);a[:,:,:,10:14]=1
        b=torch.zeros_like(a);b[:,:,:,11:15]=1
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        grid=np.stack([2*(xx-1)/(w-1)-1,2*yy/(h-1)-1],axis=-1)
        state=ImageState(.85);state.update(a)
        out=state.update(b,(grid,np.ones((h,w),np.float32),False))
        self.assertTrue(torch.allclose(out[:,:,:,2:-2],b[:,:,:,2:-2],atol=2e-6))
        self.assertTrue(torch.equal(state.update(a,(None,None,True)),a))

    def test_metrics_reference(self):
        rng=np.random.default_rng(4);x=rng.random((32,40,3),dtype=np.float32)
        p,s=quality(x,x);self.assertGreater(p,100);self.assertAlmostEqual(s,1,places=5)
        # An actual corruption must reduce both metrics.
        p2,s2=quality(x*.5,x);self.assertLess(p2,p);self.assertLess(s2,s)

    def test_ssim_matches_independent_implementation(self):
        from skimage.metrics import structural_similarity
        from lowlight import quantize
        rng=np.random.default_rng(6)
        a=quantize(rng.random((48,64,3),dtype=np.float32)).astype(np.float32)/255
        b=quantize(a*.8).astype(np.float32)/255
        _,ours=quality(a,b)
        ref=structural_similarity(a,b,data_range=1.,gaussian_weights=True,
                                sigma=1.5,use_sample_covariance=False,channel_axis=-1)
        self.assertAlmostEqual(ours,float(ref),places=5)


if __name__=='__main__':unittest.main()
