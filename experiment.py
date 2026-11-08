"""Scene-disjoint validation followed by one frozen held-out evaluation.

No neural-network training is claimed. This experiment extends an official
pretrained image enhancer with lightweight video processing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import cv2
import numpy as np
import torch

from lowlight import (ROOT, CurveState, apply_curve, flow_context, image_numpy,
                      load_model, quality, quantize, reference_stats, save_rgb, tensor_image, timed_sync)

VALIDATION = [f"pair{i}" for i in range(1,7)]
TEST = [f"pair{i}" for i in [13,15,21,23,31,33,50,52,58,60,68,70]]


def pairs(scene, frames):
    root = ROOT / "data/raw/indoor_np"
    lq, gt = sorted((root/"input"/scene).glob("*.npy")), sorted((root/"GT"/scene).glob("*.npy"))
    if len(lq) < frames or len(gt) < frames:
        raise RuntimeError(f"Incomplete sequence {scene}: {len(lq)} LQ, {len(gt)} GT, need {frames}")
    # Match upstream positional pairing, NOT identical filenames.
    for i, (a,b) in enumerate(zip(lq[:frames],gt[:frames])):
        yield i,a,b,np.load(a)[...,::-1].copy().astype(np.float32)/255,np.load(b)[...,::-1].copy().astype(np.float32)/255


def evaluate(scenes, frames, candidates, output_dir, device="auto", save_examples=False):
    output_dir.mkdir(parents=True,exist_ok=True)
    net, dev = load_model(device); torch.set_num_threads(6)
    cv2.setNumThreads(4)
    rows, sequence_rows = [], []
    for scene in scenes:
        states = {name:CurveState(p['beta'],p.get('align',True)) for name,p in candidates.items() if p.get('beta',0)>0}
        prev_rgb, previous_errors = None, {}
        for i,lp,gp,rgb,gt in pairs(scene,frames):
            stats = reference_stats(gt)
            x = tensor_image(rgb,dev)
            start = timed_sync(dev)
            with torch.inference_mode(): _,base_tensor,curves = net(x)
            network_ms = (timed_sync(dev)-start)*1000
            base = image_numpy(base_tensor)
            start = time.perf_counter()
            context = None if prev_rgb is None else flow_context(rgb,prev_rgb)
            flow_ms = (time.perf_counter()-start)*1000
            images = {'Input':rgb,'Gamma':np.power(rgb,0.4), 'ZeroDCE':base}
            for name,p in candidates.items():
                t0 = timed_sync(dev)
                if name.startswith('Gamma_'):
                    pred=np.power(rgb,p['gamma']); extra_ms=0.
                else:
                    with torch.inference_mode():
                        r = states[name].update(curves, context if p.get('align',True) else (None,None,context[2]) if context else None) if name in states else curves
                        pred=image_numpy(apply_curve(x,r)) if name in states else base
                    if p.get('gamma',1.) != 1.: pred=np.power(pred,p['gamma'])
                    extra_ms=(timed_sync(dev)-t0)*1000 + (flow_ms if p.get('align',True) and name in states else 0.)
                images[name]=pred
                p['_last_extra_ms']=extra_ms
            for name,pred in images.items():
                psnr,ssim=quality(pred,gt,stats)
                rounded=quantize(pred).astype(np.float32)/255
                error=rounded-gt
                tce=float(np.mean(np.abs(error-previous_errors[name]))) if name in previous_errors else None
                brightness_error=float(error.mean())
                previous_errors[name]=error
                rows.append(dict(scene=scene,index=i,lq=lp.name,gt=gp.name,method=name,psnr=psnr,ssim=ssim,
                                 tce=tce,brightness_error=brightness_error,network_ms=network_ms,
                                 extra_ms=candidates.get(name,{}).get('_last_extra_ms',0.),
                                 cut=bool(context[2]) if context else False))
                if save_examples and i in (0,frames//2,frames-1):
                    folder=output_dir/'examples'/scene;folder.mkdir(parents=True,exist_ok=True)
                    save_rgb(folder/f'{i:03d}_{name}.png',pred)
            if save_examples and i in (0,frames//2,frames-1):
                save_rgb(output_dir/'examples'/scene/f'{i:03d}_GT.png',gt)
            prev_rgb=rgb
        print(f"Finished {scene}: {frames} paired consecutive frames",flush=True)
    for scene in scenes:
        for method in sorted(set(r['method'] for r in rows)):
            rr=[r for r in rows if r['scene']==scene and r['method']==method]
            v=dict(scene=scene,method=method,frames=len(rr))
            for k in ('psnr','ssim','tce','network_ms','extra_ms'):
                v[k]=float(np.mean([r[k] for r in rr if r[k] is not None]))
            v['brightness_flicker']=float(np.std(np.diff([r['brightness_error'] for r in rr])))
            sequence_rows.append(v)
    methods={}
    for name in sorted(set(r['method'] for r in rows)):
        seq=[r for r in sequence_rows if r['method']==name]
        methods[name]={k:float(np.mean([r[k] for r in seq])) for k in ('psnr','ssim','tce','brightness_flicker','network_ms','extra_ms')}
    for fname,records in [('per_frame.csv',rows),('per_scene.csv',sequence_rows)]:
        with (output_dir/fname).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    config={n:{k:v for k,v in p.items() if not k.startswith('_')} for n,p in candidates.items()}
    summary=dict(scenes=scenes,frames_per_scene=frames,resolution=[960,512],methods=methods,candidates=config,
                 pairing='independent filename sorting and positional pairing, as in SDSD upstream',
                 runtime_note='network_ms shared; extra_ms includes alignment when used; excludes metrics and IO; first-frame warm-up retained',
                 software=dict(python=platform.python_version(),torch=torch.__version__,opencv=cv2.__version__,device=dev),
                 weights_sha256=hashlib.sha256((ROOT/'weights/Epoch99.pth').read_bytes()).hexdigest())
    (output_dir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(methods,indent=2),flush=True)
    return summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['smoke','validation','test'],required=True)
    ap.add_argument('--frames',type=int,default=30);ap.add_argument('--device',default='auto')
    args=ap.parse_args()
    if args.stage=='smoke':
        evaluate(['pair1'],3,{'Temporal':dict(beta=.6,align=True,gamma=1.)},ROOT/'results/smoke',args.device,True);return
    if args.stage=='validation':
        candidates={f'Gamma_{g:g}':dict(gamma=g,beta=0) for g in [.2,.3,.4,.5,.6]}
        candidates.update({f'Exposure_{g:g}':dict(gamma=g,beta=0) for g in [.6,.8,1.,1.2]})
        candidates.update({f'Temporal_{b:g}':dict(beta=b,align=True,gamma=1.) for b in [.3,.6,.85]})
        candidates['NaiveEMA']=dict(beta=.6,align=False,gamma=1.)
        s=evaluate(VALIDATION,args.frames,candidates,ROOT/'results/validation',args.device)
        # Exposure calibration uses validation GT only. Temporal strength is chosen
        # for lowest paired TCE subject to <=0.05 dB PSNR loss on validation.
        exposure=max([n for n in candidates if n.startswith('Exposure_')],key=lambda n:s['methods'][n]['psnr'])
        gamma=candidates[exposure]['gamma']
        admissible=[n for n in candidates if n.startswith('Temporal_') and s['methods'][n]['psnr']>=s['methods']['ZeroDCE']['psnr']-.05]
        # Include the unchanged baseline; do not force a temporal modification
        # when every candidate worsens the chosen temporal endpoint.
        temporal=min(admissible+['ZeroDCE'],key=lambda n:s['methods'][n]['tce'])
        if temporal=='ZeroDCE':temporal=None
        beta=candidates[temporal]['beta'] if temporal else 0.
        chosen=dict(gamma=gamma,beta=beta,align=True,validation_scenes=VALIDATION,test_scenes=TEST,
                    rule='validation PSNR for gamma; minimum validation TCE with <=0.05dB PSNR loss for temporal beta',
                    gamma_baseline=max([p['gamma'] for n,p in candidates.items() if n.startswith('Gamma_')],
                        key=lambda g:s['methods'][f'Gamma_{g:g}']['psnr']))
        (ROOT/'records/selected_config.json').write_text(json.dumps(chosen,indent=2),encoding='utf-8')
        print('Frozen config',chosen,flush=True)
    else:
        config_path=ROOT/'records/selected_config.json';c=json.loads(config_path.read_text())
        candidates={'Exposure':dict(beta=0,gamma=c['gamma']),
                    'NaiveEMA':dict(beta=c['beta'],align=False,gamma=1.),
                    'Temporal':dict(beta=c['beta'],align=True,gamma=1.),
                    'Improved':dict(beta=c['beta'],align=True,gamma=c['gamma']),
                    f"Gamma_{c['gamma_baseline']:g}":dict(beta=0,gamma=c['gamma_baseline'])}
        (ROOT/'records/test_config_sha256.txt').write_text(hashlib.sha256(config_path.read_bytes()).hexdigest())
        evaluate(TEST,args.frames,candidates,ROOT/'results/test',args.device,True)


if __name__=='__main__':main()
