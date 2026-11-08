"""Second pre-test validation: inspect temporal noise suppression, not only curves.

The additional validation is motivated by noise visible in the validation
preview. Test GT is not read until the configuration has been frozen.
"""
import argparse
import csv
import hashlib
import json
import time

import cv2
import numpy as np
import torch

from experiment import VALIDATION,TEST,pairs
from lowlight import (ROOT,ImageState,CurveState,apply_curve,flow_context,image_numpy,load_model,
                      quality,quantize,reference_stats,save_rgb,tensor_image,timed_sync)


def evaluate(scenes,frames,candidates,out,device='auto',save=False):
    out.mkdir(parents=True,exist_ok=True)
    net,dev=load_model(device);torch.set_num_threads(6);cv2.setNumThreads(4)
    rows=[]
    for scene in scenes:
        states={n:ImageState(p['beta'],p.get('align',True)) for n,p in candidates.items() if p.get('beta',0)>0}
        curve_states={n:CurveState(p['curve_beta'],True) for n,p in candidates.items() if p.get('curve_beta',0)>0}
        prev=None;old_errors={}
        for i,lp,gp,rgb,gt in pairs(scene,frames):
            stats=reference_stats(gt);x=tensor_image(rgb,dev)
            t=timed_sync(dev)
            with torch.inference_mode(): _,base,curves=net(x)
            network_ms=(timed_sync(dev)-t)*1000
            t=time.perf_counter();context=flow_context(rgb,prev) if prev is not None else None
            flow_ms=(time.perf_counter()-t)*1000
            outputs={'Input':rgb,'ZeroDCE':image_numpy(base)};times={'Input':0.,'ZeroDCE':network_ms}
            for name,p in candidates.items():
                t=timed_sync(dev)
                with torch.inference_mode():
                    pred=base
                    if name in curve_states:pred=apply_curve(x,curve_states[name].update(curves,context))
                    pred=pred.clamp(0,1).pow(p.get('gamma',1.))
                    if p.get('gamma_input'):
                        pred=x.pow(p['gamma_input'])
                    if p.get('spatial'):
                        arr=image_numpy(pred)
                        arr=cv2.bilateralFilter(arr,5,.1,3.)
                        pred=tensor_image(arr,dev)
                    if name in states:pred=states[name].update(pred,context)
                    outputs[name]=image_numpy(pred)
                times[name]=(timed_sync(dev)-t)*1000+network_ms+(flow_ms if p.get('align',True) and (name in states or name in curve_states) else 0.)
            for name,pred in outputs.items():
                psnr,ssim=quality(pred,gt,stats);error=quantize(pred).astype(np.float32)/255-gt
                tce=float(np.mean(np.abs(error-old_errors[name]))) if name in old_errors else None
                old_errors[name]=error
                rows.append(dict(scene=scene,index=i,lq=lp.name,gt=gp.name,method=name,psnr=psnr,ssim=ssim,
                    tce=tce,brightness_error=float(error.mean()),runtime_ms=times[name],cut=bool(context[2]) if context else False))
                if save and i in [0,frames//2,frames-1]:save_rgb(out/'examples'/scene/f'{i:03d}_{name}.png',pred)
            if save and i in [0,frames//2,frames-1]:save_rgb(out/'examples'/scene/f'{i:03d}_GT.png',gt)
            prev=rgb
        print('Finished',scene,flush=True)
    seqs=[]
    for scene in scenes:
        for method in sorted({r['method'] for r in rows}):
            rr=[r for r in rows if r['scene']==scene and r['method']==method]
            v=dict(scene=scene,method=method,frames=len(rr))
            for k in ('psnr','ssim','tce','runtime_ms'):v[k]=float(np.mean([r[k] for r in rr if r[k] is not None]))
            v['brightness_flicker']=float(np.std(np.diff([r['brightness_error'] for r in rr])))
            seqs.append(v)
    methods={n:{k:float(np.mean([r[k] for r in seqs if r['method']==n])) for k in ['psnr','ssim','tce','brightness_flicker','runtime_ms']} for n in sorted({r['method'] for r in rows})}
    for fname,recs in [('per_frame.csv',rows),('per_scene.csv',seqs)]:
        with (out/fname).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(recs[0]));w.writeheader();w.writerows(recs)
    summary=dict(scenes=scenes,frames_per_scene=frames,resolution=[960,512],methods=methods,candidates=candidates,
                 timing='Wall clock for shared network + method-specific processing; excludes metrics/IO, includes first-frame warm-up; not a standalone comparative speed benchmark')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(methods,indent=2),flush=True)
    return summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['validation','test','freeze','confirmation'],required=True)
    ap.add_argument('--frames',type=int,default=30);a=ap.parse_args()
    initial=json.loads((ROOT/'records/selected_config.json').read_text())
    if a.stage=='freeze':
        source=ROOT/'records/initial_config.json'
        if not source.exists():source=ROOT/'records/final_config.json'
        c=json.loads(source.read_text());c['gamma']=1.
        c['confirmation_scenes']=[f'pair{i}' for i in range(7,13)]
        c['demonstration_scenes']=c['confirmation_scenes']+c['test_scenes']
        c['selection']='Fixed gamma=1 after exposure calibration failed on the initial 12-scene extrapolation check; retain beta=.85 from validation; evaluate ONCE on untouched pair7-pair12'
        c['initial_test_role']='Initial 12 scenes became diagnostic after revealing exposure-shift failure; they are not an untouched final test set for the revised method'
        c['training']='Official network unchanged; beta selected on validation; gamma=1 fixed after diagnostic failure and before independent confirmation'
        (ROOT/'records/final_config.json').write_text(json.dumps(c,indent=2),encoding='utf-8')
        print('Revised configuration frozen before confirmation',flush=True)
        return
    if a.stage=='validation':
        gamma=initial['gamma']
        candidates={f'FlowRGB_{b:g}':dict(beta=b,align=True,gamma=gamma) for b in [.3,.6,.85]}
        candidates.update({'Exposure':dict(beta=0,gamma=gamma),'Spatial':dict(beta=0,gamma=gamma,spatial=True),
            'NaiveRGB':dict(beta=.6,align=False,gamma=gamma),
            'Curve':dict(beta=0,curve_beta=initial['beta'],gamma=gamma),
            'Gamma':dict(beta=0,gamma_input=initial['gamma_baseline'])})
        s=evaluate(VALIDATION,a.frames,candidates,ROOT/'results/validation_temporal')
        # Primary quality endpoint fixed before reading test results.
        chosen=max([n for n in candidates if n.startswith('FlowRGB_')],key=lambda n:s['methods'][n]['psnr'])
        final=dict(candidates[chosen],curve_beta=initial['beta'],gamma_baseline=initial['gamma_baseline'],
                   validation_scenes=VALIDATION,test_scenes=TEST,selection='highest validation scene-mean PSNR among flow RGB beta=0.3/0.6/0.85',
                   training='No neural network retraining; official Zero-DCE checkpoint; postprocessing selected on validation only')
        (ROOT/'records/final_config.json').write_text(json.dumps(final,indent=2),encoding='utf-8')
        (ROOT/'records/initial_config.json').write_text(json.dumps(final,indent=2),encoding='utf-8')
        print('FROZEN',final,flush=True)
    elif a.stage=='test':
        path=ROOT/'records/final_config.json';c=json.loads(path.read_text())
        candidates={'Improved':dict(beta=c['beta'],align=True,gamma=c['gamma']),
                    'Exposure':dict(beta=0,gamma=c['gamma']),
                    'FlowOnly':dict(beta=c['beta'],align=True,gamma=1.),
                    'NaiveRGB':dict(beta=c['beta'],align=False,gamma=c['gamma']),
                    'Spatial':dict(beta=0,gamma=c['gamma'],spatial=True),
                    'Curve':dict(beta=0,curve_beta=c['curve_beta'],gamma=c['gamma']),
                    'Gamma':dict(beta=0,gamma_input=c['gamma_baseline'])}
        (ROOT/'records/final_test_config_sha256.txt').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
        evaluate(TEST,a.frames,candidates,ROOT/'results/test',save=True)
    else:
        path=ROOT/'records/final_config.json';c=json.loads(path.read_text())
        (ROOT/'records/confirmation_frozen_config.json').write_bytes(path.read_bytes())
        candidates={'Final':dict(beta=c['beta'],align=True,gamma=1.),
                    'NaiveRGB':dict(beta=c['beta'],align=False,gamma=1.),
                    'Spatial':dict(beta=0,gamma=1.,spatial=True),
                    'RejectedExposure':dict(beta=c['beta'],align=True,gamma=.6),
                    'Gamma':dict(beta=0,gamma_input=c['gamma_baseline'])}
        (ROOT/'records/confirmation_config_sha256.txt').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
        evaluate(c['confirmation_scenes'],a.frames,candidates,ROOT/'results/confirmation',save=True)


if __name__=='__main__':main()
