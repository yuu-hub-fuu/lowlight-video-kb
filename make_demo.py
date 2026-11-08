"""Make a 4-panel video from real SDSD sequences; playback FPS is explicitly chosen.

All examples use the predeclared first 30 frames of every test scene, in fixed
order. No cherry-picked frames. These playback videos have no source audio.
"""
import json
from pathlib import Path
import imageio_ffmpeg
import numpy as np
import torch
import cv2
from PIL import Image,ImageDraw,ImageFont
from experiment import pairs
from lowlight import ROOT,ImageState,flow_context,image_numpy,load_model,quantize,tensor_image,save_rgb


def writer(path,size,fps=10):
    w=imageio_ffmpeg.write_frames(str(path),size,fps=fps,codec='libx264',quality=8,
        macro_block_size=2,output_params=['-movflags','+faststart']);w.send(None);return w


def main():
    conf=json.loads((ROOT/'records/final_config.json').read_text())
    out=ROOT/'output';out.mkdir(exist_ok=True)
    comparison=writer(out/'SDSD_四栏对比_18场景.mp4',(960,592))
    source_writer=writer(out/'SDSD_暗光输入_18场景.mp4',(960,512))
    enhanced_writer=writer(out/'SDSD_最终输出_18场景.mp4',(960,512))
    net,dev=load_model();torch.set_num_threads(6);cv2.setNumThreads(4)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18) if Path('C:/Windows/Fonts/arial.ttf').exists() else ImageFont.load_default()
    count=0
    try:
        for scene in conf['demonstration_scenes']:
            state=ImageState(conf['beta']);prev=None
            for i,_,_,rgb,gt in pairs(scene,30):
                context=flow_context(rgb,prev) if prev is not None else None
                with torch.inference_mode():
                    _,base,_=net(tensor_image(rgb,dev))
                    improved=state.update(base.clamp(0,1).pow(conf['gamma']),context)
                im=image_numpy(improved)
                panels=[rgb,image_numpy(base),im,gt]
                canvas=Image.new('RGB',(960,592),'#ffffff');draw=ImageDraw.Draw(canvas)
                for k,(panel,label) in enumerate(zip(panels,['Low-light input','Zero-DCE baseline','Our video processing','Normal-light reference'])):
                    x,y=(k%2)*480,(k//2)*296
                    thumb=Image.fromarray(quantize(panel)).resize((480,256),Image.Resampling.LANCZOS)
                    canvas.paste(thumb,(x,y+40));draw.text((x+8,y+10),f'{label} | {scene} #{i:02d}',font=font,fill='#111111')
                comparison.send(np.ascontiguousarray(canvas));source_writer.send(np.ascontiguousarray(quantize(rgb)));enhanced_writer.send(np.ascontiguousarray(quantize(im)))
                if i==15:
                    canvas.save(out/f'{scene}_四栏对比.png')
                prev=rgb;count+=1
            print('Video',scene,flush=True)
    finally:
        comparison.close();source_writer.close();enhanced_writer.close()
    (out/'video_metadata.json').write_text(json.dumps(dict(frames=count,fps=10,duration_seconds=count/10,
        source='SDSD indoor_np, original paired chronological frames',playback='10fps chosen for teaching; original acquisition FPS not inferred',
        scenes=conf['demonstration_scenes'],frames_per_scene=30,audio='None: source data distributed as image sequences'),indent=2),encoding='utf-8')


if __name__=='__main__':main()
