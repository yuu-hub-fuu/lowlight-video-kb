"""Package the course deliverables and a separate complete experimental subset."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
DELIVERY=ROOT.parent/'course_report_20261104/delivery'
NAME='【图像处理】-待填学号-待填姓名'


def data_archive():
    DELIVERY.mkdir(parents=True,exist_ok=True)
    target=DELIVERY/'SDSD_本次实验数据_24场景720对.zip'
    with ZipFile(target,'w',ZIP_DEFLATED,compresslevel=1) as z:
        files=sorted((ROOT/'data/raw/indoor_np').glob('*/*/*.npy'))
        assert len(files)==1440
        for i,p in enumerate(files,1):
            z.write(p,p.relative_to(ROOT).as_posix())
            if i%200==0:print('Dataset archive',i,'/',len(files),flush=True)
        z.write(ROOT/'records/download_manifest.json','records/download_manifest.json')
        z.writestr('数据说明.txt','SDSD作者公开数据的本次实验子集：24个场景，每场景前30组配对帧。\n原始.npy数组，960×512，BGR uint8。\n数据来源：https://github.com/JIA-Lab-research/SDSD\n解压到项目根目录可直接重现实验。原作者数据权利保留。\n')
    with ZipFile(target) as z:assert z.testzip() is None
    print('Dataset archive verified',target.stat().st_size,flush=True)


def package():
    assert (ROOT/'output/1.报告.docx').exists()
    assert (ROOT/'output/1.报告.pdf').exists()
    stage=DELIVERY/NAME;code=stage/'3.可选附件或代码/lowlight_video_project'
    code.mkdir(parents=True,exist_ok=True)
    def copy(p,dest):dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for p in ROOT.iterdir():
        if p.is_file() and p.suffix in ['.py','.md','.txt','.ps1']:copy(p,code/p.name)
    for directory in ['tools','tests','vendor','weights','records','template']:
        for p in (ROOT/directory).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':copy(p,code/p.relative_to(ROOT))
    for p in (ROOT/'results').rglob('*'):
        if p.is_file() and p.suffix in ['.json','.csv'] and 'examples' not in p.parts:
            copy(p,code/p.relative_to(ROOT))
    for scene in ['pair1','pair13']:
        for kind in ['input','GT']:
            for p in sorted((ROOT/'data/raw/indoor_np'/kind/scene).glob('*.npy'))[:5]:copy(p,code/p.relative_to(ROOT))
    copy(ROOT/'results/integration/input_silent.mp4',code/'data/sample_video.mp4')
    for p in (ROOT/'output').glob('*.docx'):copy(p,stage/'1.报告.docx')
    copy(ROOT/'output/1.报告.pdf',stage/'1.报告.pdf')
    for p in (ROOT/'output').glob('*.mp4'):copy(p,stage/'2.演示视频'/p.name)
    for p in (ROOT/'output').glob('*.png'):copy(p,code/'output'/p.name)
    copy(ROOT/'output/video_metadata.json',stage/'2.演示视频/视频说明.json')
    copy(ROOT/'output/报告正文.md',code/'output/报告正文.md')
    (stage/'提交前说明.txt').write_text(
        '本包按老师模板提供报告、视频和可选代码。\n'
        '提交前请将文件夹/ZIP名称中的“待填学号”“待填姓名”替换为本人信息，并填写报告首页。\n'
        '优先观看：2.演示视频/SDSD_四栏对比_18场景.mp4。\n'
        '代码包带原始预训练权重和2个场景各5组配对帧，可先运行 experiment.py --stage smoke。\n'
        '全部实验数据另存于同目录 SDSD_本次实验数据_24场景720对.zip；解压到代码项目根目录即可。\n'
        '也可运行 tools/download_sdsd.py 从作者公开源获取完整实验子集。\n'
        '本次没有重新训练网络；改进属于公开模型之上的视频后处理与实验分析。\n'
        '邮件发送与腾讯文档登记由本人进行，本包未执行对外提交。\n',encoding='utf-8')
    manifest={}
    for p in stage.rglob('*'):
        if p.is_file() and p.name!='文件校验.json':manifest[p.relative_to(stage).as_posix()]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    (stage/'文件校验.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    zip_path=DELIVERY/(NAME+'.zip')
    with ZipFile(zip_path,'w',ZIP_DEFLATED,compresslevel=3) as z:
        for p in stage.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(DELIVERY).as_posix())
    with ZipFile(zip_path) as z:assert z.testzip() is None
    (DELIVERY/'交付校验.json').write_text(json.dumps(dict(zip=str(zip_path),bytes=zip_path.stat().st_size,
        sha256=hashlib.sha256(zip_path.read_bytes()).hexdigest(),crc_verified=True,files=len(manifest)+1),ensure_ascii=False,indent=2),encoding='utf-8')
    print('COURSE PACKAGE',zip_path,'bytes',zip_path.stat().st_size,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data-only',action='store_true');a=ap.parse_args()
    data_archive() if a.data_only else package()
