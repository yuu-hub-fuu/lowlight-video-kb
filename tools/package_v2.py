"""原地整理第二版交付包并打 ZIP。

- 第一次运行时把第一版报告/视频备份到交付目录外的 “_第一版备份”；
- 报告、视频复制到模板要求的位置；
- ZIP 中排除：完整数据(只保留 2 个场景各 5 帧样例)、缓存、日志、中间权重、__pycache__。
"""
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT.parents[1]                       # 【图像处理】-待填学号-待填姓名
DELIVERY = STAGE.parent
VIDEO = STAGE / "2.演示视频"
BACKUP = DELIVERY / "_第一版备份"


def backup_v1():
    if BACKUP.exists(): return
    BACKUP.mkdir()
    for p in [STAGE / "1.报告.docx", STAGE / "1.报告.pdf", STAGE / "提交前说明.txt", STAGE / "文件校验.json", *VIDEO.glob("*")]:
        if p.exists(): shutil.copy2(p, BACKUP / p.name)
    print("v1 backup ->", BACKUP)


def keep(p: Path) -> bool:
    rel = p.relative_to(ROOT).as_posix() if ROOT in p.parents else None
    if rel is None: return True
    if "__pycache__" in rel or rel.endswith(".pyc"): return False
    if rel.startswith(("data/cache/", "logs/", "results_v2/val/examples/", ".venv/")): return False
    if rel.startswith("output/") and rel.endswith((".mp4", ".pdf", ".docx")): return False   # 已复制到模板位置，避免重复
    if rel.startswith("weights/") and (rel.endswith("_last.pth") or "sweep" in rel or "diag" in rel): return False
    if rel.startswith("data/raw/"):
        parts = rel.split("/")                    # data/raw/indoor_np/<kind>/<scene>/<file>
        if len(parts) < 6 or parts[4] not in ("pair1", "pair13"): return False
        return parts[5] in sorted(x.name for x in (ROOT / "/".join(parts[:5])).glob("*.npy"))[:5]
    return True


def main():
    backup_v1()
    for name in ("1.报告.docx", "1.报告.pdf"): shutil.copy2(ROOT / "output" / name, STAGE / name)
    for p in VIDEO.glob("SDSD_*.mp4"): p.unlink()
    for p in (ROOT / "output").glob("KBG_*.mp4"): shutil.copy2(p, VIDEO / p.name)
    meta = json.loads((ROOT / "output/video_metadata_v2.json").read_text(encoding="utf-8"))
    meta["主视频"] = dict(file="KBG_讲解演示视频.mp4", duration_seconds=230.2, resolution="1280x720@25fps",
                       audio="中文配音（Windows 本地语音合成 Microsoft Huihui）+ 烧录字幕", script="make_presentation.py")
    (VIDEO / "视频说明.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (STAGE / "提交前说明.txt").write_text(
        "本包按老师模板提供：1.报告（docx/pdf）、2.演示视频、3.代码附件。\n"
        "提交前请把文件夹/ZIP 名称中的“待填学号”“待填姓名”换成本人信息，并填写报告首页。\n"
        "优先观看：2.演示视频/KBG_讲解演示视频.mp4（约4分钟，中文配音+字幕：问题、数据、方法、效果对比、定量结果）。\n"
        "附加素材：KBG_四栏对比_18场景.mp4（暗光输入/Zero-DCE/本文/正常光参考）、KBG_算法过程_6场景.mp4（中间结果）。\n"
        "第二版方法 KBG-Video：噪声建模卡尔曼去噪 + 可学习双边网格 + 算子域时域平滑 + 曝光回归；网络为本项目训练。\n"
        "代码包含训练好的权重与 2 个场景各 5 帧样例数据，可先运行 python -m unittest discover -s tests。\n"
        "完整数据可运行 tools/download_sdsd.py 从作者公开源获取（命令见 README.md）。\n"
        "第一版报告与视频备份在交付目录的“_第一版备份”文件夹，不在 ZIP 内。\n"
        "邮件发送与腾讯文档登记由本人进行，本包未执行任何对外提交。\n", encoding="utf-8")
    files = [p for p in STAGE.rglob("*") if p.is_file() and keep(p) and p.name != "文件校验.json"]
    manifest = {p.relative_to(STAGE).as_posix(): dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                for p in sorted(files)}
    (STAGE / "文件校验.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    files.append(STAGE / "文件校验.json")
    zip_path = DELIVERY / (STAGE.name + ".zip")
    if zip_path.exists() and not (BACKUP / zip_path.name).exists(): shutil.copy2(zip_path, BACKUP / zip_path.name)
    with ZipFile(zip_path, "w", ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(files): z.write(p, p.relative_to(DELIVERY).as_posix())
    with ZipFile(zip_path) as z: assert z.testzip() is None
    (DELIVERY / "交付校验.json").write_text(json.dumps(dict(zip=str(zip_path), bytes=zip_path.stat().st_size,
        sha256=hashlib.sha256(zip_path.read_bytes()).hexdigest(), crc_verified=True, files=len(files)), ensure_ascii=False, indent=2), encoding="utf-8")
    print("PACKAGE", zip_path, f"{zip_path.stat().st_size/1e6:.1f} MB", len(files), "files")


if __name__ == "__main__":
    main()
