"""由实验记录生成第二版课程报告（老师模板的五节结构），所有数字从 records/ 与 results_v2/ 读取。"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REFERENCE = ROOT / "template/1.报告.docx"
EXPECTED = "0c82a0e2301dfe9a16de16dd28df672b558c2e0edd42306fafba5c29025b29de"


def jload(p): return json.loads((ROOT / p).read_text(encoding="utf-8"))


def main():
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == EXPECTED
    cfg = jload("records/v2_config.json"); test = jload("results_v2/test/summary.json")["methods"]
    val = jload("results_v2/val/summary.json")["methods"]
    noise = jload("records/noise_model.json"); ksel = jload("records/kalman_selection.json"); expo = jload("records/exposure_model.json")
    tr = jload("records/train_bgnet_kf.json"); tr_lq = jload("records/train_bgnet_lq.json")
    lr_kf = jload("records/longrun_bgnet_kf_4000iters.json"); lr_dir = jload("records/longrun_bgnet_kf_direct_4000iters.json")
    OURS = cfg["final_method"]; GRID = f"KBG-Video(α={cfg['alpha']:g})"; vf = cfg["final_val"]
    o, z, v1 = test[OURS], test["Zero-DCE"], test["Zero-DCE+运动滤波(v1)"]
    per = list(csv.DictReader((ROOT / "results_v2/test/per_scene.csv").open(encoding="utf-8-sig")))
    scenes = list(dict.fromkeys(r["scene"] for r in per))
    ps = lambda m, k: np.array([float(next(r[k] for r in per if r["scene"] == s and r["method"] == m)) for s in scenes])

    def boot(d, n=10000):
        rng = np.random.default_rng(0); m = rng.choice(d, (n, len(d))).mean(1)
        return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)

    d = Document(REFERENCE); body = d._element.body
    for el in list(body):
        if el.tag != qn("w:sectPr"): body.remove(el)
    for name, size, font in [("Normal", 11, "宋体"), ("Heading 1", 14, "黑体"), ("Heading 2", 11.5, "黑体"), ("Title", 19, "黑体")]:
        s = d.styles[name]; s.font.name = font; s.font.size = Pt(size); s.font.color.rgb = RGBColor(0, 0, 0)
        s._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), font)
        s.paragraph_format.space_after = Pt(6); s.paragraph_format.line_spacing = 1.25; s.paragraph_format.widow_control = True
        if name.startswith("Heading"): s.paragraph_format.space_before = Pt(10); s.paragraph_format.keep_with_next = True
    md = []

    def p(text, indent=True):
        q = d.add_paragraph()
        if indent: q.paragraph_format.first_line_indent = Pt(22)
        q.add_run(text); md.append(text + "\n"); return q

    def eq(text):
        q = d.add_paragraph(); q.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = q.add_run(text); r.font.name = "Cambria Math"; r.font.size = Pt(10.5); md.append(f"    {text}\n")

    def h(text, level=1): d.add_heading(text, level); md.append("#" * (level + 1) + " " + text + "\n")

    def table(headers, rows, widths, bold_row=None):
        t = d.add_table(rows=1, cols=len(headers)); t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
        for col, w_ in zip(t.columns, widths): col.width = Inches(w_)
        for c, s in zip(t.rows[0].cells, headers): c.text = str(s)
        for row in rows:
            for c, s in zip(t.add_row().cells, row): c.text = str(s)
        for ir, row in enumerate(t.rows):
            pr = row._tr.get_or_add_trPr()
            if ir == 0: pr.append(OxmlElement("w:tblHeader"))
            pr.append(OxmlElement("w:cantSplit"))
            for j, c in enumerate(row.cells):
                c.width = Inches(widths[j]); c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                tc = c._tc.get_or_add_tcPr(); b = OxmlElement("w:tcBorders")
                for side in ["top", "left", "bottom", "right"]:
                    e = OxmlElement("w:" + side); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:color"), "D9D9D9"); b.append(e)
                tc.append(b)
                mar = OxmlElement("w:tcMar")
                for side in ["top", "left", "bottom", "right"]:
                    e = OxmlElement("w:" + side); e.set(qn("w:w"), "60"); e.set(qn("w:type"), "dxa"); mar.append(e)
                tc.append(mar)
                if ir == 0 or (bold_row is not None and ir == bold_row):
                    sh = OxmlElement("w:shd"); sh.set(qn("w:fill"), "EDEDED" if ir == 0 else "EAF2FC"); tc.append(sh)
                for para in c.paragraphs:
                    para.paragraph_format.space_after = Pt(1); para.paragraph_format.line_spacing = 1.05
                    para.paragraph_format.first_line_indent = Pt(0)
                    if j > 0: para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for r in para.runs: r.font.size = Pt(9); r.bold = ir == 0 or ir == bold_row
        d.add_paragraph().paragraph_format.space_after = Pt(0)
        md.append("| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |")
        md.extend("| " + " | ".join(map(str, r)) + " |" for r in rows); md.append("")

    def image(path, caption, width=5.7):
        q = d.add_paragraph(); q.alignment = WD_ALIGN_PARAGRAPH.CENTER; q.paragraph_format.keep_with_next = True
        q.paragraph_format.first_line_indent = Pt(0)
        q.add_run().add_picture(str(ROOT / path), width=Inches(width))
        q = d.add_paragraph(caption); q.alignment = WD_ALIGN_PARAGRAPH.CENTER; q.paragraph_format.first_line_indent = Pt(0)
        for r in q.runs: r.font.size = Pt(9)
        md.append(f"![{caption}]({path})\n")

    f2, f3, f4, f5 = lambda x: f"{x:.2f}", lambda x: f"{x:.3f}", lambda x: f"{x:.4f}", lambda x: f"{x:.5f}"

    # ———————————————————— 标题与摘要 ————————————————————
    t = d.add_paragraph("KBG-Video：噪声建模卡尔曼去噪与可学习双边网格的低光照视频增强", style="Title"); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for text in ["计算机图像处理课程报告", "学号：________    姓名：________    日期：2026年11月7日"]:
        q = d.add_paragraph(text); q.alignment = WD_ALIGN_PARAGRAPH.CENTER
    dp = o["psnr"] - z["psnr"]; dssim = o["ssim"] - z["ssim"]
    p(f"摘要：低光照视频增强同时面临三个问题：提亮会把暗部噪声成倍放大；参考图像应当“多亮”并不完全由输入决定；逐帧处理会产生闪烁，而直接做帧间平均又会产生拖影。"
      f"本项目提出四级在线流水线 KBG-Video：①在提亮之前，用实测的泊松-高斯噪声模型做运动补偿的逐像素卡尔曼滤波，并以新息检验自动识别遮挡；"
      f"②用轻量网络预测 16×16×8 的双边网格，每格是一个 3×4 颜色仿射变换，相当于学习“随位置和亮度变化的灰度变换”；"
      f"③用只有两个参数的曝光回归决定整体亮度；④用光流置信度门控的运动补偿融合抑制残余闪烁（与“平滑网格系数”的方案在验证集上比较后选定）。"
      f"网络在 34 个训练场景上训练、在 6 个验证场景上早停和选参，"
      f"最后在 SDSD 作者官方的 12 个室内测试场景上评测一次。相对 Zero-DCE，PSNR 变化 {dp:+.2f} dB，SSIM 变化 {dssim:+.3f}；"
      f"扭曲误差由 {z['warp_error']:.4f} 降至 {o['warp_error']:.4f}。报告同时给出噪声标定、参数选择、消融和失败分析。")
    p("关键词：低光照视频增强；泊松-高斯噪声；卡尔曼滤波；双边网格；时域一致性；SDSD")

    # ———————————————————— 1 ————————————————————
    h("1. GitHub 链接")
    p("对照方法 Zero-DCE：https://github.com/Li-Chongyi/Zero-DCE（固定版本 e0f4adc54d0f，原网络与权重未改动）[1]。")
    p("数据集 SDSD：https://github.com/JIA-Lab-research/SDSD [2]。双边网格思想来自 HDRNet [3]，本项目为独立的轻量实现。")
    p("本项目代码随附件提供（3.可选附件或代码/lowlight_video_project），运行方式见 README.md，来源与许可见 THIRD_PARTY.md。")

    # ———————————————————— 2 ————————————————————
    h("2. 任务描述：要解决什么问题")
    p("输入一段低光照彩色视频，逐帧在线（只用当前帧和过去帧）输出增强后的视频，要求看得清、噪声少、颜色正确、帧间不闪烁也不拖影。")
    table(["难点", "现象", "本项目的对策", "课程知识点"], [
        ["噪声放大", "暗光帧需提亮约 5 倍，噪声随之放大", "提亮前在原始域做运动补偿卡尔曼滤波", "噪声模型、图像复原、最优估计"],
        ["映射形状", "全局幂律/直方图均衡无法兼顾亮部与暗部、颜色", "学习空间可变、保边的分段颜色变换（双边网格）", "灰度变换、查找表、双边滤波"],
        ["时域一致", "逐帧独立处理闪烁；直接像素平均产生拖影", "光流对齐 + 置信度门控融合；比较“平滑变换参数”方案", "运动估计、时域滤波"],
        ["曝光不确定", "参考图亮度在场景间差约 25%，无法由输入预测", "曝光由两参数回归单独负责，网络学习形状", "监督学习的泛化"],
    ], [0.95, 1.85, 1.95, 1.25])
    p("课程关联：《第3章 图像增强》的灰度变换、直方图处理和空间滤波（本项目实现为对照基线并作为网络的初始化），"
      "《第3章-2 深度学习图像增强》的低光照增强（Zero-DCE 为对照），以及图像复原中的噪声模型与最优滤波。")

    # ———————————————————— 3 ————————————————————
    h("3. 数据描述：数据是什么")
    p("SDSD（ICCV 2021）通过机械臂重复相同运动，分别采集暗光与正常光视频并对齐，得到真实的配对连续帧 [2]。本项目使用 indoor_np："
      "每帧 960×512、uint8、BGR 的 .npy 数组，读入后转为 RGB 并归一化到 [0,1]。暗光与参考序列各自按文件名排序后按位置配对，与作者代码一致。"
      "下载器从作者公开的分卷压缩包按需读取成员并校验 CRC32。")
    table(["用途", "场景", "帧数", "作用"], [
        ["训练", "pair1–6、14、16–20、22、24–30、32、34–46（作者训练池，共34个）", "740", "噪声标定、卡尔曼选参、网络训练、曝光回归"],
        ["验证", "pair7–12（6个）", "180", "早停；α、β 选择；然后冻结配置"],
        ["测试", "pair13、15、21、23、31、33、50、52、58、60、68、70（作者官方室内测试划分）", "360", "冻结后只评测一次"],
    ], [0.55, 3.15, 0.55, 1.75])
    p("同一视频的帧只属于一个集合，不存在相邻帧泄漏。说明：第一版作业曾在这 12 个测试场景上查看过 Zero-DCE 系列的结果；"
      "第二版的所有参数（噪声模型、卡尔曼参数、网络权重、曝光回归、α、β）均只由训练集和验证集确定，测试集在冻结配置后只运行一次。"
      "每个场景只评前 30 帧，结果不等同于完整 SDSD 榜单。")
    stats = []
    for name, S in (("训练", list(jload("records/train_bgnet_kf.json")["train_scenes"])), ("验证", cfg["val_scenes"]), ("测试", cfg["test_scenes"])):
        stats.append(name)
    p(f"数据分析：在训练场景中，参考图平均亮度与暗光输入平均亮度的相关系数只有 {expo['train_corr_scene_means']:.2f}。"
      "也就是说，“正常光参考应该多亮”有相当一部分无法从输入推断。这一点直接影响了第 4.4 节的设计。")
    image("output/figures/fig_noise.png", f"图1　噪声标定（训练场景）：对齐后相邻帧差分的稳健方差随亮度近似线性增长，拟合 σ²=a·x+b。", 4.3)

    # ———————————————————— 4 ————————————————————
    h("4. 代码修改描述：我们改了什么")
    p("第一版只在 Zero-DCE 输出上加光流时域滤波（PSNR 提升 0.09 dB）。第二版重写了处理链路，新增约 1500 行代码，全部模块均有单元测试。"
      "整条流水线逐帧在线运行，只使用当前帧与过去帧：")
    eq("z_t →① 卡尔曼去噪 x̂_t →② BGNet 网格切片 y_t →③ 曝光回归缩放 →④ 运动补偿置信度融合 → 输出")
    image("output/" + next(p_.name for p_ in sorted((ROOT / "output").glob(f"{cfg['test_scenes'][0]}_算法过程.png"))),
          f"图2　算法中间结果（测试场景 {cfg['test_scenes'][0]} 第16帧）。上排：暗光输入×4、卡尔曼去噪后×4、卡尔曼增益 K（越暗表示越多使用历史帧）；"
          "下排：BGNet+曝光回归的单帧输出、运动补偿融合后的最终输出、正常光参考。", 6.0)

    h("4.1 ① 噪声建模与运动补偿卡尔曼滤波（kalman.py）", 2)
    p("噪声标定（tools/calibrate_noise.py）：在训练场景中，用光流对齐相邻两帧，取前后向一致性高的像素做差分，按亮度分箱。"
      "用中位数绝对偏差（MAD）稳健估计方差，以抑制光流残差的影响，再拟合泊松（散粒）加高斯（读出）模型：")
    eq(f"Var(n | x) = a·x + b,   a = {noise['a']:.3e},   b = {noise['b']:.3e}")
    p(f"在把模型用于检验之前，我们还发现 SDSD 的噪声有明显的空间相关性：3×3 均值后的方差是原来的 κ = {noise['kappa']:.2f} 倍，"
      "而独立同分布噪声应为 1/9 ≈ 0.11。若忽略这一点，后面的新息检验会把大量正常像素误判为“运动”。因此 κ 也作为标定参数写入模型。")
    p("滤波过程：状态是“该像素真实的暗光亮度”。每帧先用暗光帧间光流把上一帧的估计 x̂ 和方差 P 变形到当前帧，再按卡尔曼增益融合当前观测：")
    eq("x⁻ = W(x̂_{t−1}),   P⁻ = W(P_{t−1}) + Q,   R = a·z̄ + b")
    eq("ρ = exp( − d̄² / (τ²·κ·(P⁻+R)) ),   K = ρ·P⁻/(P⁻+R) + (1−ρ)")
    eq("x̂_t = x⁻ + K·(z_t − x⁻),   P_t = K·R")
    p("其中 d̄ 是 3×3 平均后的新息（观测与预测之差）。静止且对齐可靠的区域，K 会逐帧减小，相当于沿运动轨迹越平均越久；"
      "遮挡、光流失败或镜头切换处，新息超出噪声模型允许的范围，ρ→0、K→1，自动退回当前帧，不会产生拖影。"
      "与固定系数的帧平均不同，每个像素的融合比例由噪声大小和预测不确定度共同决定，这正是维纳/卡尔曼最优估计的思想。")
    c = ksel["chosen"]; ema85 = next(r for r in ksel["controls"] if r["kind"] == "ema" and r["beta"] == 0.85)
    iid = next(r for r in ksel["controls"] if r["kind"] == "kalman_iid_kappa")
    p(f"参数选择（tools/tune_kalman.py，只用训练场景）：以“去噪后经固定幂律提亮的 PSNR 与 SSIM”为代理指标，在 q0、q1、τ 的 20 组组合中，"
      f"按“PSNR 不低于最优 0.01 dB 以内者取 SSIM 最高”的事先规则选出 q0={c['q0']}、q1={c['q1']}、τ={c['tau']}。"
      f"同样参数下，若按独立噪声取 κ=1/9，代理 SSIM 从 {c['ssim']:.4f} 降至 {iid['ssim']:.4f}，验证了 κ 标定的作用（图3）。")
    image("output/figures/fig_kalman_selection.png", "图3　卡尔曼参数网格与固定系数 EMA 的代理指标对比（训练场景）。卡尔曼配置构成的前沿在相同 SSIM 下 PSNR 更高。", 4.3)

    h("4.2 ② 可学习双边网格 BGNet（bgnet.py）", 2)
    p("网络只看 256×256 的缩略图：4 层步长为 2 的卷积得到 16×16 特征，局部分支与全局平均池化分支相加后，"
      "输出 16×16 个空间格 × 8 个亮度区间、每格 12 个数（3×4 颜色仿射矩阵）。全分辨率上，一个逐点小网络给出引导图 g∈[0,1]，"
      "对每个像素按 (x, y, g) 在网格中做三线性插值（“切片”），得到该像素自己的变换：")
    eq("out(p) = A(p, g(p))·u(p) + b(p, g(p)),   u = x̂^0.4")
    p("与课程的联系：每个亮度区间一个变换，本质上是一张随位置缓变的分段灰度变换/颜色查找表；沿亮度维插值使亮度差大的相邻像素使用不同变换，"
      "这与双边滤波“空间近且灰度近才相互影响”的保边原理相同。网格自由度只有 16×16×8×12，"
      "只能表达平滑的色调映射，不会凭空生成纹理，也不会放大噪声的空间结构。网络参数共 "
      f"{tr['params']/1e3:.0f} K，960×512 单帧推理约数毫秒。")
    p("初始化：网格每格设为 A=1.5·I、b=0，网络在训练前严格等于传统幂律变换 s=1.5·r^0.4（单元测试验证），训练只学习相对传统变换的修正。")
    image("output/figures/fig_tone_curves.png", "图4　从训练好的网格中读出的色调曲线（曝光缩放之前）。学到的映射比传统幂律更陡：暗部压得更深、中间调提得更高；两处位置的曲线几乎重合，说明在现有数据量下网络主要学到全局色调形状，空间可变能力没有被充分利用。", 4.3)

    h("4.3 时域稳定：两种方案的验证集比较", 2)
    vg, vk = val[GRID], val["卡尔曼+BGNet"]
    p("方案 A（算子域平滑）：对网格系数做指数平滑 Γ_t = αΓ_{t−1} + (1−α)Γ̂_t，再用当前帧的引导图切片。网格是 32×60 像素量级的低频变换，"
      "相机运动几个像素时几乎不变，因此理论上不会产生拖影。"
      "方案 B（像素域运动补偿融合）：用暗光帧间光流把上一帧输出对齐到当前帧，以光度残差、前后向一致性构成的置信度 c 门控融合：")
    eq("out_t = (1 − β·c)·y_t + β·c·W(out_{t−1})")
    p(f"两者的系数都在验证集 {{0.5, 0.7, 0.85, 0.95}} 中选择。验证结果：方案 A 在最优 α={cfg['alpha']:g} 时扭曲误差只从 {vk['warp_error']:.5f} 变为 {vg['warp_error']:.5f}，"
      f"几乎没有作用；方案 B 在 β={cfg['beta_pixel']:g} 时扭曲误差降到 {vf['warp_error']:.5f}，SSIM 升到 {vf['ssim']:.4f}，验证 PSNR 也最高。"
      "原因是：经过卡尔曼去噪和网格映射后，剩余的帧间变化主要是像素级的残余噪声，而不是整体色调的抖动，平滑变换参数管不到它。"
      "因此按“验证 PSNR 最高”的规则，最终流水线采用方案 B（第一版的运动补偿模块在新流水线中复用），方案 A 作为消融保留。"
      "这一选择在读取测试集之前完成并写入 records/v2_config.json。镜头切换时卡尔曼与融合状态同时重置。")

    h("4.4 曝光回归与曝光解耦训练", 2)
    p(f"直接用 L1 监督训练时，验证集 PSNR 往往在训练早期最好，之后持续下降（图5）。原因是第 3 节的数据分析：参考亮度有相当部分无法由输入预测，"
      "网络会借助画面内容“记住”每个训练场景应有的亮度。因此我们把“整体多亮”与“如何映射”分开：")
    eq(f"log μ* = c0 + c1·log μ_in,   c0 = {expo['c0']:.3f},   c1 = {expo['c1']:.3f}")
    cv = expo["loso_cv_mean_abs_log_error"]
    p(f"三种候选模型（常数目标、固定增益、对数线性）用训练场景留一交叉验证选择，平均对数误差分别为 {cv['M0']:.3f}、{cv['M1']:.3f}、{cv['M2']:.3f}，"
      f"选中对数线性模型。斜率 c1≈{expo['c1']:.2f} 表明参考亮度与输入亮度近似满足 γ≈{expo['c1']:.2f} 的幂律关系，与传统幂律提亮的经验一致。"
      "训练 BGNet 时，先把输出整体缩放到参考图均值再计算 L1 + SSIM 损失（缩放系数不回传梯度），再加 0.05 倍原始 L1 防止亮度漂移。"
      "这样网络只学色调形状、颜色和局部对比。推理时由曝光回归给出每帧的目标亮度，把网络输出整体缩放到该亮度。")
    best_kf = max(tr["log"], key=lambda r: r["val_psnr"])
    p(f"即使解耦后，验证指标仍在训练早期达到最好。因此最终采用验证集早停：每 50 步验证一次，取验证 PSNR 最高的第 {tr['best_iter']} 步权重"
      f"（验证 PSNR {best_kf['val_psnr']:.2f} dB）。这说明 34 个场景仍然偏少，网络的有效作用是“在传统幂律变换附近做有限修正”。我们如实报告这一点。")
    image("output/figures/fig_training.png", "图5　长时间训练的验证曲线：直接监督与解耦训练都在早期达到最好，之后对训练场景过拟合。", 4.3)

    h("4.5 对照方法与代码文件", 2)
    table(["文件", "内容"], [
        ["kalman.py / tools/calibrate_noise.py / tools/tune_kalman.py", "① 噪声模型与 κ 标定、卡尔曼滤波、固定系数 EMA 对照、参数选择"],
        ["bgnet.py / train_bgnet.py", "② 双边网格网络、网格时域平滑（方案 A）、曝光解耦训练、验证早停"],
        ["kbg.py / tools/fit_exposure.py", "完整在线流水线 KBGVideo、③ 曝光回归（留一交叉验证）、④ 运动补偿融合（方案 B）"],
        ["classical.py", "传统基线：幂律、直方图均衡、CLAHE、多尺度 Retinex、引导滤波、引导滤波 Retinex + NLM 去噪"],
        ["v2_experiment.py / splits.py", "统一评测（val 选参冻结 → test 一次）、场景级划分"],
        ["tools/build_cache.py", "预计算对齐帧与卡尔曼结果（内存映射），使 16 GB 内存可训练 34 个场景"],
        ["make_demo_v2.py / enhance_video.py", "演示视频；任意 MP4 处理（保留音轨）"],
        ["tests/test_v2.py", "卡尔曼降噪与突变检测、网格初始化等于幂律、平滑器、划分不相交等 9 项测试"],
    ], [2.6, 3.4])

    # ———————————————————— 5 ————————————————————
    h("5. 结果对比：发生了什么变化")
    h("5.1 评价指标", 2)
    p("PSNR、SSIM：与配对正常光参考比较，输出先取整到 8 位；SSIM 用 11×11 高斯窗（σ=1.5），按通道平均。"
      "GT-mean PSNR/SSIM：把输出整体亮度按参考均值缩放后再计算，这是低光增强文献中常用的补充指标，用来区分“曝光猜得准不准”与“增强质量”。"
      "扭曲误差：用相邻两帧参考图估计光流，把上一帧输出对齐到当前帧，在可靠区域统计平均绝对差，衡量闪烁与抖动，越小越稳定。"
      "所有指标先在场景内平均，再对 12 个场景等权平均；指标在视频编码前计算。")

    h("5.2 官方 12 个测试场景的主结果", 2)
    main_rows = [("暗光输入", "暗光输入"), ("幂律变换", "幂律变换 s=1.5r^0.4"), ("直方图均衡", "直方图均衡"), ("CLAHE", "CLAHE"),
                 ("多尺度Retinex", "多尺度 Retinex"), ("引导滤波Retinex+NLM", "引导滤波 Retinex + NLM"),
                 ("Zero-DCE", "Zero-DCE [1]"), ("Zero-DCE+运动滤波(v1)", "第一版：Zero-DCE + 运动滤波"), (OURS, "KBG-Video（本文）")]
    rows = [[lab, f2(test[k]["psnr"]), f4(test[k]["ssim"]), f2(test[k]["psnr_gm"]), f4(test[k]["ssim_gm"]),
             f5(test[k]["warp_error"]) if k != "暗光输入" else "—", f"{test[k]['runtime_ms']:.0f}"] for k, lab in main_rows]
    table(["方法", "PSNR↑", "SSIM↑", "GT-mean PSNR↑", "GT-mean SSIM↑", "扭曲误差↓", "耗时/ms"], rows,
          [1.95, 0.6, 0.62, 0.78, 0.78, 0.72, 0.55], bold_row=len(rows))
    best_psnr = max((test[k]["psnr"], lab) for k, lab in main_rows)
    best_ssim = max((test[k]["ssim"], lab) for k, lab in main_rows)
    p(f"相对 Zero-DCE，KBG-Video 的 PSNR 变化 {o['psnr']-z['psnr']:+.2f} dB，SSIM 变化 {o['ssim']-z['ssim']:+.4f}，"
      f"GT-mean PSNR 变化 {o['psnr_gm']-z['psnr_gm']:+.2f} dB，扭曲误差变化 {(o['warp_error']/z['warp_error']-1)*100:+.0f}%；"
      f"相对第一版，PSNR 变化 {o['psnr']-v1['psnr']:+.2f} dB，SSIM 变化 {o['ssim']-v1['ssim']:+.4f}。"
      f"表中 PSNR 最高的是“{best_psnr[1]}”（{best_psnr[0]:.2f} dB），SSIM 最高的是“{best_ssim[1]}”（{best_ssim[0]:.4f}）。"
      "耗时为本机墙钟时间（含光流；传统方法在 CPU），仅供量级参考。")
    image("output/figures/fig_test_bars.png", "图6　测试集主要方法对比（12 个场景平均）。", 6.2)
    sb = jload("records/split_brightness.json"); tgt, vgt = sb["test_gt_mean"], sb["val_gt_mean"]
    p(f"需要如实说明：在官方测试集上，最终方法的 PSNR 低于 Zero-DCE，PSNR、SSIM 和 GT-mean PSNR 也都低于第一版（Zero-DCE + 运动滤波）；"
      f"它的优势集中在帧间稳定性（扭曲误差全表最低）以及相对 Zero-DCE 的结构/噪声（SSIM）。这与验证集结论（验证 PSNR {vf['psnr']:.2f} dB，"
      f"高于 Zero-DCE 的 {val['Zero-DCE']['psnr']:.2f} dB）相反。原因是分布差异：测试场景参考图的平均亮度约 {tgt:.2f}，验证场景约 {vgt:.2f}；"
      f"我们只用 34 个场景训练的 BGNet 学到的色调形状（图4）在更暗的测试场景上不再合适，"
      f"GT-mean PSNR 从 Zero-DCE 的 {z['psnr_gm']:.2f} dB 降到 {o['psnr_gm']:.2f} dB 即说明问题出在映射形状而非整体曝光。"
      f"不加去噪的 BGNet 在测试集上进一步跌到 {test['BGNet(无去噪)']['psnr']:.2f} dB / {test['BGNet(无去噪)']['ssim']:.3f}，"
      "而 Zero-DCE 在大量外部数据上训练，泛化更稳。测试集按规则只评测一次，我们没有据此回头修改方法。")
    dps, dss = ps(OURS, "psnr") - ps("Zero-DCE", "psnr"), ps(OURS, "ssim") - ps("Zero-DCE", "ssim")
    bp, bs = boot(dps), boot(dss)
    p(f"逐场景配对比较（图7）：KBG-Video 在 {int((dps > 0).sum())}/12 个场景 PSNR 高于 Zero-DCE、{int((dss > 0).sum())}/12 个场景 SSIM 更高。"
      f"场景级自助法（10000 次重采样）给出的平均差异 95% 置信区间：ΔPSNR {bp[0]:+.2f} dB [{bp[1]:+.2f}, {bp[2]:+.2f}]，"
      f"ΔSSIM {bs[0]:+.4f} [{bs[1]:+.4f}, {bs[2]:+.4f}]。12 个场景的样本量有限，区间宽度应一并看待。")
    image("output/figures/fig_per_scene.png", "图7　12 个测试场景逐个比较 PSNR（同一竖线连接同一场景的两种方法）。", 5.6)

    h("5.3 消融实验（测试集）", 2)
    abl = [("幂律变换+曝光回归", "传统幂律 + ③曝光回归（无网络、无去噪）"), ("BGNet(无去噪)", "②③：BGNet 直接作用于原始暗光帧"),
           ("EMA去噪+BGNet", "固定系数 EMA 代替卡尔曼 + ②③"), ("卡尔曼(无新息检验)+BGNet", "卡尔曼去掉新息检验 + ②③"),
           ("卡尔曼+BGNet", "①②③（无时域融合）"), (GRID, f"①②③ + 方案 A 网格平滑 α={cfg['alpha']:g}"),
           (OURS, f"①②③④ 最终 KBG-Video（方案 B，β={cfg['beta_pixel']:g}）")]
    rows = [[lab, f2(test[k]["psnr"]), f4(test[k]["ssim"]), f2(test[k]["psnr_gm"]), f5(test[k]["warp_error"]), f5(test[k]["tce"])] for k, lab in abl]
    table(["变体", "PSNR↑", "SSIM↑", "GT-mean PSNR↑", "扭曲误差↓", "TCE↓"], rows, [2.75, 0.6, 0.65, 0.8, 0.72, 0.62], bold_row=len(rows))
    T = test
    def cmp(a, b, k, unit="", fmt="{:+.3f}"): return fmt.format(T[a][k] - T[b][k]) + unit
    p(f"①卡尔曼去噪的作用：与“BGNet（无去噪）”相比，加入卡尔曼后 SSIM 变化 {cmp('卡尔曼+BGNet','BGNet(无去噪)','ssim',fmt='{:+.4f}')}，"
      f"PSNR 变化 {cmp('卡尔曼+BGNet','BGNet(无去噪)','psnr',' dB','{:+.2f}')}。训练中还观察到无去噪版本出现过验证指标骤降，说明先去噪也让学习更稳定。"
      f"卡尔曼与固定系数 EMA 相比，SSIM 差 {cmp('卡尔曼+BGNet','EMA去噪+BGNet','ssim',fmt='{:+.4f}')}，扭曲误差差 {cmp('卡尔曼+BGNet','EMA去噪+BGNet','warp_error',fmt='{:+.5f}')}；"
      f"去掉新息检验后 SSIM 差 {cmp('卡尔曼(无新息检验)+BGNet','卡尔曼+BGNet','ssim',fmt='{:+.4f}')}。")
    p(f"④运动补偿融合的作用：与无时域融合相比，扭曲误差变化 {cmp(OURS,'卡尔曼+BGNet','warp_error',fmt='{:+.5f}')}，"
      f"SSIM 变化 {cmp(OURS,'卡尔曼+BGNet','ssim',fmt='{:+.4f}')}，PSNR 变化 {cmp(OURS,'卡尔曼+BGNet','psnr',' dB','{:+.2f}')}；"
      f"方案 A 网格平滑的扭曲误差变化仅 {cmp(GRID,'卡尔曼+BGNet','warp_error',fmt='{:+.5f}')}，与验证集结论一致。"
      f"②网络相对“传统幂律 + 曝光回归”的增益为 PSNR {cmp('卡尔曼+BGNet','幂律变换+曝光回归','psnr',' dB','{:+.2f}')}、"
      f"SSIM {cmp('卡尔曼+BGNet','幂律变换+曝光回归','ssim',fmt='{:+.4f}')}，其中包含了①的去噪贡献。")

    h("5.4 可视化", 2)
    p("下图为测试集前两个场景的第 16 帧（按场景编号固定选取，不挑帧）。主演示视频：2.演示视频/KBG_讲解演示视频.mp4（约 4 分钟，中文配音加字幕，讲解问题、数据、方法与结果）。对比素材：2.演示视频/KBG_四栏对比_18场景.mp4"
      "（12 个测试场景 + 6 个验证场景，每场景 30 帧，10 fps 播放），以及展示中间结果的 KBG_算法过程_6场景.mp4。")
    for s in cfg["test_scenes"][:2]:
        image(f"output/{s}_KBG四栏对比.png", f"图　{s} 第16帧：暗光输入 / Zero-DCE / KBG-Video / 正常光参考。", 5.6)

    h("5.5 结论与局限", 2)
    p("结论：在不使用未来帧的前提下，“先在原始域按噪声模型去噪、再用低自由度的保边变换提亮、曝光单独估计、最后做运动补偿融合”的组合，"
      "在帧间稳定性和去噪上明显优于逐帧深度增强（Zero-DCE），但自训练映射网络的泛化限制了 PSNR。噪声空间相关性 κ 的标定和曝光解耦，是让经典理论在真实数据上生效的两处关键细节；"
      "“平滑变换参数即可消除闪烁”的设想没有得到数据支持，残余闪烁主要来自像素级噪声。")
    p("局限：⓪最重要的一点——自训练的映射网络泛化不足，测试集 PSNR 不如 Zero-DCE 与第一版；更稳妥的组合是把①卡尔曼去噪和④运动补偿融合接在外部预训练的增强网络两侧，这需要在新的未见场景上重新确认；①只评测了室内子集，每场景 30 帧；②SDSD 参考图曝光有不可预测的成分，原始 PSNR 的上限受此限制，因此同时报告了 GT-mean 指标；"
      "③训练场景仍偏少，网络必须依赖早停，在传统幂律附近做有限修正；④光流使用 CPU 上的 Farneback，快速大运动下对齐质量有限，"
      "可换成 GPU 光流网络（如 RAFT）；⑤测试场景在第一版中被查看过，虽然第二版参数从未在其上调整，独立性仍弱于全新数据。")
    h("参考文献", 2)
    for text in ["[1] Guo C, Li C, Guo J, et al. Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement. CVPR, 2020.",
                 "[2] Wang R, Xu X, Fu C W, et al. Seeing Dynamic Scene in the Dark: A High-Quality Video Dataset with Mechatronic Alignment. ICCV, 2021.",
                 "[3] Gharbi M, Chen J, Barron J T, et al. Deep Bilateral Learning for Real-Time Image Enhancement. ACM TOG (SIGGRAPH), 2017.",
                 "[4] He K, Sun J, Tang X. Guided Image Filtering. IEEE TPAMI, 2013.",
                 "[5] Foi A, et al. Practical Poissonian-Gaussian Noise Modeling and Fitting for Single-Image Raw-Data. IEEE TIP, 2008.",
                 "[6] 课程课件：《第3章 图像增强》《第3章-2 深度学习图像增强》。"]:
        p(text, indent=False)

    out = ROOT / "output/1.报告.docx"
    d.core_properties.title = "KBG-Video 低光照视频增强"; d.core_properties.author = ""; d.core_properties.subject = "计算机图像处理课程项目"
    d.save(out)
    (ROOT / "output/报告正文.md").write_text("\n".join(md), encoding="utf-8")
    (ROOT / "records/report_metrics_v2.json").write_text(json.dumps(dict(ours=o, zero_dce=z, v1=v1, config=cfg,
        paired_vs_zero_dce=dict(dpsnr_mean_ci=bp, dssim_mean_ci=bs, wins_psnr=int((dps > 0).sum()), wins_ssim=int((dss > 0).sum()))),
        indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
