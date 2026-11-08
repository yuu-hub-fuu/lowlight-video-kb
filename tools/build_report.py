"""Create the five-part course report from measured results, using the supplied DOCX.

Run with the bundled workspace Python (python-docx, Pillow). The input reference
is retained unchanged. A PDF is produced by the packaged renderer, not here.
"""
from pathlib import Path
import csv
import hashlib
import json
from copy import deepcopy

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'template/1.报告.docx'
if not REFERENCE.exists():REFERENCE=ROOT.parent/'course_report_20261104/source/1.报告.docx'
EXPECTED='0c82a0e2301dfe9a16de16dd28df672b558c2e0edd42306fafba5c29025b29de'


def main():
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest()==EXPECTED
    results=json.loads((ROOT/'results/confirmation/summary.json').read_text())
    diagnostics=json.loads((ROOT/'results/test/summary.json').read_text())
    val=json.loads((ROOT/'results/validation_temporal/summary.json').read_text())
    first=json.loads((ROOT/'results/validation/summary.json').read_text())
    config=json.loads((ROOT/'records/final_config.json').read_text())
    metrics=results['methods'];base=metrics['ZeroDCE'];ours=metrics['Final']
    d=Document(REFERENCE)
    body=d._element.body
    for element in list(body):
        if element.tag!=qn('w:sectPr'):body.remove(element)
    # Keep the source section geometry and expand its five text slots.
    for name,size,font in [('Normal',11,'宋体'),('Heading 1',14,'黑体'),('Heading 2',11.5,'黑体'),('Title',20,'黑体')]:
        s=d.styles[name];s.font.name=font;s.font.size=Pt(size);s.font.color.rgb=RGBColor(0,0,0)
        s._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),font)
        s.paragraph_format.space_after=Pt(6)
        s.paragraph_format.line_spacing=1.25
        s.paragraph_format.widow_control=True
        if name.startswith('Heading'):
            s.paragraph_format.space_before=Pt(10);s.paragraph_format.keep_with_next=True
    for st in d.styles:
        if st.type==1 and ('Heading' in st.name or st.name in ('Title','Subtitle')):
            st.font.color.rgb=RGBColor(0,0,0)
    md=[]
    def p(text,bold=False):
        q=d.add_paragraph();q.paragraph_format.first_line_indent=Pt(22)
        if text.startswith(('[1]','[2]')):q.alignment=WD_ALIGN_PARAGRAPH.LEFT
        r=q.add_run(text);r.bold=bold;md.append(text+'\n');return q
    def h(text,level=1):d.add_heading(text,level);md.append('#'*(level+1)+' '+text+'\n')
    def table(headers,rows,widths):
        t=d.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
        for col,w in zip(t.columns,widths):col.width=Inches(w)
        for c,s in zip(t.rows[0].cells,headers):c.text=str(s)
        for row in rows:
            for c,s in zip(t.add_row().cells,row):c.text=str(s)
        for ir,row in enumerate(t.rows):
            pr=row._tr.get_or_add_trPr()
            if ir==0:pr.append(OxmlElement('w:tblHeader'))
            # Prevent splitting a short numeric row across pages.
            pr.append(OxmlElement('w:cantSplit'))
            for j,c in enumerate(row.cells):
                c.width=Inches(widths[j]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                tc=c._tc.get_or_add_tcPr();b=OxmlElement('w:tcBorders')
                for side in ['top','left','bottom','right']:
                    e=OxmlElement('w:'+side);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');b.append(e)
                tc.append(b)
                margins=OxmlElement('w:tcMar')
                for side in ['top','left','bottom','right']:
                    e=OxmlElement('w:'+side);e.set(qn('w:w'),'80');e.set(qn('w:type'),'dxa');margins.append(e)
                tc.append(margins)
                if ir==0:
                    sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'EDEDED');tc.append(sh)
                for para in c.paragraphs:
                    para.paragraph_format.space_after=Pt(2);para.paragraph_format.line_spacing=1.1
                    if j>0 and len(str(c.text))<20:para.alignment=WD_ALIGN_PARAGRAPH.CENTER
                    for r in para.runs:r.font.size=Pt(9.5);r.bold=ir==0
        d.add_paragraph().paragraph_format.space_after=Pt(0)
        md.append('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |')
        md.extend('| '+' | '.join(map(str,row))+' |' for row in rows);md.append('')
        return t
    def image(name,caption):
        path=ROOT/'output'/name
        q=d.add_paragraph();q.alignment=WD_ALIGN_PARAGRAPH.CENTER
        q.paragraph_format.keep_with_next=True;q.add_run().add_picture(str(path),width=Inches(5.65))
        q=d.add_paragraph(caption);q.alignment=WD_ALIGN_PARAGRAPH.CENTER
        for r in q.runs:r.font.size=Pt(9)
        md.append(f'![{caption}]({name})\n')

    title=d.add_paragraph('低光照视频增强与运动感知时域处理',style='Title');title.alignment=WD_ALIGN_PARAGRAPH.CENTER
    q=d.add_paragraph('计算机图像处理课程报告');q.alignment=WD_ALIGN_PARAGRAPH.CENTER
    q=d.add_paragraph('学号：________    姓名：________    日期：2026年11月4日');q.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p('摘要：低光照视频增强不仅需要提高单帧亮度，还需要兼顾暗部噪声与帧间稳定性。本项目基于公开的 Zero-DCE 预训练网络，使用 SDSD 真实暗光／正常光配对连续帧，完成从单张图像推理到视频处理的扩展。我们比较曝光校准、增强曲线平滑、空间滤波、无对齐帧平均以及运动感知时域滤波。初始外推检查发现固定曝光校准难以跨场景泛化，因此保留原网络亮度映射，仅加入运动感知时域滤波，再在6个此前未使用的场景独立确认。网络结构和权重保持原样，贡献集中在视频处理代码、对照实验和失败分析。')
    p('关键词：低光照视频；图像增强；Zero-DCE；运动补偿；时域滤波')

    h('1. GitHub 链接')
    p('基础方法与网络代码：https://github.com/Li-Chongyi/Zero-DCE。采用固定版本 e0f4adc54d0f，原始网络位于 vendor/zero_dce/model.py，预训练参数为 Epoch99.pth。完整版本与文件校验记录见附件。[1]')
    p('数据及作者评测代码：https://github.com/JIA-Lab-research/SDSD。新增代码随附件提供，运行方式见README，上游来源与使用说明见 THIRD_PARTY.md。')

    h('2. 任务描述：要解决什么问题')
    p('本项目输入一段低光照视频，输出逐帧增强后的视频。处理对象是像素亮度、颜色、噪声与时序变化，对应课程的灰度变换、图像增强、滤波及深度学习增强内容。')
    p('逐帧增强能够提高可见度，但输入噪声也可能被放大。相邻帧包含重复的场景信息，因此可以利用时间冗余抑制噪声；然而运动和遮挡会使直接平均产生拖影。我们要检验：先估计帧间运动并对齐，再根据对齐可靠性控制历史信息占比，能否改善单帧增强的质量。')
    p('课程关联：《第3章 图像增强》涉及平滑与保边的权衡；《第3章-2 深度学习图像增强》第31–32页介绍低光照增强，第44页讨论去噪后结构细节保留。本项目将同样的问题扩展到连续视频。')

    h('3. 数据描述：数据是什么')
    p('SDSD 是 ICCV 2021 工作发布的真实低光照视频数据集，通过可重复的机械运动采集并对齐暗光与正常光视频，包含70对室内视频和80对室外视频。本实验使用作者发布的 indoor_np；数据以按时间排序的 .npy 图像序列存储，不是独立照片集合。[2]')
    table(['用途','场景','配对帧数'],[
        ['验证','pair1–pair6，来自作者训练场景池','6×30＝180'],
        ['初始检查','pair13、15、21、23、31、33、50、52、58、60、68、70','12×30＝360'],
        ['最终确认','pair7–pair12，修正方案冻结后才下载与评价','6×30＝180'],
        ['合计','24个独立场景，均取各自前30帧','720组，共1440个数组']],[.65,3.95,1.15])
    p('原始数组为 960×512、uint8、BGR。读取时转为 RGB 并除以255，得到 [0,1] 输入。依据作者的数据读取逻辑，暗光和参考序列分别排序后按位置配对；二者文件名可能不同，按同名匹配会造成错误。下载器从作者的分卷压缩文件读取指定成员，验证 CRC32，并记录原始文件的 SHA-256。')
    p('本实验没有重新训练神经网络。6个验证场景用于选择初始参数，12个初始检查场景来自作者的室内测试划分。由于其中暴露的失败影响了方案修正，这12个场景随后作为诊断集，不再称为修正方案的未见测试集。最终方案固定后，才获取pair7–pair12作独立确认。没有把同一视频的相邻帧随机拆入不同集合。每场景仅评价前30帧，不代表完整SDSD评测。')

    h('4. 代码修改描述：我们改了什么')
    h('4.1 从单图推理扩展到连续视频',2)
    p('上游低光照测试脚本按图片读取，并在每次处理时创建网络和加载权重。我们将模型加载移到序列循环之外，新增连续帧读取、颜色与数值范围检查、视频编码和音轨处理。enhance_video.py 支持直接输入 MP4，保持输入帧率；SDSD示例视频按选定播放速度合成，不能把该播放速度当作原始采集帧率。')
    h('4.2 验证集曝光校准',2)
    p(f'初始方案曾对Zero-DCE输出增加全局幂律变换，候选指数为0.6、0.8、1.0和1.2，按验证场景平均PSNR选择0.6。但该固定校准在随后12个场景上造成过度提亮，最终关闭此模块，即指数固定为1。另设直接伽马校正基线，指数在验证集0.2–0.6候选中选择为 {config["gamma_baseline"]:g}。曝光校准的失败结果予以保留。')
    h('4.3 运动感知时域滤波',2)
    p('先在四分之一宽高的暗光输入上估计当前帧到上一帧的反向光流，同时计算正向光流。把光流放大到原分辨率后，将历史增强图像变换到当前帧坐标。置信度由光度残差、正反向光流一致性和有效边界共同决定；可靠区域融合更多历史信息，不可靠或遮挡区域优先保留当前帧。')
    p(f'融合形式为“当前结果×(1−权重)＋对齐历史结果×权重”，权重等于置信度乘时域系数。该系数只在验证集的0.3、0.6、0.85中选择，本次固定为 {config["beta"]:g}。检测到镜头切换或开始新视频时重置状态。镜头切换检测是保守的工程规则，极暗场景仍可能漏检。')
    h('4.4 消融与评测实现',2)
    p('为区分提亮、空间去噪和时间融合的作用，最终确认集设置原图、直接伽马校正、原始Zero-DCE、空间双边滤波、无对齐时域平均、最终运动感知滤波，以及被关闭的曝光校准变体。空间和无对齐对照均保持原始Zero-DCE亮度映射。另保留增强曲线平滑的验证记录，以检验最初假设。')
    table(['文件','新增职责'],[
        ['lowlight.py','模型复用、曲线处理、光流与置信度、时域状态、指标'],
        ['experiment.py','曝光及曲线平滑的验证实验'],
        ['temporal_experiment.py','图像时域滤波对照及冻结后的测试'],
        ['enhance_video.py / make_demo.py','实际视频处理、四栏对比视频生成'],
        ['tools/download_sdsd.py','按场景获取真实帧，校验CRC32与记录来源'],
        ['tests/test_core.py','状态重置、数值范围和独立SSIM核对']],[2.2,3.55])

    h('5. 结果对比：发生了什么变化')
    h('5.1 评价设置',2)
    p('PSNR和SSIM基于与输入配对的正常光参考图像，数值越高越好。输出先四舍五入至8位，再计算RGB PSNR；SSIM使用11×11高斯窗、标准差1.5并按通道平均。先计算场景内平均，再对场景等权平均。指标在视频编码前计算，展示视频的压缩损耗不计入。')
    p('辅助指标TCE定义为相邻两帧重建误差变化的平均绝对值：先求每帧“输出−参考”的误差，再求相邻误差之差。TCE越低，表示重建误差在时间上变化越小，但不能独立证明画面更清晰或没有拖影，必须与PSNR、SSIM及视频一起看。')
    h('5.2 最终6个独立确认场景的定量结果',2)
    labels={'Input':'暗光输入','Gamma':'直接伽马校正','ZeroDCE':'原始Zero-DCE','Spatial':'Zero-DCE＋空间滤波','NaiveRGB':'Zero-DCE＋无对齐时域平均','Final':'最终：运动感知时域滤波','RejectedExposure':'附加固定曝光校准（未采用）'}
    rows=[]
    for key,label in labels.items():
        v=metrics[key];rows.append([label,f'{v["psnr"]:.3f}',f'{v["ssim"]:.4f}',f'{v["tce"]:.5f}'])
    table(['方法','PSNR↑ / dB','SSIM↑','TCE↓'],rows,[2.65,1.05,1.,1.05])
    delta_p=ours['psnr']-base['psnr'];delta_s=ours['ssim']-base['ssim'];delta_t=ours['tce']-base['tce']
    p(f'相对原始Zero-DCE，最终方案的场景平均PSNR变化为 {delta_p:+.3f} dB，SSIM变化为 {delta_s:+.4f}，TCE变化为 {delta_t:+.5f}。这些数字来自本次实际运行，详细的逐帧与逐场景结果保存在 results/confirmation/per_frame.csv 和 per_scene.csv。')
    spatial=metrics['Spatial'];naive=metrics['NaiveRGB']
    p(f'最终方案与空间滤波相比，PSNR变化为 {ours["psnr"]-spatial["psnr"]:+.3f} dB；与相同系数的无对齐时域平均相比变化为 {ours["psnr"]-naive["psnr"]:+.3f} dB。应同时检查SSIM、TCE与视频边缘，不能仅以较低的时序变化断言不存在拖影。')
    p(f'需要说明，直接伽马校正的PSNR为 {metrics["Gamma"]["psnr"]:.3f} dB，高于最终方案，但其SSIM为 {metrics["Gamma"]["ssim"]:.4f}，低于最终方案。附加曝光校准在这6个确认场景也能提高PSNR，却曾在初始12场景明显失败，说明其效果依赖场景亮度分布。最终方案的优势主要是相对原始Zero-DCE的结构保留与时序误差改善，PSNR仅小幅增加，不能宣称所有指标最优。')
    h('5.3 未获支持的初始假设',2)
    p(f'在6个验证场景中，原始Zero-DCE的TCE为 {first["methods"]["ZeroDCE"]["tce"]:.6f}；曲线平滑强度0.3、0.6、0.85对应的TCE分别为 '+ '、'.join(f'{first["methods"][f"Temporal_{b:g}"]["tce"]:.6f}' for b in [.3,.6,.85])+'，均未降低该指标。因此没有把曲线平滑作为最终有效模块。随后根据验证图中增强噪声的现象，检验显式运动补偿后的图像时域滤波；整个调整过程在最终测试之前完成。')
    diag=diagnostics['methods']
    p(f'曝光校准的外推失败同样重要：在初始12个检查场景中，原始Zero-DCE为 {diag["ZeroDCE"]["psnr"]:.3f} dB；曝光校准加时域处理降至 {diag["Improved"]["psnr"]:.3f} dB；不额外校准曝光、仅做运动时域处理为 {diag["FlowOnly"]["psnr"]:.3f} dB。由此关闭固定曝光模块。该决定使用了初始检查结果，因此最终效果必须由后来才使用的6个确认场景检验，不能直接把初始检查中最好的变体当成最终独立成绩。')
    h('5.4 可视化与适用边界',2)
    p('下面展示按确认集顺序确定的前两个场景中间帧。完整对比视频包含6个最终确认场景及12个初始诊断场景，每场景30帧，统一以10 fps播放，共54秒。四栏包含同一时刻的输入、基线、最终处理与参考，便于观察颜色、暗部噪声及物体边缘；只有最终确认集用于上述独立成绩。')
    image('pair7_四栏对比.png','图1　独立确认场景pair7第16帧：暗光输入／Zero-DCE／最终处理／正常光参考。')
    image('pair8_四栏对比.png','图2　独立确认场景pair8第16帧，布局同图1。帧间现象应结合附带视频观察。')
    p('本实验的改进属于已有网络之上的视频处理扩展，不能表述为原创Zero-DCE或新的光流算法。样本仅来自室内子集、每场景30帧，且采用一个公开预训练模型；没有验证室外强运动、长时视频和所有镜头切换情况。时域融合还可能损失快速运动细节，后续可增加更长序列和遮挡较多场景的专门评测。')
    p('实现检查：核心单元测试通过，SSIM与独立库实现进行数值核对；数据下载校验CRC32，原始权重记录SHA-256。GPU推理与指标计算已完成，性能日志属于本次环境记录，不作为不同算法的严格速度排名。')
    h('5.5 参考文献与材料',2)
    for text in [
        '[1] Guo C, Li C, Guo J, et al. Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement. CVPR, 2020: 1780–1789. https://arxiv.org/abs/2001.06826',
        '[2] Wang R, Xu X, Fu C W, et al. Seeing Dynamic Scene in the Dark: A High-Quality Video Dataset with Mechatronic Alignment. ICCV, 2021. https://github.com/JIA-Lab-research/SDSD',
        '[3] 课程课件：《第3章 图像增强》《第3章-2 深度学习图像增强》。',
        '附件：完整项目代码、原始预训练权重、公开数据下载脚本、少量原始帧、验证与测试CSV/JSON、输入／增强／四栏对比视频。'
    ]:p(text)
    out=ROOT/'output/1.报告.docx';out.parent.mkdir(exist_ok=True)
    d.core_properties.title='低光照视频增强与运动感知时域处理'
    d.core_properties.author='';d.core_properties.subject='计算机图像处理课程项目'
    d.save(out)
    (ROOT/'output/报告正文.md').write_text('\n'.join(md),encoding='utf-8')
    (ROOT/'records/report_metrics.json').write_text(json.dumps(dict(baseline=base,improved=ours,
        delta=dict(psnr=delta_p,ssim=delta_s,tce=delta_t),validation_config=config),indent=2),encoding='utf-8')
    print(out)


if __name__=='__main__':main()
