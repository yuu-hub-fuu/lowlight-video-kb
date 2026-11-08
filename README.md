# KBG-Video：噪声建模卡尔曼去噪 + 可学习双边网格的低光照视频增强

> 测试集结论（如实）：相对 Zero-DCE，SSIM 0.694→0.772、扭曲误差降低 84%，但 PSNR 低 0.56 dB，且不如第一版；详见报告 5.2 节。网格系数平滑（方案 A）在验证集上几乎无效，保留为消融，见 `bgnet.GridSmoother`。

图像处理课程项目（第二版）。在作者公开的 SDSD 真实暗光/正常光**配对连续帧**上，实现并评测一条
“先去噪、再提亮、平滑变换而非像素”的四级视频增强流水线，并与传统方法、Zero-DCE、第一版方案和直接监督 U-Net 对比。

```
暗光帧 ─┬─ 光流(Farneback, 1/4 分辨率, 前后向一致性置信度)
        ├─ ① 运动补偿逐像素卡尔曼滤波   泊松-高斯噪声模型 σ²=a·x+b，新息检验自动处理遮挡     kalman.py
        ├─ ② 可学习双边网格 BGNet       16×16×8 网格，每格 3×4 颜色仿射，按引导图三线性切片   bgnet.py
        ├─ ③ 曝光回归                  log μ* = c0 + c1·log μ_in（只用训练场景拟合）          tools/fit_exposure.py
        └─ ④ 运动补偿置信度融合        out = (1−βc)·y + βc·W(out_prev)，β=0.85（验证集选定）  kbg.KBGVideo.final
```

## 课程知识点对应

| 模块 | 课程内容 |
| --- | --- |
| 噪声标定、卡尔曼滤波、新息检验 | 图像噪声模型、图像复原、维纳/卡尔曼最优估计 |
| 光流对齐、置信度 | 运动估计、几何变换与插值 |
| 双边网格、引导图切片 | 灰度变换/查找表、双边滤波与保边平滑 |
| 幂律、HE、CLAHE、MSR、引导滤波 Retinex + NLM | 灰度变换、直方图处理、Retinex、空间域滤波（作为对照基线） |
| BGNet / TRNet 训练、曝光解耦 | 深度学习图像增强、监督学习与泛化 |

## 文件

| 文件 | 作用 |
| --- | --- |
| `kalman.py` | 第①级：卡尔曼去噪、固定系数 EMA 对照 |
| `bgnet.py` | 第②③级：双边网格网络、网格时域平滑 |
| `kbg.py` | 完整在线流水线 `KBGVideo` |
| `classical.py` | 传统基线：幂律、HE、CLAHE、MSR、引导滤波、Retinex+NLM |
| `vnet.py` / `pipeline.py` | 对照组 TRNet（两帧输入 U-Net，直接监督） |
| `splits.py` | 场景级数据划分（训练 34 / 验证 6 / 官方测试 12） |
| `train_bgnet.py` / `train_trnet.py` | 训练脚本（验证集早停） |
| `v2_experiment.py` | 统一评测：val 阶段选参并冻结，test 阶段只跑一次（`--part i/k` 多进程并行，`--merge k` 合并） |
| `tools/calibrate_noise.py` | 噪声模型 a、b 与空间相关系数 κ 标定 |
| `tools/tune_kalman.py` | 卡尔曼参数选择（只用训练场景） |
| `tools/fit_exposure.py` | 曝光回归（留一场景交叉验证选模型） |
| `tools/build_cache.py` | 预计算对齐帧/置信度/卡尔曼结果（内存映射，16 GB 内存可训练） |
| `tools/make_figures.py` / `tools/build_report_v2.py` | 由实验记录生成插图与报告 |
| `make_presentation.py` | 讲解版演示视频（分镜动画 + 中文配音 + 字幕，数字从实验记录读取并核对） |
| `make_demo_v2.py` / `enhance_video.py` | 演示视频；处理任意 MP4 |
| 第一版文件 `lowlight.py` `experiment.py` `temporal_experiment.py` `make_demo.py` | 保留，作为对照与历史记录 |

## 数据划分与评测约定

- 数据：SDSD `indoor_np`，960×512，uint8 BGR → RGB [0,1]；暗光与参考序列分别排序后按位置配对（与作者代码一致）。
- 训练 34 个场景（均来自作者训练池）：pair1–6（30 帧）、pair14,16–20,22,24（40 帧）、pair25–30,32,34–46（12 帧）。
- 验证 pair7–12（30 帧）：早停、α/β 选择。
- 测试：作者官方室内测试场景 pair13,15,21,23,31,33,50,52,58,60,68,70（各前 30 帧），`records/v2_config.json` 冻结后只评测一次。
  说明：这 12 个场景在第一版中曾作为诊断集看过 Zero-DCE 系列结果；第二版的所有参数都没有在它们上面调过。
- 指标：8 位取整后 RGB PSNR、高斯窗 SSIM；**GT-mean PSNR/SSIM**（输出整体亮度按参考均值对齐后再评，分离曝光误差与增强质量）；
  **扭曲误差**（用参考帧光流对齐相邻输出帧后的平均绝对差，衡量闪烁/抖动）；TCE；场景等权平均。

## 运行

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
# 处理自己的视频（默认 KBG-Video；--mode v1 / baseline 为对照）
.\.venv\Scripts\python.exe enhance_video.py --input data/sample_video.mp4 --output output/sample_enhanced.mp4
```

完整复现（数据约 3 GB，下载器带 CRC32 校验）：

```powershell
python tools/download_sdsd.py                                    # 第一版 24 场景
python tools/download_sdsd.py --scenes pair14,pair16,pair17,pair18,pair19,pair20,pair22,pair24 --frames 40
python tools/download_sdsd.py --scenes pair25,pair26,pair27,pair28,pair29,pair30,pair32,pair34,pair35,pair36,pair37,pair38,pair39,pair40,pair41,pair42,pair43,pair44,pair45,pair46 --frames 12
python tools/calibrate_noise.py ; python tools/tune_kalman.py ; python tools/build_cache.py ; python tools/fit_exposure.py
python train_bgnet.py --input kf ; python train_bgnet.py --input kf --loss direct ; python train_bgnet.py --input lq
python train_trnet.py --temporal 1 --iters 3000 ; python train_trnet.py --temporal 0 --iters 3000
python v2_experiment.py --stage val ; python v2_experiment.py --stage test
python make_demo_v2.py ; python make_presentation.py ; python tools/make_figures.py ; python tools/build_report_v2.py
```

本机环境：RTX 4060 Laptop（CUDA，PyTorch 2.6），光流与 NLM 在 CPU（pip 版 OpenCV 不带 CUDA）。
原始方法、数据、权重来源见 `THIRD_PARTY.md`。
