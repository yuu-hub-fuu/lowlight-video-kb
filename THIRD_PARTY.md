# 第三方来源与本项目修改

本项目是图像处理课程的复现与工程改进项目，不声称原创 Zero-DCE、光流或时域滤波方法。

## Zero-DCE

- 作者仓库：https://github.com/Li-Chongyi/Zero-DCE
- 固定版本：e0f4adc54d0f23348c4a9b84acc08fe8778d5bfd
- 原始网络：`vendor/zero_dce/model.py`，保持上游原样。
- 原始权重：`weights/Epoch99.pth`，保持上游原样，没有重新训练。
- 原始说明及许可：`vendor/zero_dce/README.upstream.md`。
- 上游注明 Attribution-NonCommercial 4.0 International，供非商业学术研究使用。
- 文献：Guo et al. Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement. CVPR 2020.

## SDSD

- 作者项目：https://github.com/JIA-Lab-research/SDSD
- 本次核对版本：07330bb0ac124d2d56f6e92908d650dde3f7f7c2
- 论文：Wang et al. Seeing Dynamic Scene in the Dark: A High-Quality Video Dataset with Mechatronic Alignment. ICCV 2021.
- 数据来自作者公开 Google Drive 分卷压缩文件，非自行采集。
- 采用 indoor_np，原始 uint8 BGR 数组，960×512。
- 下载器按 HTTP Range 获取指定 ZIP 成员，校验 CRC32，并记录每个原始 .npy 文件的 SHA-256。
- 数据权利归原作者。附带小样本仅用于本地课程复现实验；进一步分发应核对作者的数据条款。

## 本项目新增内容

`lowlight.py`、`experiment.py`、`temporal_experiment.py`、`enhance_video.py`、`make_demo.py`、
`tools/download_sdsd.py`、测试、结果记录、课程报告及运行文档为本次项目实现。
主要新增：连续帧读取、独立场景验证、验证集曝光校准、运动对齐与置信度门控、
曲线/图像时域处理对照、镜头切换重置、视频输出和评测记录。

## 第二版新增（KBG-Video）

- 双边网格思路参考：Gharbi M, Chen J, Barron J T, et al. Deep Bilateral Learning for Real-Time Image Enhancement. ACM TOG (SIGGRAPH), 2017。
  本项目 `bgnet.py` 为按论文思想独立编写的轻量实现，未复制其代码。
- 引导滤波：He K, Sun J, Tang X. Guided Image Filtering. ECCV 2010 / TPAMI 2013。`classical.py` 中为独立实现。
- 卡尔曼滤波、泊松-高斯噪声模型、新息检验为经典理论，`kalman.py` 为本项目实现；噪声参数由本项目在 SDSD 训练场景上标定。
- 第二版新增训练数据：SDSD 训练池 pair14,16–20,22,24（各40帧）与 pair25–30,32,34–46（各12帧），来源同上，校验方式同上。
- 本项目训练的权重 `weights/bgnet_*.pth`、`weights/trnet_*.pth` 仅用于课程实验。
