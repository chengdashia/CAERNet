# CAERNet 代码说明

`classify/CAERNet/` 是当前唯一正式代码目录。项目已经从多份显卡代码副本整理为一份源码 + 多硬件配置。当前论文主线是 CLIP/VLM 视觉-语言分支，MS-CAERNet 保留为视觉-only 对照和消融分支。

## 已实现方法

- CLIP zero-shot：基于艺术风格 prompt 的文本原型分类。
- CLIP linear probe：冻结 CLIP image encoder，只训练线性分类头。
- CLIP adapter：冻结 CLIP image encoder，训练轻量 adapter + 分类头。
- CLIP MLSSA：从多层 ViT patch tokens 提取通道均值和标准差，通过可学习层融合与门控残差注入 CLS 内容特征。
- MS-CAERNet：ResNet50 多尺度坐标注意力 + CASG（Context-Aware Scale Gate）跨尺度融合 + energy barrier / SupCon 视觉分支。CASG 通过上下文感知门控机制动态加权 ResNet 各阶段的多尺度特征，避免简单均值融合导致的信息退化。
- CNN baselines：ResNet50、EfficientNetV2-S、ConvNeXt-Tiny、RegNetY 等。

关键源码：

```text
src/models/clip_art.py         CLIP/VLM 分类模型
src/models/clip_mlssa.py       多层风格统计适配模型
src/models/style_statistics.py patch token 风格统计与层融合
src/models/ms_caernet.py       MS-CAERNet 视觉分支
src/models/registry.py         模型注册入口
src/prompts.py                 CLIP prompt YAML 解析
src/losses.py                  CE、energy、barrier、SupCon loss
src/train.py                   通用训练逻辑
src/eval.py                    通用评估逻辑
```

## 环境

```bash
cd /Users/dong/Documents/SCI
pip install -r classify/requirements.txt
```

CLIP/VLM 实验需要 `open_clip_torch`。CLIP 模型通过 `clip_pretrained` 字段指定预训练来源（如 `openai`），首次运行时由 open_clip 自动下载。ViT-B-16 和 ViT-L-14 均支持。

## 数据

ArtBench-10 使用 ImageFolder 格式：

```text
classify/data/artbench10/
  train/
  test/
```

论文训练使用：

```text
classify/data/artbench10_paper/
  val/
  test/
```

首次训练前运行：

```bash
cd /home/kmyh/classify/CAERNet
python prepare_data.py
```

当前划分策略：

- `train`：从原始训练集移出验证集后约 45,000 张。
- `val`：从原始训练集抽取，每类 500 张。
- `test`：保留 ArtBench-10 原始测试集 10,000 张。

`run_training.py` 只检查数据是否存在，不再自动拆分数据。

## 训练入口

统一训练入口：

```bash
cd /Users/dong/Documents/SCI
PYTHONPATH=classify/CAERNet python classify/CAERNet/run_training.py \
  --config classify/CAERNet/configs/experiments/clip_adapter_vit_b16.yaml \
  --hardware a100
```

快速检查：

```bash
PYTHONPATH=classify/CAERNet python classify/CAERNet/run_training.py \
  --config classify/CAERNet/configs/experiments/clip_linear_vit_b16.yaml \
  --hardware a100 \
  --dry-run
```

硬件 profile：

```text
configs/hardware/a100.yaml
configs/hardware/4090d.yaml
configs/hardware/3060ti.yaml
```

第一轮实验顺序见：

```text
../docs/experiment_matrix.md
```

## MLSSA 半重做实验

MLSSA 将当前 residual adapter 保留为 baseline。新方法的主要机制是：

- patch-token 均值表示整体色彩和激活分布；
- patch-token 标准差表示纹理、对比度与空间变化；
- block 4、8、12 的统计特征通过可学习权重融合；
- 向量门控制风格特征对 CLS 内容特征的残差修正；
- 正交损失抑制内容特征与风格特征重复编码。

A100 上的权重路径保持为：

```text
/home/kmyh/classify/models/open_clip/vit_b16_openai.bin
```

先检查完整模型：

```bash
cd /home/kmyh/classify/CAERNet
PYTHONPATH=. python run_training.py \
  --config configs/mlssa/clip_mlssa_full.yaml \
  --hardware a100 \
  --dry-run
```

推荐运行顺序：

```bash
# 1. 适配方法主对比
PYTHONPATH=. python run_mlssa_ablation_a100.py \
  --groups comparison --seeds 41 42 43

# 2. MLSSA 组件消融
PYTHONPATH=. python run_mlssa_ablation_a100.py \
  --groups components --seeds 41 42 43

# 3. 层选择先用 seed 42 筛选
PYTHONPATH=. python run_mlssa_ablation_a100.py \
  --groups layers --seeds 42
```

新配置只按验证集选择 checkpoint，训练结束后才对官方测试集评估一次。

## 审稿补充实验（A100）

补充实验统一使用验证集选 checkpoint，并在训练结束后仅对最佳 checkpoint
评估一次测试集。测试指标写入每个 run 目录下的
`test_metrics.json`，不会写入逐 epoch 的 `history.csv`。

先检查全部配置能否构建：

```bash
cd /home/kmyh
PYTHONPATH=classify/CAERNet python classify/CAERNet/run_revision_ablation_a100.py \
  --groups all \
  --seeds 41 \
  --dry-run
```

按实验组运行：

```bash
# adapter-only / partial-only / adapter+partial / full visual fine-tuning
PYTHONPATH=classify/CAERNet python classify/CAERNet/run_revision_ablation_a100.py \
  --groups adaptation

# 解冻深度：0 / 1 / 2 / 4 / full visual encoder
PYTHONPATH=classify/CAERNet python classify/CAERNet/run_revision_ablation_a100.py \
  --groups depth

# CE / label smoothing / MixUp / SupCon / full recipe
PYTHONPATH=classify/CAERNet python classify/CAERNet/run_revision_ablation_a100.py \
  --groups regularizers

# 标准 CLIP-Adapter feature-blending baseline
PYTHONPATH=classify/CAERNet python classify/CAERNet/run_revision_ablation_a100.py \
  --groups clip_adapter

# 1% / 5% / 10% / 25% 分层低数据量实验
PYTHONPATH=classify/CAERNet python classify/CAERNet/run_revision_ablation_a100.py \
  --groups low_data
```

脚本默认使用 seeds `41 42 43`，也可显式指定：

```bash
PYTHONPATH=classify/CAERNet python classify/CAERNet/run_revision_ablation_a100.py \
  --groups adaptation \
  --seeds 41 42 43
```

重跑主表中的可训练 baseline：

```bash
PYTHONPATH=classify/CAERNet python classify/CAERNet/run_revision_baselines_a100.py
```

该脚本包含五个 CNN/attention baseline、统一协议的 CLIP linear probe 和
标准 CLIP-Adapter，并对每个方法运行 seeds `41 42 43`。

CoOp、CoCoOp、MaPLe、PromptSRC 和 Tip-Adapter 应使用作者官方仓库，避免
在本项目中维护不完整复刻。以下命令默认只打印计划和缺失仓库：

```bash
python classify/CAERNet/run_official_clip_baselines_a100.py
```

将官方仓库放到 `/home/kmyh/classify/third_party/` 并完成各仓库的
ArtBench-10 dataset config 后，增加 `--execute` 执行。官方仓库与对应地址
会在缺失时由脚本打印。
