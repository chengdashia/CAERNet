# CAERNet 代码说明

`classify/CAERNet/` 是当前唯一正式代码目录。项目已经从多份显卡代码副本整理为一份源码 + 多硬件配置。当前论文主线是 CLIP/VLM 视觉-语言分支，MS-CAERNet 保留为视觉-only 对照和消融分支。

## 已实现方法

- CLIP zero-shot：基于艺术风格 prompt 的文本原型分类。
- CLIP linear probe：冻结 CLIP image encoder，只训练线性分类头。
- CLIP adapter：冻结 CLIP image encoder，训练轻量 adapter + 分类头。
- MS-CAERNet：ResNet50 多尺度坐标注意力 + energy / SupCon 视觉分支。
- CNN baselines：ResNet50、EfficientNetV2-S、ConvNeXt-Tiny、RegNetY 等。

关键源码：

```text
src/models/clip_art.py         CLIP/VLM 分类模型
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

CLIP/VLM 实验需要 `open_clip_torch`。第一次运行 CLIP 或 torchvision 预训练模型时会下载权重。

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
