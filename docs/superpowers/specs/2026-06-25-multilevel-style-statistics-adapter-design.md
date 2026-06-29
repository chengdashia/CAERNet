# Multi-Level Style Statistics Adapter Design

## Objective

Replace the current residual MLP as the proposed contribution with an
art-oriented CLIP visual adaptation mechanism. The new model must explicitly
represent distributed style evidence from CLIP patch tokens while preserving
the existing CLIP backbone, ArtBench-10 data pipeline, training loop, metrics,
and historical residual-adapter results.

The proposed model is named **Multi-Level Style Statistics Adapter (MLSSA)**.
The existing `clip_adapter` model remains unchanged and becomes a direct
baseline.

## Research Claim

The bounded methodological claim is:

> Artistic style recognition benefits from adapting both the global semantic
> representation and the distributional statistics of intermediate CLIP patch
> tokens. MLSSA extracts multi-level style statistics, separates them from the
> CLS content representation, and performs gated residual fusion before
> classification.

MixUp, label smoothing, supervised contrastive learning, and partial fine-tuning
remain training choices rather than claimed algorithmic contributions.

## Architecture

### CLIP Backbone

The backbone is OpenCLIP ViT-B/16. Most visual Transformer blocks remain frozen.
The last two visual blocks, final visual normalization, and visual projection
are trainable with a learning-rate scale of 0.01.

MLSSA collects token sequences from three visual depths:

- shallow-middle layer: block 4 output;
- middle layer: block 8 output;
- final layer: block 12 output.

Layer indices are configurable through `style_layers`. Negative indices are
accepted and resolved against the number of visual blocks. The default is
`[3, 7, 11]` using zero-based indexing.

The initial implementation supports the OpenCLIP ViT path exposed through
`visual.transformer.resblocks`. Unsupported visual backbones fail with a clear
error rather than silently falling back to global image features.

### Content Representation

The projected, normalized final CLS embedding is the content representation:

\[
h_c = \frac{g_\theta(x)}{\|g_\theta(x)\|_2},
\qquad h_c \in \mathbb{R}^{d}.
\]

For ViT-B/16, \(d=512\).

### Multi-Level Style Statistics

For selected layer \(\ell\), the CLS token is removed and the remaining patch
tokens are denoted by:

\[
P_\ell \in \mathbb{R}^{N\times C},
\]

where \(N\) is the number of patches and \(C\) is the Transformer width.

The layer descriptor contains channel-wise mean and standard deviation:

\[
\mu_\ell = \frac{1}{N}\sum_{i=1}^{N}P_{\ell,i},
\]

\[
\sigma_\ell =
\sqrt{\frac{1}{N}\sum_{i=1}^{N}
(P_{\ell,i}-\mu_\ell)^2+\epsilon}.
\]

Mean represents the global activation distribution associated with palette and
coarse appearance. Standard deviation represents spatial variation associated
with texture, contrast, and brushstroke heterogeneity.

The concatenated statistic is normalized and projected:

\[
s_\ell =
\phi_\ell\left(
\operatorname{LN}([\mu_\ell;\sigma_\ell])
\right),
\qquad s_\ell\in\mathbb{R}^{d}.
\]

Each selected layer has its own projection MLP:

```text
LayerNorm(2C)
Linear(2C, style_dim)
GELU
Dropout
Linear(style_dim, d)
```

The default `style_dim` is 256. This is a true bottleneck relative to the
concatenated \(2C=1536\) ViT-B/16 statistic.

### Learned Layer Fusion

Each layer receives a scalar importance score:

\[
q_\ell = w^\top \tanh(W_s s_\ell).
\]

The normalized weights are:

\[
\alpha_\ell =
\frac{\exp(q_\ell)}
{\sum_j\exp(q_j)}.
\]

The fused style feature is:

\[
h_s =
\operatorname{Norm}
\left(
\sum_\ell \alpha_\ell s_\ell
\right).
\]

The model returns the layer weights for diagnostics. Their mean value over the
test set can be reported to show which CLIP depths contribute to style
recognition.

### Gated Content-Style Fusion

A vector gate decides how much style correction should enter each feature
dimension:

\[
g = \sigma(W_g[h_c;h_s]+b_g),
\]

\[
z =
\operatorname{Norm}
\left(
h_c + g\odot h_s
\right).
\]

The final classifier predicts:

\[
o = W_c z+b_c.
\]

This fusion preserves the pretrained CLS representation through an explicit
skip path while allowing style statistics to modify dimensions selected by the
learned gate.

## Content-Style Decoupling

The model exposes `content_features` and `style_features` in its auxiliary
output. A squared cosine orthogonality penalty discourages the style branch from
duplicating the global content representation:

\[
\mathcal{L}_{orth} =
\frac{1}{B}\sum_{i=1}^{B}
\left(
\frac{h_{c,i}^{\top}h_{s,i}}
{\|h_{c,i}\|_2\|h_{s,i}\|_2}
\right)^2.
\]

The complete main objective is:

\[
\mathcal{L} =
\mathcal{L}_{ce}
+ \lambda_c\mathcal{L}_{supcon}
+ \lambda_o\mathcal{L}_{orth}.
\]

The default orthogonality weight is 0.05. Energy loss is disabled in the final
MLSSA experiments.

The training loop computes supervised contrastive loss on clean, non-MixUp
images. MixUp images are used only for mixed-label cross-entropy. This avoids
assigning a single hard contrastive class to an image containing two classes.

## Model Interface

Add architecture name:

```yaml
model:
  architecture: clip_mlssa
```

`ClipMLSSAClassifier.forward(images)` returns:

```python
{
    "logits": logits,
    "features": fused_features,
    "content_features": content_features,
    "style_features": style_features,
    "style_layer_weights": layer_weights,
}
```

The training and evaluation code must accept both this dictionary interface and
the existing tensor/tuple interfaces. Evaluation uses only `logits`.

Required model configuration:

```yaml
model:
  architecture: clip_mlssa
  clip_model_name: ViT-B-16
  clip_pretrained: /home/kmyh/classify/models/open_clip/vit_b16_openai.bin
  style_layers: [3, 7, 11]
  style_dim: 256
  fusion_hidden_dim: 256
  dropout: 0.1
  unfreeze_last_n_blocks: 2
```

Required loss configuration:

```yaml
loss:
  label_smoothing: 0.1
  contrastive_lambda: 0.1
  contrastive_temperature: 0.07
  orthogonality_lambda: 0.05
  energy_lambda: 0.0
  unknown_energy_lambda: 0.0
```

## Token Extraction

Token extraction must reproduce the OpenCLIP ViT visual forward path:

1. patch convolution;
2. flatten and transpose into patch tokens;
3. prepend the class embedding;
4. add positional embeddings and patch dropout;
5. apply `ln_pre`;
6. transpose to sequence-first form when required by OpenCLIP;
7. execute visual residual blocks sequentially;
8. capture selected block outputs;
9. apply final normalization and visual projection to the final CLS token.

Captured tokens remain connected to the computation graph for unfrozen blocks.
Frozen early blocks execute without parameter gradients but must not detach
their outputs, because later trainable blocks depend on them.

## Training Changes

### Clean Contrastive Pass

When MixUp and supervised contrastive learning are both enabled:

1. run the mixed images and compute mixed-label cross-entropy;
2. run the original clean images;
3. compute supervised contrastive loss from clean fused features and original
   labels;
4. compute orthogonality loss from clean content/style features;
5. combine the terms.

When MixUp is disabled, one clean forward pass supplies all losses.

The additional clean forward pass increases training cost but makes the
objective semantically correct and applies equally to MLSSA and the revised
residual-adapter comparison.

### Optimizer Groups

Full learning rate:

- style statistic projections;
- layer-attention scorer;
- fusion gate;
- classifier.

Scaled backbone learning rate:

- trainable CLIP visual blocks;
- final visual normalization;
- visual projection.

Frozen CLIP parameters are excluded from the optimizer.

## Controlled Experiments

### Main Adaptation Comparison

All methods use the same augmentation, optimizer, schedule, label smoothing,
MixUp, supervised contrastive implementation, validation selection, and three
seeds:

1. linear probe;
2. residual adapter only;
3. partial fine-tuning only;
4. residual adapter + partial fine-tuning;
5. full visual fine-tuning;
6. CLIP-Adapter baseline;
7. MLSSA style statistics only with frozen CLIP;
8. MLSSA + partial fine-tuning;
9. MLSSA + partial fine-tuning + orthogonality loss.

### MLSSA Component Ablation

Use the final training recipe and three seeds:

1. final CLS only;
2. final-layer mean statistics;
3. final-layer mean + standard deviation;
4. multi-layer mean + standard deviation with uniform averaging;
5. multi-layer statistics with learned layer fusion;
6. learned layer fusion + vector gate;
7. complete model with orthogonality loss.

### Layer Selection

Run seed 42 for:

- `[11]`;
- `[7, 11]`;
- `[3, 7, 11]`;
- `[1, 3, 5, 7, 9, 11]`.

The selected setting is then rerun with seeds 41 and 43.

### Generalization

At least one generalization experiment is required:

- preferred: WikiArt style classification using a documented train/validation/
  test split and the overlapping style vocabulary;
- fallback: ArtBench-10 stratified 1%, 5%, 10%, and 25% training subsets with
  linear probe, CLIP-Adapter, residual adapter, and MLSSA under identical
  subsets.

Low-data comparisons must use fixed subset manifests shared by all methods.

## Test Strategy

Unit tests must cover:

- channel mean and standard deviation against manually computed tensors;
- exclusion of the CLS token from style statistics;
- layer index resolution and invalid-index errors;
- learned layer weights summing to one;
- fused feature shape and L2 normalization;
- orthogonality loss for parallel and orthogonal features;
- dictionary-output compatibility in training and evaluation;
- clean-feature SupCon path when MixUp is enabled;
- optimizer grouping of MLSSA modules versus CLIP backbone;
- dry-run construction with a fake OpenCLIP ViT.

Existing residual-adapter, dataset, checkpoint, and final-test tests must remain
green.

## Output and Reproducibility

Each run stores:

- effective YAML configuration;
- validation history;
- best and last checkpoints;
- final held-out test metrics;
- mean style-layer weights for the final checkpoint;
- trainable and total parameter counts.

Test data is evaluated only once after selecting the best validation
checkpoint. New experiment configurations use `test_each_epoch: false` and
`test_after_training: true`.

## Paper Positioning

The title should refer to visual style statistics rather than generic
vision-language adapters. A working title is:

> Multi-Level Style Statistics Adaptation of CLIP for Artistic Style
> Classification

The three contribution claims become:

1. a multi-level patch-statistics representation designed for artistic style
   evidence;
2. gated content-style fusion with an explicit decoupling objective;
3. controlled evaluation against residual adapters, partial/full fine-tuning,
   CLIP adaptation baselines, and a generalization setting.

The text encoder remains absent from the final model and is described only as
part of CLIP pretraining and zero-shot baselines.

## Scope Boundaries

This implementation does not add:

- a separate CNN texture backbone;
- text prompt learning in the proposed model;
- generative style transfer;
- open-set or out-of-distribution objectives;
- larger CLIP backbones before ViT-B/16 ablations are complete.

These exclusions keep the new contribution focused and experimentally
identifiable.
