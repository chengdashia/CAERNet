# MLSSA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a multi-level CLIP patch-statistics adapter with gated content-style fusion and orthogonality regularization, while retaining the current residual adapter as a baseline.

**Architecture:** A new `clip_mlssa` model requests intermediate patch tokens from OpenCLIP ViT-B/16, computes per-layer channel mean and standard deviation, projects and fuses them into a style feature, then gates that feature into the normalized final CLS embedding. Training accepts structured model output, computes orthogonality loss, and uses clean images for supervised contrastive learning when MixUp is enabled.

**Tech Stack:** Python 3.11, PyTorch, OpenCLIP, torchvision, pytest, YAML.

---

### Task 1: Structured Model Output Helpers

**Files:**
- Create: `src/model_output.py`
- Modify: `src/eval.py`
- Modify: `src/train.py`
- Test: `tests/test_mlssa.py`

- [ ] Write tests showing that tensor, tuple, and dictionary model outputs expose logits and optional features through one helper.
- [ ] Run `PYTHONPATH=. pytest -q tests/test_mlssa.py -k model_output` and verify failure because `src.model_output` does not exist.
- [ ] Implement `unpack_model_output(output)` returning a dictionary with `logits`, `features`, `content_features`, `style_features`, and `style_layer_weights`.
- [ ] Replace tuple-only parsing in evaluation, dry-run, and training with this helper.
- [ ] Run the focused tests and existing core/revision tests.

### Task 2: Style Statistics and Orthogonality Loss

**Files:**
- Create: `src/models/style_statistics.py`
- Modify: `src/losses.py`
- Test: `tests/test_mlssa.py`

- [ ] Write tests for CLS exclusion, exact mean/std values, normalized learned layer weights, fused output normalization, and orthogonality loss.
- [ ] Run the focused tests and verify failures for missing symbols.
- [ ] Implement `patch_token_statistics`, `resolve_layer_indices`, `MultiLevelStyleStatistics`, and `content_style_orthogonality_loss`.
- [ ] Run focused tests until green.

### Task 3: CLIP MLSSA Classifier

**Files:**
- Create: `src/models/clip_mlssa.py`
- Modify: `src/models/registry.py`
- Test: `tests/test_mlssa.py`

- [ ] Write a fake OpenCLIP ViT test that verifies intermediate layer selection, dictionary output shapes, normalized features, and trainable final-block gradients.
- [ ] Run the focused test and verify failure because `clip_mlssa` is unsupported.
- [ ] Implement `ClipMLSSAClassifier` using `visual.forward_intermediates` with a narrow legacy OpenCLIP compatibility path.
- [ ] Register `clip_mlssa` and pass only MLSSA-specific configuration fields.
- [ ] Run focused tests until green.

### Task 4: Correct Training Objectives

**Files:**
- Modify: `src/train.py`
- Modify: `src/losses.py`
- Test: `tests/test_mlssa.py`

- [ ] Write tests showing that MixUp classification uses mixed images while SupCon and orthogonality use a clean second forward pass.
- [ ] Run the tests and verify the current single-forward implementation fails.
- [ ] Add `orthogonality_lambda` to `train_one_epoch`, accumulate/log `orthogonality_loss`, and use clean output for feature losses when MixUp is active.
- [ ] Keep energy and pseudo-unknown behavior unchanged.
- [ ] Run focused and full test suites.

### Task 5: Optimizer Grouping

**Files:**
- Modify: `src/train.py`
- Test: `tests/test_mlssa.py`

- [ ] Write a test proving MLSSA heads receive full learning rate and unfrozen CLIP parameters receive the scaled learning rate.
- [ ] Verify the test fails with current classifier keyword grouping.
- [ ] Add MLSSA module names to the existing CLIP head keyword set.
- [ ] Run focused tests until green.

### Task 6: Controlled Experiment Configurations

**Files:**
- Create: `configs/mlssa/clip_mlssa_full.yaml`
- Create: `configs/mlssa/clip_mlssa_no_orth.yaml`
- Create: `configs/mlssa/clip_mlssa_single_final.yaml`
- Create: `configs/mlssa/clip_mlssa_uniform_fusion.yaml`
- Create: `configs/mlssa/clip_mlssa_mean_only.yaml`
- Create: `configs/mlssa/clip_mlssa_frozen.yaml`
- Create: `run_mlssa_ablation_a100.py`
- Test: `tests/test_mlssa.py`

- [ ] Write tests that load every configuration, check final-test-only evaluation, and verify the intended controlled difference.
- [ ] Run tests and verify missing files fail.
- [ ] Add configurations with the same optimizer, augmentation, MixUp, label smoothing, and SupCon settings.
- [ ] Add an A100 sequential runner supporting seed overrides and experiment groups.
- [ ] Run configuration tests and script `--dry-run` argument tests.

### Task 7: Documentation and Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-06-25-multilevel-style-statistics-adapter-design.md` only if implementation constraints require a recorded correction.

- [ ] Document the new architecture, local weight path, dry-run command, and recommended A100 run order.
- [ ] Run `PYTHONPATH=. pytest -q`.
- [ ] Run `python -m compileall src run_mlssa_ablation_a100.py`.
- [ ] Run `git diff --check`.
- [ ] Inspect the final diff and confirm no unrelated existing changes were reverted.
