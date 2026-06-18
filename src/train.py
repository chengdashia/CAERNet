from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime.
    tqdm = None

CAERNET_ROOT = Path(__file__).resolve().parents[1]
CLASSIFY_ROOT = CAERNET_ROOT.parent
REPO_ROOT = CLASSIFY_ROOT.parent
if str(CAERNET_ROOT) not in sys.path:
    sys.path.insert(0, str(CAERNET_ROOT))

from src.datasets import build_dataloaders, build_imagefolder_dataset
from src.eval import evaluate
from src.losses import (
    classification_loss,
    energy_barrier_loss,
    energy_regularization_loss,
    mixup_data,
    mixup_cross_entropy,
    supervised_contrastive_loss,
)
from src.models import build_model


# Edit these parameters for the run you want, then start this file directly.
# Use RUN_NAME to keep outputs from different experiments separate.
RUN_NAME = "baseline_resnet50"

DATA = {
    "train_dir": CLASSIFY_ROOT / "data" / "artbench10" / "train",
    "val_dir": CLASSIFY_ROOT / "data" / "artbench10" / "test",
    "image_size": 224,
    "num_workers": 4,
}

MODEL = {
    "architecture": "resnet50",
    "pretrained": False,
    "use_coord_attention": False,
}

LOSS = {
    "energy_lambda": 0.0,
    "energy_margin": -5.0,
}

TRAIN = {
    "seed": 42,
    "device": "cuda",  # auto, cuda, or cpu
    "epochs": 40,
    "batch_size": 32,
    "lr": 0.0003,
    "weight_decay": 0.0001,
    "optimizer": "adamw",
    "scheduler": "none",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device: torch.device,
    epoch: int | None = None,
    total_epochs: int | None = None,
    energy_lambda: float = 0.0,
    energy_margin: float = -5.0,
    label_smoothing: float = 0.0,
    contrastive_lambda: float = 0.0,
    contrastive_temperature: float = 0.1,
    unknown_energy_lambda: float = 0.0,
    unknown_energy_margin: float = -1.0,
    gradient_clip: float = 0.0,
    mixup_alpha: float = 0.0,
    pseudo_unknown_mixup_alpha: float = 0.5,
    use_amp: bool = False,
    grad_accumulation_steps: int = 1,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    model.train()
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_loss = 0.0
    total_ce_loss = 0.0
    total_energy_loss = 0.0
    total_unknown_energy_loss = 0.0
    total_contrastive_loss = 0.0
    total_examples = 0
    total_correct = 0
    optimizer.zero_grad(set_to_none=True)

    progress_label = "train"
    if epoch is not None and total_epochs is not None:
        progress_label = f"epoch {epoch}/{total_epochs}"
    iterator = dataloader
    progress = None
    if tqdm is not None:
        progress = tqdm(
            dataloader,
            desc=progress_label,
            total=len(dataloader),
            dynamic_ncols=True,
            leave=True,
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| "
                "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            ),
        )
        iterator = progress

    for images, targets in iterator:
        images = images.to(device)
        targets = targets.to(device)
        original_images = images

        # Apply MixUp augmentation
        if mixup_alpha > 0:
            images, targets_a, targets_b, lam = mixup_data(images, targets, alpha=mixup_alpha)

        # Forward pass in AMP autocast; loss computed outside in FP32 for numerical stability
        with torch.amp.autocast("cuda", enabled=use_amp):
            output = model(images)

        if isinstance(output, tuple):
            logits, features = output
        else:
            logits, features = output, None

        # Cast to FP32 for numerically stable loss computation
        if use_amp:
            logits = logits.float()
            if features is not None:
                features = features.float()

        if mixup_alpha > 0:
            ce_loss = mixup_cross_entropy(logits, targets_a, targets_b, lam, label_smoothing=label_smoothing)
            energy_loss = energy_regularization_loss(logits, margin=energy_margin)
            if features is not None and contrastive_lambda > 0.0:
                contrastive_loss = supervised_contrastive_loss(
                    features, targets_a, temperature=contrastive_temperature,
                )
            else:
                contrastive_loss = logits.new_tensor(0.0)
            batch_loss = (
                ce_loss
                + energy_lambda * energy_loss
                + contrastive_lambda * contrastive_loss
            )
            parts = {
                "ce_loss": float(ce_loss.detach().cpu()),
                "energy_loss": float(energy_loss.detach().cpu()),
                "contrastive_loss": float(contrastive_loss.detach().cpu()),
            }
            loss = batch_loss
        else:
            loss, parts = classification_loss(
                logits,
                targets,
                energy_lambda=energy_lambda,
                energy_margin=energy_margin,
                label_smoothing=label_smoothing,
                features=features,
                contrastive_lambda=contrastive_lambda,
                contrastive_temperature=contrastive_temperature,
            )

        if unknown_energy_lambda > 0.0:
            index = torch.randperm(original_images.size(0), device=original_images.device)
            lam_unknown = pseudo_unknown_mixup_alpha
            pseudo_unknown_images = (
                lam_unknown * original_images
                + (1.0 - lam_unknown) * original_images[index]
            )
            with torch.amp.autocast("cuda", enabled=use_amp):
                unknown_output = model(pseudo_unknown_images)
            unknown_logits = unknown_output[0] if isinstance(unknown_output, tuple) else unknown_output
            if use_amp:
                unknown_logits = unknown_logits.float()

            # When MixUp is active, the main logits come from mixed images.
            # For the energy barrier we need clean ID logits from original
            # images so both sides of the barrier compare the same thing.
            if mixup_alpha > 0.0:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    id_barrier_output = model(original_images)
                id_barrier_logits = id_barrier_output[0] if isinstance(id_barrier_output, tuple) else id_barrier_output
                if use_amp:
                    id_barrier_logits = id_barrier_logits.float()
            else:
                id_barrier_logits = logits

            barrier_loss, barrier_parts = energy_barrier_loss(
                id_logits=id_barrier_logits,
                unknown_logits=unknown_logits,
                id_margin=energy_margin,
                unknown_margin=unknown_energy_lambda,
            )
            loss = loss + unknown_energy_lambda * barrier_loss
            parts["unknown_energy_loss"] = barrier_parts["unknown_energy_loss"]
        else:
            parts["unknown_energy_loss"] = 0.0

        # Scale loss for gradient accumulation
        loss = loss / grad_accumulation_steps
        scaler.scale(loss).backward()

        batch_size = targets.size(0)
        predictions = logits.argmax(dim=1)
        # Report unscaled loss for logging
        loss_val = (loss * grad_accumulation_steps).detach().cpu().item()
        total_loss += loss_val * batch_size
        total_ce_loss += parts["ce_loss"] * batch_size
        total_energy_loss += parts["energy_loss"] * batch_size
        total_unknown_energy_loss += parts["unknown_energy_loss"] * batch_size
        total_contrastive_loss += parts["contrastive_loss"] * batch_size
        if mixup_alpha > 0:
            # MixUp: accuracy is lam * acc_a + (1-lam) * acc_b
            correct_a = predictions.eq(targets_a).float()
            correct_b = predictions.eq(targets_b).float()
            total_correct += int((lam * correct_a + (1.0 - lam) * correct_b).sum().detach().cpu())
        else:
            total_correct += int(predictions.eq(targets).sum().detach().cpu())
        total_examples += batch_size

        # Step optimizer every grad_accumulation_steps micro-batches
        if (total_examples // batch_size) % grad_accumulation_steps == 0:
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if progress is not None:
            denominator = max(total_examples, 1)
            progress.set_postfix(
                {
                    "loss": f"{total_loss / denominator:.4f}",
                    "ce": f"{total_ce_loss / denominator:.4f}",
                    "contrast": f"{total_contrastive_loss / denominator:.4f}",
                    "acc": f"{total_correct / denominator:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

    if progress is not None:
        progress.close()

    denominator = max(total_examples, 1)
    return {
        "loss": _safe_scalar(total_loss) / denominator,
        "ce_loss": _safe_scalar(total_ce_loss) / denominator,
        "energy_loss": _safe_scalar(total_energy_loss) / denominator,
        "unknown_energy_loss": _safe_scalar(total_unknown_energy_loss) / denominator,
        "contrastive_loss": _safe_scalar(total_contrastive_loss) / denominator,
        "accuracy": _safe_scalar(total_correct) / denominator,
    }


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _default_config() -> dict:
    return {
        "data": DATA,
        "model": MODEL,
        "loss": LOSS,
        "train": TRAIN,
        "output": {
            "run_dir": CLASSIFY_ROOT / "outputs" / "runs" / RUN_NAME,
        },
    }


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested CUDA, but torch.cuda.is_available() is false.")
    return torch.device(device_name)


def _build_optimizer(model, config: dict, architecture: str = ""):
    optimizer_name = config["train"].get("optimizer", "adamw").lower()
    lr = config["train"]["lr"]
    weight_decay = config["train"].get("weight_decay", 0.0)
    backbone_lr_scale = config["train"].get("backbone_lr_scale", 1.0)

    # Build param groups for differential learning rates
    if backbone_lr_scale < 1.0:
        backbone_params = []
        head_params = []
        # Keywords that identify classifier/head layers (randomly initialised,
        # need full learning rate). Pretrained backbone layers get scaled LR.
        classifier_keywords = {"classifier", "fc.", "head.", "classifier_head"}
        # MTH model: randomly-initialised layers that must receive full LR
        if architecture == "mth_dcsam_csft":
            classifier_keywords |= {
                "attention",        # DCSAM
                "shallow_proj",     # shallow→transformer projection
                "deep_proj",        # deep→CNN projection
                "transformer",      # TransformerEncoder
                "embedding",        # fusion embedding MLP + BN
            }
        if architecture.startswith("ms_caernet_resnet50"):
            classifier_keywords |= {
                "attentions",       # coordinate-attention blocks
                "projectors",       # multi-scale projection heads
                "scale_gate",       # learnable scale fusion gate
            }
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(kw in name for kw in classifier_keywords):
                head_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = [
            {"params": backbone_params, "lr": lr * backbone_lr_scale, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr, "weight_decay": weight_decay},
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": lr, "weight_decay": weight_decay}]

    if optimizer_name == "adam":
        return torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=config["train"].get("momentum", 0.9),
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _build_scheduler(optimizer, config: dict):
    scheduler_name = config["train"].get("scheduler", "none").lower()

    if scheduler_name in {"none", "null"}:
        return None
    if scheduler_name == "cosine":
        total_epochs = config["train"]["epochs"]
        warmup_epochs = config["train"].get("warmup_epochs", 0)
        base_lr = config["train"]["lr"]
        min_lr = config["train"].get("min_lr", 0.0)
        min_lr_ratio = (min_lr / base_lr) if base_lr > 0 else 0.0

        def lr_lambda(epoch):
            # epoch = scheduler.last_epoch:
            #   0 at init (controls epoch 1's LR),
            #   1 after first step() (controls epoch 2's LR), etc.
            if epoch < warmup_epochs:
                return (epoch + 1) / max(warmup_epochs, 1)
            cosine_epoch = epoch - warmup_epochs
            cosine_total = max(total_epochs - warmup_epochs, 1)
            progress = min(cosine_epoch / cosine_total, 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def _apply_freeze_policy(model, config: dict):
    freeze = config["train"].get("freeze", "none").lower()
    if freeze in {"none", "false"}:
        return

    if freeze == "backbone":
        for parameter in model.parameters():
            parameter.requires_grad = False

        unfrozen = False
        for module_name in ("fc", "classifier", "head", "classifier_head"):
            module = getattr(model, module_name, None)
            if module is None:
                continue
            for parameter in module.parameters():
                parameter.requires_grad = True
            unfrozen = True

        if not unfrozen:
            for name, parameter in model.named_parameters():
                if "classifier" in name or name.endswith(".fc.weight") or name.endswith(".fc.bias"):
                    parameter.requires_grad = True
                    unfrozen = True

        if not unfrozen:
            raise ValueError("freeze=backbone could not identify a classifier/head module")
        return

    raise ValueError(f"Unsupported freeze policy: {freeze}")


def _to_yaml_safe(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_yaml_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_yaml_safe(item) for item in value]
    return value


def _safe_scalar(value: Any) -> float:
    """Convert any numeric value to a plain Python float, even if it is a
    0-dim torch tensor or numpy scalar that survives the conversion chain."""
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _to_float_if_possible(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _build_eval_loader(
    eval_dir: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    class_names: list[str],
    normalize: str = "imagenet",
):
    dataset = build_imagefolder_dataset(
        eval_dir,
        image_size,
        train=False,
        normalize=normalize,
    )
    if dataset.classes != class_names:
        raise ValueError(
            "Train and evaluation class folders must match. "
            f"train={class_names}, eval={dataset.classes}"
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )


def _write_history(history: list[dict[str, float]], path: Path):
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    tmp_path.replace(path)


def build_checkpoint_payload(
    model,
    optimizer,
    scheduler,
    class_names: list[str],
    config: dict,
    metrics: dict[str, float],
    epoch: int,
    best_epoch: int,
    best_accuracy: float,
) -> dict[str, Any]:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "class_names": class_names,
        "config": config,
        "metrics": metrics,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_accuracy": best_accuracy,
    }
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    else:
        payload["scheduler_state"] = None
    return payload


def _save_checkpoint(payload: dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _load_resume_checkpoint(
    resume_path: Path,
    model,
    optimizer,
    scheduler,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint


def _model_kwargs(model_config: dict) -> dict:
    ignored = {"architecture", "pretrained", "use_coord_attention"}
    return {key: value for key, value in model_config.items() if key not in ignored}


def train_from_config(config_path: Path | dict | None = None, dry_run: bool = False):
    if isinstance(config_path, dict):
        config = config_path
    else:
        config = _load_config(config_path) if config_path else _default_config()
    set_seed(config["train"].get("seed", 42))

    device = _resolve_device(config["train"].get("device", "cpu"))
    train_loader, val_loader, class_names = build_dataloaders(
        train_dir=config["data"]["train_dir"],
        val_dir=config["data"]["val_dir"],
        image_size=config["data"]["image_size"],
        batch_size=config["train"]["batch_size"],
        num_workers=config["data"].get("num_workers", 4),
        augment=config["data"].get("augment", "basic"),
        normalize=config["data"].get("normalize", "imagenet"),
    )
    model = build_model(
        architecture=config["model"]["architecture"],
        num_classes=len(class_names),
        class_names=class_names,
        pretrained=config["model"].get("pretrained", False),
        use_coord_attention=config["model"].get("use_coord_attention", False),
        **_model_kwargs(config["model"]),
    ).to(device)

    # Optional torch.compile for 20-40% speedup (PyTorch >= 2.0)
    if config["train"].get("compile", False):
        if hasattr(torch, "compile"):
            model = torch.compile(model, mode=config["train"].get("compile_mode", "reduce-overhead"))
            print(f"torch.compile enabled (mode={config['train'].get('compile_mode', 'reduce-overhead')})")
        else:
            print("WARNING: compile=True but torch.compile not available (need PyTorch >= 2.0)")

    _apply_freeze_policy(model, config)
    optimizer = _build_optimizer(model, config, architecture=config["model"]["architecture"])
    scheduler = _build_scheduler(optimizer, config)

    use_amp = config["train"].get("amp", False) and device.type == "cuda"
    if dry_run:
        images, _ = next(iter(train_loader))
        images = images.to(device)
        model.eval()
        with torch.no_grad():
            output = model(images[: min(2, images.size(0))])
        logits = output[0] if isinstance(output, tuple) else output
        print(
            {
                "dry_run": True,
                "architecture": config["model"]["architecture"],
                "classes": class_names,
                "logits_shape": tuple(logits.shape),
            }
        )
        return

    eval_config = config.get("eval", {})
    test_loader = None
    test_metric_prefix = "test"
    if eval_config.get("test_each_epoch", False):
        test_dir = eval_config.get("test_dir")
        if not test_dir:
            raise ValueError("eval.test_each_epoch=true requires eval.test_dir.")
        test_loader = _build_eval_loader(
            eval_dir=test_dir,
            image_size=config["data"]["image_size"],
            batch_size=config["train"]["batch_size"],
            num_workers=config["data"].get("num_workers", 4),
            class_names=class_names,
            normalize=config["data"].get("normalize", "imagenet"),
        )
        if eval_config.get("tta_horizontal_flip", False):
            test_metric_prefix = "test_tta"

    output_dir = Path(config["output"]["run_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(_to_yaml_safe(config), sort_keys=False),
        encoding="utf-8",
    )
    history_path = output_dir / "history.csv"
    best_accuracy = -1.0
    best_epoch = 0
    history = []
    if config["train"].get("eval_only", False):
        val_metrics = evaluate(model, val_loader, device, num_classes=len(class_names))
        row = {"epoch": 0, "lr": _safe_scalar(optimizer.param_groups[0]["lr"])}
        row.update({f"val_{key}": _safe_scalar(value) for key, value in val_metrics.items()})
        if test_loader is not None:
            test_metrics = evaluate(
                model,
                test_loader,
                device,
                num_classes=len(class_names),
                tta_horizontal_flip=eval_config.get("tta_horizontal_flip", False),
            )
            row.update({
                f"{test_metric_prefix}_{key}": _safe_scalar(value)
                for key, value in test_metrics.items()
            })
        row["best_epoch"] = 0
        row["best_val_accuracy"] = val_metrics["accuracy"]
        history.append(row)
        print(row)
        _write_history(history, history_path)
        return

    patience = config["train"].get("patience", 0)
    epochs_without_improvement = 0
    start_epoch = 1
    resume_path = config["train"].get("resume")
    if resume_path:
        checkpoint = _load_resume_checkpoint(
            resume_path=Path(resume_path),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_epoch = int(checkpoint.get("best_epoch", 0))
        best_accuracy = float(checkpoint.get("best_accuracy", checkpoint.get("metrics", {}).get("accuracy", -1.0)))
        if history_path.exists():
            with history_path.open("r", encoding="utf-8", newline="") as handle:
                history = [
                    {
                        key: _to_float_if_possible(value)
                        for key, value in row.items()
                    }
                    for row in csv.DictReader(handle)
                ]
        print(f"Resumed checkpoint: {resume_path} at epoch {start_epoch}")
    print(
        {
            "run_dir": str(output_dir),
            "device": str(device),
            "architecture": config["model"]["architecture"],
            "use_coord_attention": config["model"].get("use_coord_attention", False),
            "dropout": config["model"].get("dropout", 0.0),
            "energy_lambda": config["loss"].get("energy_lambda", 0.0),
            "label_smoothing": config["loss"].get("label_smoothing", 0.0),
            "contrastive_lambda": config["loss"].get("contrastive_lambda", 0.0),
            "epochs": config["train"]["epochs"],
            "batch_size": config["train"]["batch_size"],
            "optimizer": config["train"].get("optimizer", "adamw"),
            "scheduler": config["train"].get("scheduler", "none"),
            "lr": config["train"]["lr"],
            "weight_decay": config["train"].get("weight_decay", 0.0),
            "backbone_lr_scale": config["train"].get("backbone_lr_scale", 1.0),
            "warmup_epochs": config["train"].get("warmup_epochs", 0),
            "gradient_clip": config["train"].get("gradient_clip", 0.0),
            "mixup_alpha": config["train"].get("mixup_alpha", 0.0),
            "patience": config["train"].get("patience", 0),
            "amp": use_amp,
            "compile": config["train"].get("compile", False),
            "grad_accumulation_steps": config["train"].get("grad_accumulation_steps", 1),
        }
    )

    # Initialize GradScaler once so its dynamic state (growth interval,
    # backoff) persists across epochs instead of resetting every epoch.
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(start_epoch, config["train"]["epochs"] + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=config["train"]["epochs"],
            energy_lambda=config["loss"].get("energy_lambda", 0.0),
            energy_margin=config["loss"].get("energy_margin", -5.0),
            label_smoothing=config["loss"].get("label_smoothing", 0.0),
            contrastive_lambda=config["loss"].get("contrastive_lambda", 0.0),
            contrastive_temperature=config["loss"].get("contrastive_temperature", 0.1),
            unknown_energy_lambda=config["loss"].get("unknown_energy_lambda", 0.0),
            unknown_energy_margin=config["loss"].get("unknown_energy_margin", -1.0),
            gradient_clip=config["train"].get("gradient_clip", 0.0),
            mixup_alpha=config["train"].get("mixup_alpha", 0.0),
            pseudo_unknown_mixup_alpha=config["train"].get("pseudo_unknown_mixup_alpha", 0.5),
            use_amp=use_amp,
            grad_accumulation_steps=config["train"].get("grad_accumulation_steps", 1),
            scaler=amp_scaler,
        )
        val_metrics = evaluate(model, val_loader, device, num_classes=len(class_names))
        test_metrics = None
        if test_loader is not None:
            test_metrics = evaluate(
                model,
                test_loader,
                device,
                num_classes=len(class_names),
                tta_horizontal_flip=eval_config.get("tta_horizontal_flip", False),
            )

        row = {"epoch": epoch, "lr": _safe_scalar(optimizer.param_groups[0]["lr"])}
        row.update({f"train_{key}": _safe_scalar(value) for key, value in train_metrics.items()})
        row.update({f"val_{key}": _safe_scalar(value) for key, value in val_metrics.items()})
        if test_metrics is not None:
            row.update({
                f"{test_metric_prefix}_{key}": _safe_scalar(value)
                for key, value in test_metrics.items()
            })
        history.append(row)
        print(row)

        row["best_epoch"] = best_epoch
        row["best_val_accuracy"] = best_accuracy

        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            row["best_epoch"] = best_epoch
            row["best_val_accuracy"] = best_accuracy
            epochs_without_improvement = 0
            _save_checkpoint(
                build_checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    class_names=class_names,
                    config=config,
                    metrics=val_metrics,
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_accuracy=best_accuracy,
                ),
                output_dir / "best.pt",
            )
        elif patience > 0:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch}: "
                    f"no improvement for {patience} epochs "
                    f"(best val accuracy: {best_accuracy:.4f} at epoch {best_epoch})"
                )
                break
        _save_checkpoint(
            build_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                class_names=class_names,
                config=config,
                metrics=val_metrics,
                epoch=epoch,
                best_epoch=best_epoch,
                best_accuracy=best_accuracy,
            ),
            output_dir / "last.pt",
        )
        _write_history(history, history_path)
        if scheduler is not None:
            scheduler.step()

    _write_history(history, history_path)


def main():
    parser = argparse.ArgumentParser(description="Train an art style classifier.")
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional YAML config. If omitted, the parameters at the top of train.py are used.",
    )
    args = parser.parse_args()
    train_from_config(args.config)


if __name__ == "__main__":
    main()
