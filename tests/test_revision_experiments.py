from __future__ import annotations

import importlib
import json
import tempfile
from pathlib import Path

import torch
import yaml
from PIL import Image

from run_training import apply_cli_overrides
from src.datasets import build_dataloaders
from src.models.clip_art import ClipArtClassifier, _build_feature_adapter
from src.train import train_from_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_tiny_imagefolder(root: Path, count: int = 2) -> None:
    for class_index, class_name in enumerate(("baroque", "realism")):
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for image_index in range(count):
            value = 40 + class_index * 100 + image_index
            Image.new("RGB", (24, 24), color=(value, value, value)).save(
                class_dir / f"{image_index}.png"
            )


def test_final_test_evaluation_uses_best_checkpoint_and_writes_json() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_tiny_imagefolder(root / "train")
        _write_tiny_imagefolder(root / "val")
        _write_tiny_imagefolder(root / "test")
        run_dir = root / "run"
        config = {
            "data": {
                "train_dir": str(root / "train"),
                "val_dir": str(root / "val"),
                "image_size": 24,
                "num_workers": 0,
            },
            "model": {
                "architecture": "resnet18",
                "pretrained": False,
                "use_coord_attention": False,
            },
            "loss": {"energy_lambda": 0.0},
            "train": {
                "seed": 42,
                "device": "cpu",
                "epochs": 1,
                "batch_size": 2,
                "lr": 0.001,
                "weight_decay": 0.0,
                "optimizer": "sgd",
                "scheduler": "none",
            },
            "eval": {
                "test_each_epoch": False,
                "test_after_training": True,
                "test_dir": str(root / "test"),
                "tta_horizontal_flip": False,
            },
            "output": {"run_dir": str(run_dir)},
        }

        train_from_config(config)

        metrics_path = run_dir / "test_metrics.json"
        assert metrics_path.exists()
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert set(metrics) >= {"accuracy", "macro_f1", "ece", "loss"}
        history_header = (run_dir / "history.csv").read_text(encoding="utf-8").splitlines()[0]
        assert "test_accuracy" not in history_header


def test_cli_seed_override_creates_unique_run_directory() -> None:
    config = {
        "train": {"seed": 42},
        "output": {
            "run_name": "clip_revision",
            "run_dir": "classify/outputs/runs/A100/clip_revision",
        },
    }

    apply_cli_overrides(
        config,
        seed=41,
        run_suffix="seed41",
        final_test_only=True,
    )

    assert config["train"]["seed"] == 41
    assert config["output"]["run_name"] == "clip_revision_seed41"
    assert config["output"]["run_dir"] == "classify/outputs/runs/A100/clip_revision_seed41"
    assert config["eval"] == {
        "test_each_epoch": False,
        "test_after_training": True,
    }


def test_standard_clip_adapter_blends_adapter_and_pretrained_features() -> None:
    class IdentityClip(torch.nn.Module):
        def encode_image(self, images: torch.Tensor) -> torch.Tensor:
            return images

    class ConstantAdapter(torch.nn.Module):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[0.0, 1.0]], dtype=features.dtype).expand_as(features)

    model = ClipArtClassifier.__new__(ClipArtClassifier)
    torch.nn.Module.__init__(model)
    model.mode = "clip_adapter_baseline"
    model.clip = IdentityClip()
    model.clip_trainable = False
    model.adapter = ConstantAdapter()
    model.adapter_ratio = 0.25
    model.classifier = None
    model.logit_scale = 1.0
    model.register_buffer("text_features", torch.eye(2), persistent=False)

    logits, features = model(torch.tensor([[1.0, 0.0]]))

    expected = torch.nn.functional.normalize(torch.tensor([[0.75, 0.25]]), dim=-1)
    assert torch.allclose(features, expected)
    assert torch.allclose(logits, expected)


def test_standard_clip_adapter_uses_official_bottleneck_shape() -> None:
    adapter = _build_feature_adapter(
        embed_dim=512,
        adapter_dim=128,
        baseline=True,
        dropout=0.1,
    )

    assert isinstance(adapter[0], torch.nn.Linear)
    assert adapter[0].in_features == 512
    assert adapter[0].out_features == 128
    assert adapter[0].bias is None
    assert isinstance(adapter[-1], torch.nn.ReLU)
    assert adapter[2].bias is None


def test_full_visual_unfreeze_keeps_nonvisual_clip_parameters_frozen() -> None:
    class FakeClip(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.LayerNorm(4),
            )
            self.text = torch.nn.Linear(4, 4)

    model = ClipArtClassifier.__new__(ClipArtClassifier)
    torch.nn.Module.__init__(model)
    model.clip = FakeClip()
    for parameter in model.clip.parameters():
        parameter.requires_grad = False

    model._unfreeze_visual_encoder()

    assert all(parameter.requires_grad for parameter in model.clip.visual.parameters())
    assert not any(parameter.requires_grad for parameter in model.clip.text.parameters())


def test_low_data_fraction_is_stratified_and_deterministic() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_tiny_imagefolder(root / "train", count=10)
        _write_tiny_imagefolder(root / "val", count=2)

        first, _, _ = build_dataloaders(
            train_dir=root / "train",
            val_dir=root / "val",
            image_size=24,
            batch_size=2,
            num_workers=0,
            train_fraction=0.2,
            subset_seed=41,
        )
        second, _, _ = build_dataloaders(
            train_dir=root / "train",
            val_dir=root / "val",
            image_size=24,
            batch_size=2,
            num_workers=0,
            train_fraction=0.2,
            subset_seed=41,
        )

        assert len(first.dataset) == 4
        assert first.dataset.indices == second.dataset.indices
        targets = [first.dataset.dataset.targets[index] for index in first.dataset.indices]
        assert targets.count(0) == 2
        assert targets.count(1) == 2


def test_revision_a100_script_lists_all_controlled_groups() -> None:
    runner = importlib.import_module("run_revision_ablation_a100")

    assert set(runner.CONFIG_GROUPS) == {
        "adaptation",
        "depth",
        "regularizers",
        "clip_adapter",
        "low_data",
    }
    assert len(runner.CONFIG_GROUPS["adaptation"]) == 6
    assert len(runner.CONFIG_GROUPS["depth"]) == 5
    assert len(runner.CONFIG_GROUPS["regularizers"]) == 5
    assert len(runner.CONFIG_GROUPS["low_data"]) == 4
    for paths in runner.CONFIG_GROUPS.values():
        for path in paths:
            assert path.is_file(), path
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            assert config["eval"]["test_each_epoch"] is False
            assert config["eval"]["test_after_training"] is True
    for filename in ("adaptation_full_visual.yaml", "adaptation_adapter_full_visual.yaml"):
        config = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "revision" / filename).read_text(encoding="utf-8")
        )
        assert config["model"]["unfreeze_visual"] is True


def test_official_baseline_script_declares_requested_methods() -> None:
    runner = importlib.import_module("run_official_clip_baselines_a100")

    assert set(runner.BASELINE_REPOS) == {
        "coop",
        "cocoop",
        "maple",
        "promptsrc",
        "tip_adapter",
    }


def test_revision_baseline_script_runs_every_trainable_main_table_baseline() -> None:
    runner = importlib.import_module("run_revision_baselines_a100")

    assert len(runner.CONFIGS) == 7
    assert all(path.is_file() for path in runner.CONFIGS)
