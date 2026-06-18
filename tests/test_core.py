import tempfile
import json
import importlib
from pathlib import Path

import torch
import yaml
from PIL import Image

from src.datasets import build_dataloaders
from src.diagnostics import class_counts, summarize_history
from src.losses import energy_barrier_loss, energy_regularization_loss
from src.metrics import classification_metrics
from src.models import build_model as build_registered_model
from src.models.clip_art import ClipArtClassifier
from src.models.coord_attention import CoordAttention
from src.models.resnet_ca import build_model
from src.model_summary import summarize_model
from src.collect_results import collect_run_result
from src.eval_run import resolve_eval_paths
from src.train import _build_optimizer
from src.train import build_checkpoint_payload
from src.train import train_from_config
from src.train import train_one_epoch
from src.eval import evaluate
from run_training import build_effective_config


def _write_tiny_imagefolder(root: Path, classes=("baroque", "cubism"), count=3):
    for class_index, class_name in enumerate(classes):
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for image_index in range(count):
            value = 30 + class_index * 80 + image_index
            image = Image.new("RGB", (32, 32), color=(value, value, value))
            image.save(class_dir / f"{image_index}.png")


def test_energy_regularization_loss_is_differentiable():
    logits = torch.tensor([[2.0, 0.3], [0.1, 1.4]], requires_grad=True)

    loss = energy_regularization_loss(logits, margin=-1.0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_coordinate_attention_preserves_feature_shape():
    module = CoordAttention(channels=16)
    features = torch.randn(2, 16, 8, 10)

    output = module(features)

    assert output.shape == features.shape


def test_resnet50_model_builds_for_baseline_reproduction():
    model = build_model(
        architecture="resnet50",
        num_classes=10,
        pretrained=False,
        use_coord_attention=False,
    )
    logits = model(torch.randn(2, 3, 64, 64))

    assert logits.shape == (2, 10)


def test_clean_baseline_backbones_build_for_artbench():
    architectures = [
        "resnet50",
        "mobilenet_v3_small",
        "efficientnet_v2_s",
        "convnext_tiny",
        "regnet_y_400mf",
    ]

    for architecture in architectures:
        model = build_model(
            architecture=architecture,
            num_classes=10,
            pretrained=False,
            use_coord_attention=False,
        )
        model.eval()
        with torch.no_grad():
            logits = model(torch.randn(1, 3, 64, 64))

        assert logits.shape == (1, 10)


def test_core_models_have_independent_builder_modules():
    expected_builders = {
        "src.models.resnet50_baseline": "build_resnet50",
        "src.models.convnext_tiny": "build_convnext_tiny",
        "src.models.efficientnet_v2_s": "build_efficientnet_v2_s",
        "src.models.ms_caernet": "MultiScaleCAERNet",
        "src.models.ms_caernet_full": "build_ms_caernet_full",
        "src.models.ms_caernet_no_energy": "build_ms_caernet_no_energy",
        "src.models.ms_caernet_no_contrastive": "build_ms_caernet_no_contrastive",
    }

    for module_name, symbol_name in expected_builders.items():
        module = importlib.import_module(module_name)
        assert hasattr(module, symbol_name), f"{module_name}.{symbol_name}"


def test_registry_is_the_single_model_dispatch_entrypoint():
    registry = importlib.import_module("src.models.registry")

    assert registry.build_model is build_registered_model


def test_history_diagnostics_report_best_epochs_and_gap():
    rows = [
        {
            "epoch": 1.0,
            "train_accuracy": 0.70,
            "val_accuracy": 0.60,
            "val_loss": 0.90,
            "test_tta_accuracy": 0.61,
        },
        {
            "epoch": 2.0,
            "train_accuracy": 0.85,
            "val_accuracy": 0.68,
            "val_loss": 0.80,
            "test_tta_accuracy": 0.69,
        },
        {
            "epoch": 3.0,
            "train_accuracy": 0.90,
            "val_accuracy": 0.66,
            "val_loss": 0.95,
            "test_tta_accuracy": 0.67,
        },
    ]

    summary = summarize_history(rows)

    assert summary["best_val_epoch"] == 2
    assert summary["best_test_epoch"] == 2
    assert summary["min_val_loss_epoch"] == 2
    assert abs(summary["generalization_gap_at_best_val"] - 0.17) < 1e-12


def test_dataset_diagnostics_count_imagefolder_classes(tmp_path):
    root = tmp_path / "data"
    _write_tiny_imagefolder(root, classes=("a", "b"), count=2)

    counts = class_counts(root)

    assert counts == {"a": 2, "b": 2}


def test_ms_caernet_ablation_architecture_names_build():
    for architecture in (
        "ms_caernet_resnet50_full",
        "ms_caernet_resnet50_no_energy",
        "ms_caernet_resnet50_no_contrastive",
    ):
        model = build_registered_model(
            architecture=architecture,
            num_classes=10,
            pretrained=False,
            embed_dim=64,
            dropout=0.0,
        )
        model.eval()

        with torch.no_grad():
            logits, embedding = model(torch.randn(2, 3, 64, 64))

        assert logits.shape == (2, 10)
        assert embedding.shape == (2, 64)


def test_caernet_can_use_resnet50_backbone():
    model = build_model(
        architecture="resnet50",
        num_classes=10,
        pretrained=False,
        use_coord_attention=True,
    )
    logits = model(torch.randn(2, 3, 64, 64))

    assert logits.shape == (2, 10)


def test_ms_caernet_returns_logits_and_embedding_for_contrastive_training():
    model = build_registered_model(
        architecture="ms_caernet_resnet50",
        num_classes=10,
        pretrained=False,
        embed_dim=128,
        dropout=0.1,
    )
    model.eval()

    with torch.no_grad():
        logits, embedding = model(torch.randn(2, 3, 64, 64))

    assert logits.shape == (2, 10)
    assert embedding.shape == (2, 128)


def test_ms_caernet_new_modules_use_full_learning_rate():
    model = build_registered_model(
        architecture="ms_caernet_resnet50",
        num_classes=10,
        pretrained=False,
        embed_dim=128,
        dropout=0.1,
    )
    config = {
        "train": {
            "optimizer": "adamw",
            "lr": 8.0e-5,
            "weight_decay": 0.001,
            "backbone_lr_scale": 0.1,
        },
    }

    optimizer = _build_optimizer(
        model,
        config,
        architecture="ms_caernet_resnet50",
    )
    lr_by_parameter = {
        id(parameter): group["lr"]
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    for name, parameter in model.named_parameters():
        if name.startswith(("attentions.", "projectors.", "scale_gate.", "classifier.")):
            assert lr_by_parameter[id(parameter)] == config["train"]["lr"], name


def test_model_summary_reports_parameter_counts():
    model = build_registered_model(
        architecture="ms_caernet_resnet50",
        num_classes=10,
        pretrained=False,
        embed_dim=128,
    )

    summary = summarize_model(model)

    assert summary["total_params"] > 0
    assert summary["trainable_params"] > 0
    assert summary["total_params_m"] == round(summary["total_params"] / 1_000_000, 4)


def test_collect_run_result_reads_history_and_test_metrics():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        (run_dir / "history.csv").write_text(
            "epoch,val_accuracy,val_macro_f1,train_loss\n"
            "1,0.40,0.35,1.0\n"
            "2,0.55,0.50,0.8\n",
            encoding="utf-8",
        )
        (run_dir / "test_metrics.json").write_text(
            json.dumps({"accuracy": 0.53, "macro_f1": 0.49, "ece": 0.08}),
            encoding="utf-8",
        )

        row = collect_run_result("demo", run_dir)

    assert row["name"] == "demo"
    assert row["best_epoch"] == 2
    assert row["best_val_accuracy"] == 0.55
    assert row["test_accuracy"] == 0.53
    assert row["test_macro_f1"] == 0.49


def test_resolve_eval_paths_defaults_to_best_checkpoint_and_test_metrics():
    paths = resolve_eval_paths(
        run_dir=Path("classify/outputs/runs/demo"),
        config=Path("classify/CAERNet/configs/demo_eval.yaml"),
        checkpoint=None,
        output=None,
    )

    assert paths["run_dir"] == Path("classify/outputs/runs/demo")
    assert paths["checkpoint"] == Path("classify/outputs/runs/demo/best.pt")
    assert paths["output"] == Path("classify/outputs/runs/demo/test_metrics.json")
    assert paths["config"] == Path("classify/CAERNet/configs/demo_eval.yaml")


def test_checkpoint_payload_contains_resume_state():
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    payload = build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        class_names=["a", "b"],
        config={"model": {"architecture": "linear"}},
        metrics={"accuracy": 0.5},
        epoch=3,
        best_epoch=2,
        best_accuracy=0.6,
    )

    assert set(payload) >= {
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "class_names",
        "config",
        "metrics",
        "epoch",
        "best_epoch",
        "best_accuracy",
    }
    assert payload["epoch"] == 3
    assert payload["best_epoch"] == 2
    assert payload["best_accuracy"] == 0.6


def test_metrics_report_macro_scores():
    metrics = classification_metrics(
        y_true=[0, 1, 1, 0],
        y_pred=[0, 1, 0, 0],
        num_classes=2,
    )

    assert set(metrics) >= {"accuracy", "macro_f1", "precision", "recall"}
    assert metrics["accuracy"] == 0.75
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_train_and_eval_run_on_tiny_imagefolder():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_tiny_imagefolder(root / "train")
        _write_tiny_imagefolder(root / "val")

        train_loader, val_loader, class_names = build_dataloaders(
            train_dir=root / "train",
            val_dir=root / "val",
            image_size=32,
            batch_size=2,
            num_workers=0,
        )
        model = build_model(
            architecture="resnet18",
            num_classes=len(class_names),
            pretrained=False,
            use_coord_attention=True,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=torch.device("cpu"),
            energy_lambda=0.01,
            energy_margin=-1.0,
        )
        eval_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=torch.device("cpu"),
            num_classes=len(class_names),
        )

        assert train_metrics["loss"] > 0
        assert 0.0 <= eval_metrics["accuracy"] <= 1.0


def test_train_from_config_can_log_test_tta_metrics_each_epoch():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_tiny_imagefolder(root / "train")
        _write_tiny_imagefolder(root / "val")
        _write_tiny_imagefolder(root / "test")
        config_path = root / "config.yaml"
        run_dir = root / "run"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "data": {
                        "train_dir": str(root / "train"),
                        "val_dir": str(root / "val"),
                        "image_size": 32,
                        "num_workers": 0,
                    },
                    "model": {
                        "architecture": "resnet18",
                        "pretrained": False,
                        "use_coord_attention": False,
                    },
                    "loss": {
                        "energy_lambda": 0.0,
                        "energy_margin": -5.0,
                    },
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
                        "test_each_epoch": True,
                        "test_dir": str(root / "test"),
                        "tta_horizontal_flip": True,
                    },
                    "output": {
                        "run_dir": str(run_dir),
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        train_from_config(config_path)

        history = (run_dir / "history.csv").read_text(encoding="utf-8").splitlines()
        assert "test_tta_accuracy" in history[0]
        assert "test_tta_loss" in history[0]


def test_evaluate_can_average_horizontal_flip_tta_logits():
    class EdgeModel(torch.nn.Module):
        def forward(self, images):
            left = images[:, :, :, 0].sum(dim=(1, 2))
            right = images[:, :, :, -1].sum(dim=(1, 2))
            return torch.stack([left, right], dim=1)

    image = torch.zeros(1, 3, 4, 4)
    image[:, :, :, 0] = 1.0
    image[:, :, :, -1] = 3.0
    target = torch.tensor([0])
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(image, target),
        batch_size=1,
    )

    no_tta = evaluate(
        model=EdgeModel(),
        dataloader=dataloader,
        device=torch.device("cpu"),
        num_classes=2,
        tta_horizontal_flip=False,
    )
    with_tta = evaluate(
        model=EdgeModel(),
        dataloader=dataloader,
        device=torch.device("cpu"),
        num_classes=2,
        tta_horizontal_flip=True,
    )

    assert no_tta["accuracy"] == 0.0
    assert with_tta["accuracy"] == 1.0


def test_mixup_training_still_reports_energy_loss():
    images = torch.zeros(4, 3, 8, 8)
    targets = torch.tensor([0, 1, 0, 1])
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(images, targets),
        batch_size=4,
    )
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 2))
    torch.nn.init.zeros_(model[1].weight)
    torch.nn.init.zeros_(model[1].bias)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    metrics = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        energy_lambda=0.01,
        energy_margin=-5.0,
        mixup_alpha=0.2,
    )

    assert metrics["energy_loss"] > 0


def test_unknown_energy_barrier_uses_configured_unknown_margin():
    images = torch.zeros(4, 3, 8, 8)
    targets = torch.tensor([0, 1, 0, 1])
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(images, targets),
        batch_size=4,
    )
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 2))
    torch.nn.init.zeros_(model[1].weight)
    torch.nn.init.zeros_(model[1].bias)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    metrics = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        energy_margin=0.0,
        unknown_energy_lambda=0.01,
        unknown_energy_margin=-1.0,
    )

    assert metrics["unknown_energy_loss"] == 0.0


def test_paper_mth_reproduction_is_lightweight():
    model = build_registered_model(
        architecture="mth_dcsam_csft",
        num_classes=10,
        pretrained=False,
    )

    param_count_m = sum(parameter.numel() for parameter in model.parameters()) / 1_000_000

    assert 1.1 <= param_count_m <= 1.35


def test_paper_training_configs_use_validation_split():
    config_root = Path(__file__).resolve().parents[1] / "configs"
    expected_val = "classify/data/artbench10_paper/val"
    training_configs = [
        config_root / "baselines" / "resnet50_paper.yaml",
        config_root / "baselines" / "mth_dcsam_csft.yaml",
        config_root / "caernet_resnet50.yaml",
        config_root / "ms_caernet_resnet50.yaml",
        config_root / "ms_caernet_resnet50_no_energy.yaml",
        config_root / "ms_caernet_resnet50_no_contrastive.yaml",
    ]
    for config_path in training_configs:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert config["data"]["val_dir"] == expected_val


def test_clean_resnet50_config_is_ce_only():
    code_root = Path(__file__).resolve().parents[1]
    config_path = code_root / "configs" / "baselines" / "resnet50_clean_ce.yaml"

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["model"]["architecture"] == "resnet50"
    assert config["model"]["pretrained"] is True
    assert config["model"]["dropout"] == 0.0
    assert config["loss"]["label_smoothing"] == 0.0
    assert config["loss"]["energy_lambda"] == 0.0
    assert config["loss"].get("contrastive_lambda", 0.0) == 0.0
    assert config["train"]["mixup_alpha"] == 0.0
    assert config["train"]["backbone_lr_scale"] == 1.0
    assert config["data"]["val_dir"] == "classify/data/artbench10_paper/val"


def test_energy_barrier_loss_penalizes_low_unknown_energy():
    id_logits = torch.tensor([[8.0, 0.1], [0.2, 7.5]], requires_grad=True)
    unknown_logits = torch.tensor([[8.0, 0.1], [0.2, 7.5]], requires_grad=True)

    loss, parts = energy_barrier_loss(
        id_logits=id_logits,
        unknown_logits=unknown_logits,
        id_margin=-5.0,
        unknown_margin=-1.0,
    )

    assert loss.ndim == 0
    assert parts["id_energy_loss"] == 0.0
    assert parts["unknown_energy_loss"] > 0.0
    loss.backward()
    assert unknown_logits.grad is not None


def test_prompt_config_expands_label_and_descriptions(tmp_path):
    from src.prompts import load_class_prompts

    prompt_path = tmp_path / "prompts.yaml"
    prompt_path.write_text(
        yaml.safe_dump(
            {
                "templates": ["a painting in the {label} style"],
                "classes": {
                    "baroque": {
                        "label": "Baroque",
                        "descriptions": ["dramatic light and shadow"],
                    },
                    "ukiyo_e": {
                        "label": "Ukiyo-e",
                        "descriptions": ["flat color woodblock print"],
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    prompts = load_class_prompts(prompt_path, ["baroque", "ukiyo_e"])

    assert prompts["baroque"] == [
        "a painting in the Baroque style",
        "dramatic light and shadow",
    ]
    assert prompts["ukiyo_e"][0] == "a painting in the Ukiyo-e style"


def test_clip_architectures_are_registered_without_importing_open_clip():
    from src.models.registry import CLIP_BUILDERS

    assert "clip_zero_shot" in CLIP_BUILDERS
    assert "clip_linear_probe" in CLIP_BUILDERS
    assert "clip_adapter" in CLIP_BUILDERS


def test_frozen_clip_encoder_stays_eval_when_head_trains():
    model = ClipArtClassifier.__new__(ClipArtClassifier)
    torch.nn.Module.__init__(model)
    model.clip = torch.nn.Dropout(p=0.5)
    model.adapter = None
    model.classifier = torch.nn.Linear(4, 2)

    model.train()

    assert model.training is True
    assert model.classifier.training is True
    assert model.clip.training is False


def test_clip_partial_unfreeze_enables_last_visual_blocks_only():
    class FakeVisual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.Module()
            self.transformer.resblocks = torch.nn.ModuleList(
                [torch.nn.Linear(4, 4) for _ in range(3)]
            )
            self.ln_post = torch.nn.LayerNorm(4)
            self.proj = torch.nn.Parameter(torch.ones(4, 4))

    class FakeClip(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = FakeVisual()

    model = ClipArtClassifier.__new__(ClipArtClassifier)
    torch.nn.Module.__init__(model)
    model.clip = FakeClip()
    for parameter in model.clip.parameters():
        parameter.requires_grad = False

    model._unfreeze_last_visual_blocks(1)

    blocks = model.clip.visual.transformer.resblocks
    assert not any(parameter.requires_grad for parameter in blocks[0].parameters())
    assert not any(parameter.requires_grad for parameter in blocks[1].parameters())
    assert all(parameter.requires_grad for parameter in blocks[2].parameters())
    assert all(parameter.requires_grad for parameter in model.clip.visual.ln_post.parameters())
    assert model.clip.visual.proj.requires_grad is True


def test_clip_adapter_uses_full_lr_when_backbone_lr_is_scaled():
    class TinyClipAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.clip = torch.nn.Linear(4, 4)
            self.adapter = torch.nn.Linear(4, 4)
            self.classifier = torch.nn.Linear(4, 2)

    model = TinyClipAdapter()
    config = {
        "train": {
            "optimizer": "adamw",
            "lr": 3.0e-4,
            "weight_decay": 0.001,
            "backbone_lr_scale": 0.01,
        },
    }

    optimizer = _build_optimizer(model, config, architecture="clip_adapter")
    lr_by_parameter = {
        id(parameter): group["lr"]
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    for name, parameter in model.named_parameters():
        if name.startswith(("adapter.", "classifier.")):
            assert lr_by_parameter[id(parameter)] == config["train"]["lr"], name
        if name.startswith("clip."):
            assert lr_by_parameter[id(parameter)] == config["train"]["lr"] * 0.01, name


def test_clip_probe_augmentation_avoids_random_erasing():
    from src.datasets import build_transforms

    transform = build_transforms(
        image_size=224,
        train=True,
        augment="clip_probe",
        normalize="clip",
    )

    assert not any(step.__class__.__name__ == "RandomErasing" for step in transform.transforms)


def test_hardware_profile_overrides_batch_workers_and_run_dir(tmp_path):
    config_path = tmp_path / "experiment.yaml"
    hardware_path = tmp_path / "a100.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "train_dir": "classify/data/artbench10/train",
                    "val_dir": "classify/data/artbench10_paper/val",
                    "image_size": 224,
                    "num_workers": 2,
                },
                "model": {"architecture": "resnet50", "pretrained": False},
                "loss": {"energy_lambda": 0.0},
                "train": {"batch_size": 8, "epochs": 1, "device": "cpu"},
                "output": {"run_name": "clip_probe"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    hardware_path.write_text(
        yaml.safe_dump(
            {
                "hardware_tag": "A100",
                "data": {"num_workers": 16},
                "train": {"batch_size": 128, "amp": True},
                "output": {"run_root": "classify/outputs/runs/A100"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = build_effective_config(config_path, hardware_path)

    assert config["data"]["num_workers"] == 16
    assert config["train"]["batch_size"] == 128
    assert config["train"]["amp"] is True
    assert config["output"]["run_dir"] == "classify/outputs/runs/A100/clip_probe"
