from __future__ import annotations

import importlib
from pathlib import Path

import torch
import pytest
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.eval import evaluate
from src.losses import content_style_orthogonality_loss
from src.model_output import unpack_model_output
from src.models.clip_mlssa import ClipMLSSAClassifier
from src.models.registry import CLIP_BUILDERS
from src.models.style_statistics import (
    MultiLevelStyleStatistics,
    patch_token_statistics,
    resolve_layer_indices,
)
from src.train import _build_optimizer, train_one_epoch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_model_output_unpacks_tensor_tuple_and_dictionary():
    logits = torch.randn(2, 3)
    features = torch.randn(2, 4)
    content = torch.randn(2, 4)
    style = torch.randn(2, 4)
    weights = torch.softmax(torch.randn(2, 3), dim=1)

    tensor_output = unpack_model_output(logits)
    tuple_output = unpack_model_output((logits, features))
    dict_output = unpack_model_output(
        {
            "logits": logits,
            "features": features,
            "content_features": content,
            "style_features": style,
            "style_layer_weights": weights,
        }
    )

    assert tensor_output["logits"] is logits
    assert tensor_output["features"] is None
    assert tuple_output["features"] is features
    assert dict_output["content_features"] is content
    assert dict_output["style_features"] is style
    assert dict_output["style_layer_weights"] is weights


def test_patch_token_statistics_excludes_cls_token_and_matches_manual_values():
    tokens = torch.tensor(
        [
            [
                [100.0, 200.0],
                [1.0, 2.0],
                [3.0, 6.0],
                [5.0, 10.0],
            ]
        ]
    )

    statistics = patch_token_statistics(tokens, include_std=True)

    patch_tokens = tokens[:, 1:]
    expected = torch.cat(
        [
            patch_tokens.mean(dim=1),
            patch_tokens.std(dim=1, unbiased=False),
        ],
        dim=1,
    )
    assert torch.allclose(statistics, expected)


def test_patch_token_statistics_can_return_mean_only():
    tokens = torch.tensor([[[99.0], [1.0], [3.0], [5.0]]])

    statistics = patch_token_statistics(tokens, include_std=False)

    assert torch.equal(statistics, torch.tensor([[3.0]]))


def test_resolve_layer_indices_supports_negative_indices_and_rejects_duplicates():
    assert resolve_layer_indices([3, -1], block_count=12) == [3, 11]

    try:
        resolve_layer_indices([3, -9], block_count=12)
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("Expected duplicate layer indices to fail.")


def test_multilevel_style_statistics_returns_normalized_features_and_weights():
    module = MultiLevelStyleStatistics(
        token_width=4,
        output_dim=6,
        style_dim=3,
        layer_count=2,
        fusion_hidden_dim=4,
        dropout=0.0,
        include_std=True,
        learned_fusion=True,
    )
    layer_tokens = [
        torch.randn(2, 5, 4),
        torch.randn(2, 5, 4),
    ]

    style_features, layer_weights = module(layer_tokens)

    assert style_features.shape == (2, 6)
    assert layer_weights.shape == (2, 2)
    assert torch.allclose(style_features.norm(dim=1), torch.ones(2), atol=1e-5)
    assert torch.allclose(layer_weights.sum(dim=1), torch.ones(2), atol=1e-6)


def test_orthogonality_loss_distinguishes_parallel_and_orthogonal_features():
    content = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    parallel = content.clone()
    orthogonal = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    parallel_loss = content_style_orthogonality_loss(content, parallel)
    orthogonal_loss = content_style_orthogonality_loss(content, orthogonal)

    assert torch.allclose(parallel_loss, torch.tensor(1.0))
    assert torch.allclose(orthogonal_loss, torch.tensor(0.0))


class _AddBlock(nn.Module):
    def __init__(self, width: int, value: float):
        super().__init__()
        self.bias = nn.Parameter(torch.full((width,), value))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.bias


class _FakeTransformer(nn.Module):
    def __init__(self, width: int, block_count: int):
        super().__init__()
        self.width = width
        self.resblocks = nn.ModuleList(
            [_AddBlock(width, float(index + 1)) for index in range(block_count)]
        )


class _FakeVisual(nn.Module):
    def __init__(self, width: int = 4, block_count: int = 4):
        super().__init__()
        self.output_dim = width
        self.transformer = _FakeTransformer(width, block_count)
        self.ln_pre = nn.LayerNorm(width)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(torch.eye(width))

    def forward_intermediates(
        self,
        images: torch.Tensor,
        indices: list[int],
        output_fmt: str,
        output_extra_tokens: bool,
    ) -> dict[str, object]:
        assert output_fmt == "NLC"
        assert output_extra_tokens is True
        batch_size = images.size(0)
        tokens = images.new_zeros(batch_size, 5, self.output_dim)
        intermediates = []
        prefixes = []
        for index, block in enumerate(self.transformer.resblocks):
            tokens = block(tokens)
            if index in indices:
                prefixes.append(tokens[:, :1])
                intermediates.append(tokens[:, 1:])
        pooled = self.ln_post(tokens[:, 0]) @ self.proj
        return {
            "image_features": pooled,
            "image_intermediates": intermediates,
            "image_intermediates_prefix": prefixes,
        }


class _FakeClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _FakeVisual()
        self.text_projection = nn.Parameter(torch.eye(4))


class _FakeOpenClip:
    def __init__(self, clip: nn.Module):
        self.clip = clip

    def create_model_and_transforms(self, *args, **kwargs):
        return self.clip, None, None


def test_clip_mlssa_returns_structured_normalized_features(monkeypatch):
    fake_clip = _FakeClip()
    monkeypatch.setattr(
        "src.models.clip_art._require_open_clip",
        lambda: _FakeOpenClip(fake_clip),
    )

    model = ClipMLSSAClassifier(
        num_classes=3,
        class_names=["a", "b", "c"],
        clip_model_name="ViT-B-16",
        pretrained="openai",
        style_layers=[1, 3],
        style_dim=3,
        fusion_hidden_dim=3,
        dropout=0.0,
        unfreeze_last_n_blocks=2,
    )
    output = model(torch.randn(2, 3, 8, 8))

    assert output["logits"].shape == (2, 3)
    assert output["features"].shape == (2, 4)
    assert output["content_features"].shape == (2, 4)
    assert output["style_features"].shape == (2, 4)
    assert output["style_layer_weights"].shape == (2, 2)
    assert torch.allclose(output["features"].norm(dim=1), torch.ones(2), atol=1e-5)
    assert torch.allclose(
        output["style_layer_weights"].sum(dim=1),
        torch.ones(2),
        atol=1e-6,
    )

    blocks = fake_clip.visual.transformer.resblocks
    assert not any(parameter.requires_grad for parameter in blocks[0].parameters())
    assert not any(parameter.requires_grad for parameter in blocks[1].parameters())
    assert all(parameter.requires_grad for parameter in blocks[2].parameters())
    assert all(parameter.requires_grad for parameter in blocks[3].parameters())
    assert not any(
        parameter.requires_grad for parameter in fake_clip.visual.ln_pre.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in fake_clip.visual.ln_post.parameters()
    )
    assert fake_clip.visual.proj.requires_grad


def test_clip_mlssa_is_registered():
    assert "clip_mlssa" in CLIP_BUILDERS


class _StructuredToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(3, 2)
        self.forward_inputs: list[torch.Tensor] = []

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        self.forward_inputs.append(images.detach().clone())
        pooled = images.mean(dim=(2, 3))
        content = nn.functional.normalize(pooled, dim=1)
        style = nn.functional.normalize(torch.roll(pooled, shifts=1, dims=1), dim=1)
        features = nn.functional.normalize(content + style, dim=1)
        return {
            "logits": self.classifier(features),
            "features": features,
            "content_features": content,
            "style_features": style,
            "style_layer_weights": images.new_full((images.size(0), 1), 1.0),
        }


def test_evaluate_accepts_structured_model_output():
    images = torch.randn(4, 3, 4, 4)
    targets = torch.tensor([0, 0, 1, 1])
    loader = DataLoader(TensorDataset(images, targets), batch_size=4)
    model = _StructuredToyModel()

    metrics = evaluate(model, loader, torch.device("cpu"), num_classes=2)

    assert set(metrics) >= {"accuracy", "macro_f1", "loss", "ece"}
    assert metrics["style_layer_weight_0"] == pytest.approx(1.0)


def test_mixup_uses_clean_second_pass_for_feature_losses():
    torch.manual_seed(7)
    images = torch.randn(4, 3, 4, 4)
    targets = torch.tensor([0, 0, 1, 1])
    loader = DataLoader(TensorDataset(images, targets), batch_size=4)
    model = _StructuredToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    metrics = train_one_epoch(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        contrastive_lambda=0.1,
        orthogonality_lambda=0.05,
        mixup_alpha=0.2,
        use_amp=False,
    )

    assert len(model.forward_inputs) == 2
    assert not torch.equal(model.forward_inputs[0], images)
    assert torch.equal(model.forward_inputs[1], images)
    assert metrics["contrastive_loss"] > 0
    assert metrics["orthogonality_loss"] >= 0


def test_mlssa_heads_use_full_learning_rate(monkeypatch):
    fake_clip = _FakeClip()
    monkeypatch.setattr(
        "src.models.clip_art._require_open_clip",
        lambda: _FakeOpenClip(fake_clip),
    )
    model = ClipMLSSAClassifier(
        num_classes=3,
        class_names=["a", "b", "c"],
        style_layers=[1, 3],
        style_dim=3,
        fusion_hidden_dim=3,
        dropout=0.0,
        unfreeze_last_n_blocks=2,
    )
    config = {
        "train": {
            "optimizer": "adamw",
            "lr": 3.0e-4,
            "weight_decay": 0.001,
            "backbone_lr_scale": 0.01,
        }
    }

    optimizer = _build_optimizer(model, config, architecture="clip_mlssa")
    lr_by_parameter = {
        id(parameter): group["lr"]
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("style_statistics.", "fusion_gate.", "classifier.")):
            assert lr_by_parameter[id(parameter)] == config["train"]["lr"], name
        elif name.startswith("clip.visual."):
            assert lr_by_parameter[id(parameter)] == pytest.approx(3.0e-6), name


def _load_mlssa_config(filename: str) -> dict:
    path = PROJECT_ROOT / "configs" / "mlssa" / filename
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_mlssa_configs_define_controlled_component_ablation():
    full = _load_mlssa_config("clip_mlssa_full.yaml")
    no_orth = _load_mlssa_config("clip_mlssa_no_orth.yaml")
    single = _load_mlssa_config("clip_mlssa_single_final.yaml")
    single_mean = _load_mlssa_config("clip_mlssa_single_final_mean_only.yaml")
    uniform = _load_mlssa_config("clip_mlssa_uniform_fusion.yaml")
    no_gate = _load_mlssa_config("clip_mlssa_no_gate.yaml")
    mean_only = _load_mlssa_config("clip_mlssa_mean_only.yaml")
    frozen = _load_mlssa_config("clip_mlssa_frozen.yaml")

    configs = [
        full,
        no_orth,
        single,
        single_mean,
        uniform,
        no_gate,
        mean_only,
        frozen,
    ]
    for config in configs:
        assert config["model"]["architecture"] == "clip_mlssa"
        assert config["eval"]["test_each_epoch"] is False
        assert config["eval"]["test_after_training"] is True
        assert config["train"]["mixup_alpha"] == 0.2
        assert config["loss"]["contrastive_lambda"] == 0.1
        assert config["loss"]["energy_lambda"] == 0.0
        assert config["loss"]["unknown_energy_lambda"] == 0.0

    assert full["model"]["style_layers"] == [3, 7, 11]
    assert full["loss"]["orthogonality_lambda"] == 0.05
    assert no_orth["loss"]["orthogonality_lambda"] == 0.0
    assert single["model"]["style_layers"] == [11]
    assert single_mean["model"]["style_layers"] == [11]
    assert single_mean["model"]["include_std"] is False
    assert uniform["model"]["learned_fusion"] is False
    assert no_gate["model"]["use_style_gate"] is False
    assert mean_only["model"]["include_std"] is False
    assert frozen["model"]["unfreeze_last_n_blocks"] == 0


def test_mlssa_runner_declares_core_and_component_groups():
    runner = importlib.import_module("run_mlssa_ablation_a100")

    assert set(runner.CONFIG_GROUPS) == {"comparison", "components", "layers"}
    assert all(path.is_file() for paths in runner.CONFIG_GROUPS.values() for path in paths)
    run_dir = runner._expected_run_dir(
        PROJECT_ROOT / "configs" / "mlssa" / "clip_mlssa_full.yaml",
        seed=41,
        hardware="a100",
    )
    assert run_dir.as_posix().endswith(
        "classify/outputs/runs/A100/clip_mlssa_full_seed41"
    )
