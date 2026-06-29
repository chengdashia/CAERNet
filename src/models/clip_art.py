from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.prompts import load_class_prompts


def _require_open_clip():
    try:
        import open_clip  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional package.
        raise ImportError(
            "CLIP/VLM experiments require open_clip_torch. "
            "Install it with: pip install open_clip_torch"
        ) from exc
    return open_clip


def _clip_embed_dim(model) -> int:
    if hasattr(model, "text_projection") and model.text_projection is not None:
        return int(model.text_projection.shape[1])
    if hasattr(model, "visual") and hasattr(model.visual, "output_dim"):
        return int(model.visual.output_dim)
    raise ValueError("Could not infer CLIP embedding dimension from model.")


def _build_feature_adapter(
    embed_dim: int,
    adapter_dim: int,
    baseline: bool,
    dropout: float,
) -> nn.Sequential:
    if baseline:
        return nn.Sequential(
            nn.Linear(embed_dim, adapter_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(adapter_dim, embed_dim, bias=False),
            nn.ReLU(inplace=True),
        )
    return nn.Sequential(
        nn.Linear(embed_dim, adapter_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(adapter_dim, embed_dim),
    )


class ClipArtClassifier(nn.Module):
    """CLIP image encoder with zero-shot, linear, or adapter heads."""

    def __init__(
        self,
        num_classes: int,
        class_names: list[str],
        mode: str,
        clip_model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        prompt_path: str | Path | None = None,
        adapter_dim: int = 512,
        dropout: float = 0.1,
        logit_scale: float = 100.0,
        unfreeze_last_n_blocks: int = 0,
        adapter_ratio: float = 0.2,
        unfreeze_visual: bool = False,
    ):
        super().__init__()
        if mode not in {"zero_shot", "linear_probe", "adapter", "clip_adapter_baseline"}:
            raise ValueError(f"Unsupported CLIP art mode: {mode}")
        self.mode = mode
        self.num_classes = num_classes
        self.class_names = class_names
        self.clip_model_name = clip_model_name
        self.logit_scale = logit_scale
        self.adapter_ratio = adapter_ratio
        self.clip_trainable = unfreeze_visual or unfreeze_last_n_blocks > 0

        open_clip = _require_open_clip()
        pretrained_arg = pretrained
        pretrained_path = Path(pretrained)
        if pretrained_path.suffix in {".bin", ".pt", ".pth", ".ckpt"}:
            if not pretrained_path.exists():
                raise FileNotFoundError(
                    "CLIP local weight file not found: "
                    f"{pretrained_path}. Put ViT-B/16 weights at this path "
                    "or update model.clip_pretrained in the YAML config."
                )
            pretrained_arg = str(pretrained_path)
        self.clip, _, _ = open_clip.create_model_and_transforms(
            clip_model_name,
            pretrained=pretrained_arg,
        )
        if mode == "clip_adapter_baseline" and hasattr(self.clip, "logit_scale"):
            self.logit_scale = float(self.clip.logit_scale.detach().exp())
        for parameter in self.clip.parameters():
            parameter.requires_grad = False
        if unfreeze_visual:
            self._unfreeze_visual_encoder()
        elif unfreeze_last_n_blocks > 0:
            self._unfreeze_last_visual_blocks(unfreeze_last_n_blocks)
        self.clip.eval()

        embed_dim = _clip_embed_dim(self.clip)
        self.adapter: nn.Module | None = None
        self.classifier: nn.Module | None = None

        if mode == "linear_probe":
            self.classifier = nn.Linear(embed_dim, num_classes)
        elif mode in {"adapter", "clip_adapter_baseline"}:
            self.adapter = _build_feature_adapter(
                embed_dim=embed_dim,
                adapter_dim=adapter_dim,
                baseline=mode == "clip_adapter_baseline",
                dropout=dropout,
            )
            if mode == "adapter":
                self.classifier = nn.Linear(embed_dim, num_classes)
            else:
                if prompt_path is None:
                    raise ValueError("clip_adapter_baseline requires model.prompt_path")
                text_features = self._build_text_prototypes(open_clip, prompt_path)
                self.register_buffer("text_features", text_features, persistent=False)
        else:
            if prompt_path is None:
                raise ValueError("clip_zero_shot requires model.prompt_path")
            text_features = self._build_text_prototypes(open_clip, prompt_path)
            self.register_buffer("text_features", text_features, persistent=False)

    def _unfreeze_last_visual_blocks(self, n_blocks: int):
        visual = getattr(self.clip, "visual", None)
        blocks = None
        if visual is not None:
            transformer = getattr(visual, "transformer", None)
            if transformer is not None and hasattr(transformer, "resblocks"):
                blocks = transformer.resblocks
            trunk = getattr(visual, "trunk", None)
            if blocks is None and trunk is not None and hasattr(trunk, "blocks"):
                blocks = trunk.blocks
        if blocks is None:
            raise ValueError(
                "Cannot unfreeze CLIP visual blocks for this model. "
                "Expected visual.transformer.resblocks or visual.trunk.blocks."
            )

        for block in list(blocks)[-n_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

        for module_name in ("ln_post", "norm", "fc_norm"):
            module = getattr(visual, module_name, None)
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad = True
        projection = getattr(visual, "proj", None)
        if isinstance(projection, nn.Parameter):
            projection.requires_grad = True

    def _unfreeze_visual_encoder(self):
        visual = getattr(self.clip, "visual", None)
        if visual is None:
            raise ValueError("Cannot unfreeze CLIP visual encoder: model has no visual module.")
        for parameter in visual.parameters():
            parameter.requires_grad = True

    @torch.no_grad()
    def _build_text_prototypes(self, open_clip, prompt_path: str | Path) -> torch.Tensor:
        tokenizer = open_clip.get_tokenizer(self.clip_model_name)
        prompt_map = load_class_prompts(prompt_path, self.class_names)
        prototypes = []
        device = next(self.clip.parameters()).device
        for class_name in self.class_names:
            tokens = tokenizer(prompt_map[class_name]).to(device)
            text_features = self.clip.encode_text(tokens)
            text_features = nn.functional.normalize(text_features.float(), dim=-1)
            prototype = nn.functional.normalize(text_features.mean(dim=0), dim=0)
            prototypes.append(prototype)
        return torch.stack(prototypes, dim=0)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        if not getattr(self, "clip_trainable", False):
            self.clip.eval()
        grad_enabled = getattr(self, "clip_trainable", False) and self.training
        with torch.set_grad_enabled(grad_enabled):
            features = self.clip.encode_image(images)
        features = nn.functional.normalize(features.float(), dim=-1)
        return features

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "clip_trainable", False):
            self.clip.train(mode)
        else:
            self.clip.eval()
        return self

    def forward(self, images: torch.Tensor):
        features = self.encode_image(images)
        if self.mode == "zero_shot":
            logits = self.logit_scale * features @ self.text_features.T
            return logits, features

        if self.adapter is not None:
            adapted = self.adapter(features)
            if self.mode == "clip_adapter_baseline":
                features = nn.functional.normalize(
                    (1.0 - self.adapter_ratio) * features
                    + self.adapter_ratio * adapted,
                    dim=-1,
                )
                logits = self.logit_scale * features @ self.text_features.T
                return logits, features
            features = nn.functional.normalize(features + adapted, dim=-1)

        if self.classifier is None:
            raise RuntimeError("CLIP classifier head was not initialized.")
        logits = self.classifier(features)
        return logits, features


def build_clip_zero_shot(num_classes: int, **kwargs) -> ClipArtClassifier:
    return ClipArtClassifier(num_classes=num_classes, mode="zero_shot", **kwargs)


def build_clip_linear_probe(num_classes: int, **kwargs) -> ClipArtClassifier:
    return ClipArtClassifier(num_classes=num_classes, mode="linear_probe", **kwargs)


def build_clip_adapter(num_classes: int, **kwargs) -> ClipArtClassifier:
    return ClipArtClassifier(num_classes=num_classes, mode="adapter", **kwargs)


def build_clip_adapter_baseline(num_classes: int, **kwargs) -> ClipArtClassifier:
    return ClipArtClassifier(
        num_classes=num_classes,
        mode="clip_adapter_baseline",
        **kwargs,
    )
