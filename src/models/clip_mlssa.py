from __future__ import annotations

import torch
from torch import nn

from .clip_art import ClipArtClassifier
from .style_statistics import MultiLevelStyleStatistics, resolve_layer_indices


class ClipMLSSAClassifier(ClipArtClassifier):
    """CLIP classifier with multi-level patch-token style statistics."""

    def __init__(
        self,
        num_classes: int,
        class_names: list[str],
        clip_model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        style_layers: list[int] | None = None,
        style_dim: int = 256,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.1,
        unfreeze_last_n_blocks: int = 2,
        include_std: bool = True,
        learned_fusion: bool = True,
        use_style_gate: bool = True,
    ):
        super().__init__(
            num_classes=num_classes,
            class_names=class_names,
            mode="linear_probe",
            clip_model_name=clip_model_name,
            pretrained=pretrained,
            dropout=dropout,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )
        visual = self.clip.visual
        transformer = getattr(visual, "transformer", None)
        blocks = getattr(transformer, "resblocks", None)
        if blocks is None:
            raise ValueError(
                "clip_mlssa requires an OpenCLIP ViT with "
                "visual.transformer.resblocks."
            )

        self.style_layers = resolve_layer_indices(
            style_layers or [3, 7, 11],
            block_count=len(blocks),
        )
        token_width = getattr(transformer, "width", None)
        if token_width is None:
            token_width = getattr(getattr(visual, "conv1", None), "out_channels", 0)
        token_width = int(token_width)
        if token_width <= 0:
            raise ValueError("Could not infer CLIP visual token width.")
        output_dim = int(visual.output_dim)

        self.style_statistics = MultiLevelStyleStatistics(
            token_width=token_width,
            output_dim=output_dim,
            style_dim=style_dim,
            layer_count=len(self.style_layers),
            fusion_hidden_dim=fusion_hidden_dim,
            dropout=dropout,
            include_std=include_std,
            learned_fusion=learned_fusion,
        )
        self.use_style_gate = use_style_gate
        self.fusion_gate = nn.Linear(output_dim * 2, output_dim)
        self.classifier = nn.Linear(output_dim, num_classes)

    def _visual_intermediates(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        visual = self.clip.visual
        if hasattr(visual, "forward_intermediates"):
            output = visual.forward_intermediates(
                images,
                indices=self.style_layers,
                output_fmt="NLC",
                output_extra_tokens=True,
            )
            patches = output["image_intermediates"]
            prefixes = output.get("image_intermediates_prefix")
            if prefixes is None:
                raise ValueError(
                    "OpenCLIP forward_intermediates did not return CLS prefix tokens."
                )
            layer_tokens = [
                torch.cat((prefix, patch), dim=1)
                for prefix, patch in zip(prefixes, patches)
            ]
            return output["image_features"], layer_tokens

        transformer = visual.transformer
        if not hasattr(visual, "_embeds") or not hasattr(
            transformer,
            "forward_intermediates",
        ):
            raise ValueError(
                "Installed OpenCLIP version does not expose intermediate ViT tokens."
            )
        tokens = visual._embeds(images)
        tokens, layer_tokens = transformer.forward_intermediates(
            tokens,
            indices=self.style_layers,
        )
        pooled, _ = visual._pool(tokens)
        if visual.proj is not None:
            pooled = pooled @ visual.proj
        return pooled, layer_tokens

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if not self.clip_trainable:
            self.clip.eval()
        grad_enabled = self.clip_trainable and self.training
        with torch.set_grad_enabled(grad_enabled):
            content_features, layer_tokens = self._visual_intermediates(images)

        content_features = nn.functional.normalize(content_features.float(), dim=1)
        layer_tokens = [tokens.float() for tokens in layer_tokens]
        style_features, layer_weights = self.style_statistics(layer_tokens)
        if self.use_style_gate:
            gate = torch.sigmoid(
                self.fusion_gate(torch.cat((content_features, style_features), dim=1))
            )
            style_residual = gate * style_features
        else:
            style_residual = style_features
        features = nn.functional.normalize(content_features + style_residual, dim=1)
        logits = self.classifier(features)
        return {
            "logits": logits,
            "features": features,
            "content_features": content_features,
            "style_features": style_features,
            "style_layer_weights": layer_weights,
        }


def build_clip_mlssa(num_classes: int, **kwargs) -> ClipMLSSAClassifier:
    return ClipMLSSAClassifier(num_classes=num_classes, **kwargs)
