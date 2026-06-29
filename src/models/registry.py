from __future__ import annotations

from .ca_resnet import build_ca_resnet
from .clip_art import (
    build_clip_adapter,
    build_clip_adapter_baseline,
    build_clip_linear_probe,
    build_clip_zero_shot,
)
from .clip_mlssa import build_clip_mlssa
from .convnext_tiny import build_convnext_tiny
from .efficientnet_v2_s import build_efficientnet_v2_s
from .mobilenet_v3_small import build_mobilenet_v3_small
from .ms_caernet_full import build_ms_caernet_full
from .ms_caernet_no_contrastive import build_ms_caernet_no_contrastive
from .ms_caernet_no_energy import build_ms_caernet_no_energy
from .paper_mth import MobileNetTransformerHybrid
from .regnet_y import build_regnet_y_3_2gf, build_regnet_y_400mf
from .resnet18_baseline import build_resnet18
from .resnet50_baseline import build_resnet50


BASELINE_BUILDERS = {
    "resnet18": build_resnet18,
    "resnet50": build_resnet50,
    "mobilenet_v3_small": build_mobilenet_v3_small,
    "efficientnet_v2_s": build_efficientnet_v2_s,
    "convnext_tiny": build_convnext_tiny,
    "regnet_y_400mf": build_regnet_y_400mf,
    "regnet_y_3_2gf": build_regnet_y_3_2gf,
}

MS_CAERNET_BUILDERS = {
    "ms_caernet_resnet50": build_ms_caernet_full,
    "ms_caernet_resnet50_full": build_ms_caernet_full,
    "ms_caernet_resnet50_no_energy": build_ms_caernet_no_energy,
    "ms_caernet_resnet50_no_contrastive": build_ms_caernet_no_contrastive,
}

CLIP_BUILDERS = {
    "clip_zero_shot": build_clip_zero_shot,
    "clip_linear_probe": build_clip_linear_probe,
    "clip_adapter": build_clip_adapter,
    "clip_adapter_baseline": build_clip_adapter_baseline,
    "clip_mlssa": build_clip_mlssa,
}


def build_model(
    architecture: str,
    num_classes: int,
    class_names: list[str] | None = None,
    pretrained: bool = False,
    use_coord_attention: bool = False,
    **model_kwargs,
):
    """Dispatch model construction from YAML `architecture` names.

    Training scripts should call this single function; architecture-specific
    implementation details stay in their own model files.
    """
    if use_coord_attention:
        return build_ca_resnet(
            architecture=architecture,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=model_kwargs.get("dropout", 0.0),
        )

    if architecture in BASELINE_BUILDERS:
        return BASELINE_BUILDERS[architecture](
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=model_kwargs.get("dropout", 0.0),
        )

    if architecture in CLIP_BUILDERS:
        if class_names is None:
            raise ValueError(f"{architecture} requires class_names.")
        clip_kwargs = {
            "num_classes": num_classes,
            "class_names": class_names,
            "clip_model_name": model_kwargs.get("clip_model_name", "ViT-B-16"),
            "pretrained": model_kwargs.get("clip_pretrained", "openai"),
            "dropout": model_kwargs.get("dropout", 0.1),
            "unfreeze_last_n_blocks": model_kwargs.get("unfreeze_last_n_blocks", 0),
        }
        if architecture == "clip_mlssa":
            clip_kwargs.update(
                {
                    "style_layers": model_kwargs.get("style_layers", [3, 7, 11]),
                    "style_dim": model_kwargs.get("style_dim", 256),
                    "fusion_hidden_dim": model_kwargs.get("fusion_hidden_dim", 256),
                    "include_std": model_kwargs.get("include_std", True),
                    "learned_fusion": model_kwargs.get("learned_fusion", True),
                    "use_style_gate": model_kwargs.get("use_style_gate", True),
                }
            )
            return CLIP_BUILDERS[architecture](**clip_kwargs)
        return CLIP_BUILDERS[architecture](
            num_classes=num_classes,
            class_names=class_names,
            clip_model_name=model_kwargs.get("clip_model_name", "ViT-B-16"),
            pretrained=model_kwargs.get("clip_pretrained", "openai"),
            prompt_path=model_kwargs.get("prompt_path"),
            adapter_dim=model_kwargs.get("adapter_dim", 512),
            dropout=model_kwargs.get("dropout", 0.1),
            logit_scale=model_kwargs.get("logit_scale", 100.0),
            unfreeze_last_n_blocks=model_kwargs.get("unfreeze_last_n_blocks", 0),
            adapter_ratio=model_kwargs.get("adapter_ratio", 0.2),
            unfreeze_visual=model_kwargs.get("unfreeze_visual", False),
        )

    if architecture == "mth_dcsam_csft":
        return MobileNetTransformerHybrid(
            num_classes=num_classes,
            pretrained=pretrained,
            embed_dim=model_kwargs.get("embed_dim", 256),
            transformer_heads=model_kwargs.get("transformer_heads", 8),
            transformer_layers=model_kwargs.get("transformer_layers", 1),
            dropout=model_kwargs.get("dropout", 0.2),
            attention_reduction=model_kwargs.get("attention_reduction", 8),
            attention_kernel=model_kwargs.get("attention_kernel", 3),
            reduced_tail=model_kwargs.get("reduced_tail", True),
        )

    if architecture in MS_CAERNET_BUILDERS:
        return MS_CAERNET_BUILDERS[architecture](
            num_classes=num_classes,
            pretrained=pretrained,
            embed_dim=model_kwargs.get("embed_dim", 512),
            dropout=model_kwargs.get("dropout", 0.15),
            attention_reduction=model_kwargs.get("attention_reduction", 8),
        )

    supported = ", ".join(
        sorted([*BASELINE_BUILDERS, *MS_CAERNET_BUILDERS, *CLIP_BUILDERS, "mth_dcsam_csft"])
    )
    raise ValueError(f"Unsupported architecture: {architecture}. Supported: {supported}")
