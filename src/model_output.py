from __future__ import annotations

from typing import Any

import torch


def unpack_model_output(output: Any) -> dict[str, torch.Tensor | None]:
    fields = {
        "logits": None,
        "features": None,
        "content_features": None,
        "style_features": None,
        "style_layer_weights": None,
    }
    if isinstance(output, dict):
        fields.update({key: output.get(key) for key in fields})
    elif isinstance(output, tuple):
        fields["logits"] = output[0]
        if len(output) > 1:
            fields["features"] = output[1]
    else:
        fields["logits"] = output

    if fields["logits"] is None:
        raise ValueError("Model output does not contain logits.")
    return fields
