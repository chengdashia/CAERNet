from __future__ import annotations

from pathlib import Path

import yaml


def load_class_prompts(prompt_path: str | Path, class_names: list[str]) -> dict[str, list[str]]:
    """Load class-specific text prompts for CLIP-style prototype classifiers."""
    prompt_path = Path(prompt_path)
    config = yaml.safe_load(prompt_path.read_text(encoding="utf-8")) or {}
    templates = config.get("templates") or ["a painting in the {label} style"]
    classes = config.get("classes") or {}

    prompts: dict[str, list[str]] = {}
    missing = [class_name for class_name in class_names if class_name not in classes]
    if missing:
        raise ValueError(f"Prompt file {prompt_path} is missing classes: {missing}")

    for class_name in class_names:
        entry = classes[class_name] or {}
        label = entry.get("label", class_name.replace("_", " "))
        descriptions = entry.get("descriptions") or []
        class_prompts = [
            template.format(label=label, class_name=class_name)
            for template in templates
        ]
        class_prompts.extend(str(description) for description in descriptions)
        prompts[class_name] = class_prompts
    return prompts
