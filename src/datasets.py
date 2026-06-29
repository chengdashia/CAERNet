from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


NORMALIZATION_STATS = {
    "imagenet": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "clip": {
        "mean": (0.48145466, 0.4578275, 0.40821073),
        "std": (0.26862954, 0.26130258, 0.27577711),
    },
}


def _normalize_transform(normalize: str):
    if normalize not in NORMALIZATION_STATS:
        supported = ", ".join(sorted(NORMALIZATION_STATS))
        raise ValueError(f"Unsupported normalization preset: {normalize}. Supported: {supported}")
    stats = NORMALIZATION_STATS[normalize]
    return transforms.Normalize(mean=stats["mean"], std=stats["std"])


def build_transforms(
    image_size: int,
    train: bool,
    augment: str = "basic",
    normalize: str = "imagenet",
):
    interpolation = (
        InterpolationMode.BICUBIC
        if normalize == "clip"
        else InterpolationMode.BILINEAR
    )
    if train:
        if augment == "clip_probe":
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        image_size,
                        scale=(0.75, 1.0),
                        ratio=(0.9, 1.1),
                        interpolation=interpolation,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    _normalize_transform(normalize),
                ]
            )

        if augment == "strong":
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        image_size,
                        scale=(0.55, 1.0),
                        ratio=(0.8, 1.2),
                        interpolation=interpolation,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(
                        brightness=0.3,
                        contrast=0.3,
                        saturation=0.25,
                        hue=0.05,
                    ),
                    transforms.ToTensor(),
                    _normalize_transform(normalize),
                    transforms.RandomErasing(
                        p=0.25,
                        scale=(0.02, 0.15),
                        ratio=(0.3, 3.3),
                    ),
                ]
            )

        if augment == "art":
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        image_size,
                        scale=(0.6, 1.0),
                        ratio=(0.85, 1.15),
                        interpolation=interpolation,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.15,
                        hue=0.04,
                    ),
                    transforms.ToTensor(),
                    _normalize_transform(normalize),
                    transforms.RandomErasing(
                        p=0.2,
                        scale=(0.02, 0.12),
                        ratio=(0.3, 3.3),
                    ),
                ]
            )

        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                _normalize_transform(normalize),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(
                int(image_size * 1.14),
                interpolation=interpolation,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            _normalize_transform(normalize),
        ]
    )


def build_imagefolder_dataset(
    data_dir: str | Path,
    image_size: int,
    train: bool,
    normalize: str = "imagenet",
):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
    return datasets.ImageFolder(
        root=str(data_dir),
        transform=build_transforms(image_size=image_size, train=train, normalize=normalize),
    )


def build_dataloaders(
    train_dir: str | Path,
    val_dir: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    augment: str = "basic",
    normalize: str = "imagenet",
    train_fraction: float = 1.0,
    subset_seed: int = 42,
):
    train_dataset = datasets.ImageFolder(
        root=str(Path(train_dir)),
        transform=build_transforms(
            image_size=image_size,
            train=True,
            augment=augment,
            normalize=normalize,
        ),
    )
    val_dataset = build_imagefolder_dataset(
        val_dir,
        image_size,
        train=False,
        normalize=normalize,
    )

    if train_dataset.classes != val_dataset.classes:
        raise ValueError(
            "Train and validation class folders must match. "
            f"train={train_dataset.classes}, val={val_dataset.classes}"
        )

    train_classes = train_dataset.classes
    if not 0.0 < train_fraction <= 1.0:
        raise ValueError("train_fraction must be in (0, 1].")
    if train_fraction < 1.0:
        generator = torch.Generator().manual_seed(subset_seed)
        indices: list[int] = []
        targets = torch.tensor(train_dataset.targets)
        for class_index in range(len(train_classes)):
            class_indices = torch.where(targets == class_index)[0]
            keep = max(1, round(len(class_indices) * train_fraction))
            permutation = torch.randperm(len(class_indices), generator=generator)
            indices.extend(class_indices[permutation[:keep]].tolist())
        train_dataset = Subset(train_dataset, sorted(indices))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, train_classes
