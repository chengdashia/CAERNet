from pathlib import Path

from torch.utils.data import DataLoader
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
    if train:
        if augment == "strong":
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        image_size,
                        scale=(0.55, 1.0),
                        ratio=(0.8, 1.2),
                        interpolation=InterpolationMode.BILINEAR,
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
                        interpolation=InterpolationMode.BILINEAR,
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
                interpolation=InterpolationMode.BILINEAR,
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
    return train_loader, val_loader, train_dataset.classes
