import torch
from torch import nn


def mixup_data(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply MixUp augmentation.

    Returns (mixed_images, targets_a, targets_b, lambda) where lambda is the
    mixing coefficient drawn from Beta(alpha, alpha).
    """
    if alpha <= 0:
        return images, targets, targets, 1.0

    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    lam = max(lam, 1.0 - lam)  # enforce lam >= 0.5 for symmetry

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1.0 - lam) * images[index]
    return mixed_images, targets, targets[index], lam


def mixup_cross_entropy(
    logits: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy loss for MixUp: lam * CE(logits, targets_a) + (1-lam) * CE(logits, targets_b)."""
    return lam * nn.functional.cross_entropy(
        logits, targets_a, label_smoothing=label_smoothing,
    ) + (1.0 - lam) * nn.functional.cross_entropy(
        logits, targets_b, label_smoothing=label_smoothing,
    )


def energy_score(logits: torch.Tensor) -> torch.Tensor:
    return -torch.logsumexp(logits, dim=1)


def energy_regularization_loss(logits: torch.Tensor, margin: float = -5.0) -> torch.Tensor:
    energy = energy_score(logits)
    return torch.relu(energy - margin).pow(2).mean()


def energy_barrier_loss(
    id_logits: torch.Tensor,
    unknown_logits: torch.Tensor,
    id_margin: float = -5.0,
    unknown_margin: float = -1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Energy barrier for ID samples and pseudo-unknown samples.

    In-distribution samples are encouraged to have low energy, while synthetic
    unknown samples are encouraged to stay above a higher energy margin.
    """
    id_energy = energy_score(id_logits)
    unknown_energy = energy_score(unknown_logits)
    id_loss = torch.relu(id_energy - id_margin).pow(2).mean()
    unknown_loss = torch.relu(unknown_margin - unknown_energy).pow(2).mean()
    total = id_loss + unknown_loss
    return total, {
        "id_energy_loss": float(id_loss.detach().cpu()),
        "unknown_energy_loss": float(unknown_loss.detach().cpu()),
    }


def supervised_contrastive_loss(
    features: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    features = nn.functional.normalize(features, dim=1)
    logits = features @ features.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    batch_size = targets.size(0)
    self_mask = torch.eye(batch_size, device=targets.device, dtype=torch.bool)
    positive_mask = targets.unsqueeze(0).eq(targets.unsqueeze(1)) & ~self_mask

    exp_logits = torch.exp(logits) * ~self_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return features.new_tensor(0.0)

    mean_log_prob = (log_prob * positive_mask).sum(dim=1)[valid] / positive_count[valid]
    return -mean_log_prob.mean()


def classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    energy_lambda: float = 0.0,
    energy_margin: float = -5.0,
    label_smoothing: float = 0.0,
    features: torch.Tensor | None = None,
    contrastive_lambda: float = 0.0,
    contrastive_temperature: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    ce_loss = nn.functional.cross_entropy(
        logits,
        targets,
        label_smoothing=label_smoothing,
    )
    energy_loss = energy_regularization_loss(logits, margin=energy_margin)
    if features is not None and contrastive_lambda > 0.0:
        contrastive_loss = supervised_contrastive_loss(
            features,
            targets,
            temperature=contrastive_temperature,
        )
    else:
        contrastive_loss = logits.new_tensor(0.0)
    total_loss = ce_loss + energy_lambda * energy_loss + contrastive_lambda * contrastive_loss
    return total_loss, {
        "ce_loss": float(ce_loss.detach().cpu()),
        "energy_loss": float(energy_loss.detach().cpu()),
        "contrastive_loss": float(contrastive_loss.detach().cpu()),
    }
