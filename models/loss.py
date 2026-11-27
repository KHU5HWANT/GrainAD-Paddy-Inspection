"""Supervised contrastive objective (paper Eq. 3)."""
import torch
from torch import nn
from torch.nn import functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        if temperature <= 0:
            raise ValueError('temperature must be positive')
        self.temperature = temperature

    def forward(self, features, labels):
        if features.ndim != 2 or features.shape[0] != labels.numel():
            raise ValueError('Expected features [N,D] and N labels')
        if len(features) < 2:
            return features.sum() * 0
        features = F.normalize(features, dim=1)
        labels = labels.reshape(-1).to(features.device)
        diagonal = torch.eye(len(features), device=features.device, dtype=torch.bool)
        positive = labels[:, None].eq(labels[None, :]) & ~diagonal
        count = positive.sum(1)
        valid = count > 0
        if not valid.any():
            return features.sum() * 0
        logits = features @ features.T / self.temperature
        denominator = torch.logsumexp(logits.masked_fill(diagonal, -torch.inf), dim=1)
        log_prob = logits - denominator[:, None]
        return -(log_prob.masked_fill(~positive, 0).sum(1)[valid] / count[valid]).mean()


def multiview_labels(batch_size, num_views, augmented_labels):
    """Kernel positives in view-major [normal, Gaussian, image-synthetic] order.

    Synthetic anomalies form a separate group per kernel; unaltered
    augmentations remain normal positives. This extends Eq. 3 to negatives.
    """
    augmented_labels = augmented_labels.reshape(-1)
    if augmented_labels.numel() != batch_size * num_views:
        raise ValueError('Augmentation labels do not match the view batch')
    ids = torch.arange(batch_size, device=augmented_labels.device).repeat(num_views)
    return torch.cat((ids, ids + batch_size,
                      ids + (augmented_labels > 0).long() * batch_size))
