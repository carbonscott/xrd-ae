import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LatentDiversityLoss(nn.Module):
    def __init__(self, min_distance=0.1):
        super().__init__()
        self.min_distance = min_distance

    def forward(self, z):
        batch_size = z.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=z.device)

        # Normalize latent vectors
        z_normalized = F.normalize(z, p=2, dim=1)

        # Compute cosine similarity
        similarity = torch.mm(z_normalized, z_normalized.t())
        similarity = torch.clamp(similarity, -1.0, 1.0)

        # Mask out diagonal
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=z.device)
        similarity = similarity[mask].view(batch_size, -1)

        # Convert to distance and compute loss
        distance = 1.0 - similarity
        loss = F.relu(self.min_distance - distance).mean()

        return loss

class AdaptiveWeightedLoss(nn.Module):
    def __init__(self, kernel_size=15, weight_factor=2.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight_factor = weight_factor

    def compute_local_contrast(self, x):
        # Compute local mean using average pooling
        padding = self.kernel_size // 2
        local_mean = F.avg_pool2d(
            F.pad(x, (padding, padding, padding, padding), mode='reflect'),
            self.kernel_size,
            stride=1
        )

        # Compute local standard deviation
        local_var = F.avg_pool2d(
            F.pad((x - local_mean)**2, (padding, padding, padding, padding), mode='reflect'),
            self.kernel_size,
            stride=1
        )
        local_std = torch.sqrt(local_var + 1e-6)

        # Normalize to create weight map
        weight_map = 1.0 + self.weight_factor * (local_std / local_std.mean())
        return weight_map

    def forward(self, pred, target):
        # Compute base L1 loss
        base_loss = torch.abs(pred - target)

        # Compute weight map based on local contrast of target
        weight_map = self.compute_local_contrast(target)

        # Apply weights to loss
        weighted_loss = base_loss * weight_map

        return weighted_loss.mean()

class TotalLoss(nn.Module):
    def __init__(self, kernel_size, weight_factor, min_distance, div_weight):
        super().__init__()
        self.adaptive_criterion  = AdaptiveWeightedLoss(kernel_size, weight_factor)
        self.diversity_criterion = LatentDiversityLoss(min_distance)
        self.div_weight = div_weight

    def forward(self, batch, latent, batch_logits):
        rec_loss = self.adaptive_criterion(batch_logits, batch)
        div_loss = self.diversity_criterion(latent)
        total_loss = rec_loss + self.div_weight * div_loss
        return total_loss
