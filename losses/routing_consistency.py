"""
Routing-consistency loss.

Supervised-contrastive loss applied on the routing descriptor `rho`, using the
SAME class labels as the main embedding loss. Pulls same-class routing
fingerprints together, pushes different-class apart — making `rho` a
class-discriminative signal that RouteRank can fuse with the embedding cosine.

Reuses pytorch_metric_learning.SupConLoss. `rho` is already L2-normalized.
"""
import torch
import torch.nn as nn
from pytorch_metric_learning.losses import SupConLoss


class RoutingConsistencyLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.supcon = SupConLoss(temperature=temperature)

    def forward(self, rho: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # rho: [B, route_dim] (L2-normalized), labels: [B]
        # SupConLoss needs >=1 positive pair; guard tiny/degenerate batches.
        if labels.unique().numel() == labels.numel():
            # no positives at all -> contrastive term undefined; return 0
            return rho.sum() * 0.0
        return self.supcon(rho, labels)
