"""
Center Contrastive Loss (CCL) — Cai, Xiong & Tian, 2023 (arXiv:2308.00458).

Idea: maintain a learnable class-wise CENTER BANK  C in R^{num_classes x dim}.
Every embedding is compared (cosine) against ALL centers; a temperature-scaled
softmax cross-entropy pulls it toward its own class center and pushes it away
from the others. This combines the strengths of contrastive (reduce intra-class
variance) and classification (enlarge inter-class margin) WITHOUT sample mining,
and the per-class centers re-balance the supervisory signal across classes —
exactly the regime where pair/triplet losses struggle (many classes, few
samples/class, e.g. SOP / In-Shop).

The centers are nn.Parameters, so they must be added to the optimizer with their
own LR (cfg.loss_lr), like proxy-based losses. Test classes need no centers
(zero-shot retrieval uses the embedding z directly).

Call signature matches pytorch_metric_learning: forward(embeddings, labels,
indices_tuple=None) so it drops into build_embed_loss like any other loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterContrastiveLoss(nn.Module):
    def __init__(self, num_classes: int, embedding_size: int,
                 temperature: float = 0.1, margin: float = 0.0):
        super().__init__()
        self.centers = nn.Parameter(torch.empty(num_classes, embedding_size))
        nn.init.kaiming_uniform_(self.centers, a=5 ** 0.5)
        self.temperature = temperature
        self.margin = margin
        self.num_classes = num_classes

    def forward(self, embeddings, labels, indices_tuple=None):
        # cosine logits between L2-normalized embeddings and centers
        z = F.normalize(embeddings, dim=-1)
        c = F.normalize(self.centers, dim=-1)
        logits = z @ c.t()                                   # [B, num_classes]
        if self.margin > 0:                                  # additive margin on true class
            onehot = F.one_hot(labels, self.num_classes).to(logits.dtype)
            logits = logits - self.margin * onehot
        return F.cross_entropy(logits / self.temperature, labels)
