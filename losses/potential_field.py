"""
Potential Field based Metric Learning (PFML) — Bhatnagar & Ahuja, CVPR 2025
(arXiv:2405.18560).

NOTE: This is a faithful RE-IMPLEMENTATION of the method as described in the
paper (the official equations are only available as MathML on the arXiv HTML).
For an exact reproduction, prefer the authors' released code. The essence is
captured here:

Idea (from the paper): treat every embedding as a charge that creates a
continuous potential field. Same-class points generate an ATTRACTION field,
different-class points a REPULSION field. Unlike most DML losses, the strength
of interaction DECAYS with distance (a far same-class sample is treated as a
different variety; a far negative barely repels). All pairwise interactions in
the batch are summed (superposition) — no tuple mining. Learnable per-class
PROXIES AUGMENT the field (they do not replace sample-sample interactions).

Decaying kernel:  K(a,b) = exp(-alpha * max(0, ||a-b|| - delta)^2)
  - alpha : decay rate (larger => interaction vanishes faster with distance)
  - delta : flat radius; interaction is ~constant within delta, decays outside

Total energy per anchor i (minimised):
  E_i = - mean_{j: same class} K(z_i, p_j)            (attraction: lower when close)
        + lambda_rep * mean_{k: diff class} K(z_i, p_k)   (repulsion: lower when far)
where {p_*} = batch embeddings (+ proxies). Minimising E pulls same-class
together and pushes different-class apart, with forces that decay with distance.

Proxies are nn.Parameters -> add to the optimizer with loss_lr (like other
proxy/center losses). Embeddings are L2-normalized to match the pipeline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PotentialFieldLoss(nn.Module):
    def __init__(self, num_classes: int, embedding_size: int,
                 proxies_per_class: int = 1, alpha: float = 4.0,
                 delta: float = 0.0, lambda_rep: float = 1.0,
                 use_proxies: bool = True):
        super().__init__()
        self.alpha = alpha
        self.delta = delta
        self.lambda_rep = lambda_rep
        self.num_classes = num_classes
        self.ppc = proxies_per_class
        if use_proxies:
            self.proxies = nn.Parameter(
                torch.empty(num_classes * proxies_per_class, embedding_size))
            nn.init.kaiming_uniform_(self.proxies, a=5 ** 0.5)
            self.register_buffer(
                "proxy_labels",
                torch.arange(num_classes).repeat_interleave(proxies_per_class))
        else:
            self.proxies = None

    def _field(self, dist):
        """Decaying interaction: flat within delta, Gaussian decay outside."""
        d = (dist - self.delta).clamp(min=0.0)
        return torch.exp(-self.alpha * d * d)

    def forward(self, embeddings, labels, indices_tuple=None):
        z = F.normalize(embeddings, dim=-1)                  # [B, D]
        B = z.size(0)

        pts, lbl = z, labels                                 # field sources
        if self.proxies is not None:
            p = F.normalize(self.proxies, dim=-1)
            pts = torch.cat([z, p], dim=0)                   # [B+P, D]
            lbl = torch.cat([labels, self.proxy_labels.to(labels.device)], dim=0)

        dist = torch.cdist(z, pts)                           # [B, B+P] euclidean
        K = self._field(dist)                                # decaying field

        same = labels.unsqueeze(1).eq(lbl.unsqueeze(0))      # [B, B+P]
        # remove self-interaction (anchor i vs its own embedding in the first B cols)
        self_mask = torch.zeros_like(same)
        self_mask[:, :B] = torch.eye(B, dtype=torch.bool, device=z.device)
        same = same & ~self_mask
        diff = ~labels.unsqueeze(1).eq(lbl.unsqueeze(0))

        attract = -(K * same).sum(1) / same.sum(1).clamp(min=1)   # pull same-class
        repel = (K * diff).sum(1) / diff.sum(1).clamp(min=1)      # push diff-class
        return (attract + self.lambda_rep * repel).mean()
