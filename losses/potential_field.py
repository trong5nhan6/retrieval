"""
Potential Field based Metric Learning (PFML) — Bhatnagar & Ahuja, CVPR 2025
(arXiv:2405.18560).

Paper equations (Section 3.2.1):

  Attraction potential (same-class source at zi, evaluated at r):
    ψ_att(r, zi) = -1/δ^α         if ||r - zi|| < δ   (flat zone: no force → no collapse)
                 = -1/||r-zi||^α   otherwise            (power-law decay: force decreases with distance)

  Repulsion potential (diff-class source at zi, evaluated at r):
    ψ_rep(r, zi) =  1/||r-zi||^α  if ||r - zi|| < δ   (push apart when too close)
                 =  1/δ^α          otherwise            (constant beyond δ: no extra force → no over-separation)

  α : power-law exponent (larger → faster decay). Paper tunes α ∈ {0..6}; best α ∈ [3,6].
  δ : flat-zone radius (like a margin). Paper tunes δ ∈ [0.1, 0.3].
      α=0 → no decay (constant field everywhere, reduces to a counting loss).

Class potential field for class j (superposition):
  Ψ_j(r) = Σ_{i: yi=j}  ψ_att(r, zi)   (attract from same-class samples + same-class proxies)
           + Σ_{i: yi≠j} ψ_rep(r, zi)   (repel from diff-class samples + diff-class proxies)

Total energy minimised by gradient descent:
  U = Σ_i Ψ_{yi}(zi)  +  Σ_j Σ_k Ψ_j(p_{j,k})

Proxies p_{j,k} are nn.Parameters; add them to the optimiser with a higher LR
(paper: 100× backbone LR). Embeddings are L2-normalised before computing distances.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PotentialFieldLoss(nn.Module):
    """
    Args:
        num_classes       : number of training classes
        embedding_size    : dimension of the L2-normalised embedding z
        proxies_per_class : M in the paper (15 for CUB/Cars, 2 for SOP)
        alpha             : power-law exponent (paper default α=4, best range [3,6])
        delta             : flat-zone radius / margin (paper default δ=0.2, range [0.1,0.3])
        lambda_rep        : weight of repulsion term (paper default 1.0)
        use_proxies       : False → sample-only field (no proxy augmentation)
    """
    def __init__(self, num_classes: int, embedding_size: int,
                 proxies_per_class: int = 15, alpha: float = 4.0,
                 delta: float = 0.2, lambda_rep: float = 1.0,
                 use_proxies: bool = True):
        super().__init__()
        self.alpha = alpha
        self.delta = delta
        self.lambda_rep = lambda_rep
        self.num_classes = num_classes
        self.ppc = proxies_per_class

        if use_proxies and proxies_per_class > 0:
            self.proxies = nn.Parameter(
                torch.empty(num_classes * proxies_per_class, embedding_size))
            nn.init.kaiming_uniform_(self.proxies, a=5 ** 0.5)
            self.register_buffer(
                "proxy_labels",
                torch.arange(num_classes).repeat_interleave(proxies_per_class))
        else:
            self.proxies = None

    # ── Paper Eq. (1): attraction potential ────────────────────────────────
    def _att_field(self, dist: torch.Tensor) -> torch.Tensor:
        """
        ψ_att at locations given by pairwise distances `dist`.
          inside δ  → -1/δ^α  (constant ⟹ gradient=0 ⟹ no force, prevents collapse)
          outside δ → -1/d^α  (power-law, gradient pulls same-class together)
        When α=0: field = -1 everywhere (special case, no decay).
        """
        if self.alpha == 0:
            return torch.full_like(dist, -1.0)
        floor_val = -1.0 / (self.delta ** self.alpha) if self.delta > 0 else -1.0
        outer = -1.0 / dist.clamp(min=1e-8).pow(self.alpha)
        return torch.where(dist < self.delta,
                           torch.full_like(dist, floor_val),
                           outer)

    # ── Paper Eq. (2): repulsion potential ─────────────────────────────────
    def _rep_field(self, dist: torch.Tensor) -> torch.Tensor:
        """
        ψ_rep at locations given by pairwise distances `dist`.
          inside δ  → 1/d^α   (power-law repulsion, pushes diff-class apart)
          outside δ → 1/δ^α   (constant ⟹ gradient=0 ⟹ no extra force beyond δ)
        When α=0: field = 1 everywhere (no decay).
        """
        if self.alpha == 0:
            return torch.ones_like(dist)
        ceiling_val = 1.0 / (self.delta ** self.alpha) if self.delta > 0 else 1.0
        inner = 1.0 / dist.clamp(min=1e-8).pow(self.alpha)
        return torch.where(dist < self.delta,
                           inner,
                           torch.full_like(dist, ceiling_val))

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor,
                indices_tuple=None) -> torch.Tensor:
        z = F.normalize(embeddings, dim=-1)   # [B, D]
        B = z.size(0)

        # Assemble field sources: batch embeddings + proxies (if enabled)
        pts, lbl = z, labels
        if self.proxies is not None:
            p = F.normalize(self.proxies, dim=-1)
            pts = torch.cat([z, p], dim=0)                         # [B+P, D]
            lbl = torch.cat([labels, self.proxy_labels.to(labels.device)], dim=0)

        # Pairwise Euclidean distances: anchors × all sources
        dist = torch.cdist(z, pts)                                  # [B, B+P]

        # ── same-class mask (attraction) ───────────────────────────────────
        same = labels.unsqueeze(1).eq(lbl.unsqueeze(0))             # [B, B+P]
        # Remove self-interaction: anchor i vs its own embedding (first B cols)
        eye = torch.eye(B, dtype=torch.bool, device=z.device)
        same_no_self = same.clone()
        same_no_self[:, :B] = same[:, :B] & ~eye

        # ── diff-class mask (repulsion) ────────────────────────────────────
        diff = ~labels.unsqueeze(1).eq(lbl.unsqueeze(0))            # [B, B+P]

        # ── Paper Eq. (1) & (2): per-pair potentials ───────────────────────
        att = self._att_field(dist)   # [B, B+P], attraction potential values
        rep = self._rep_field(dist)   # [B, B+P], repulsion potential values

        # ── Paper Eq. (6): total potential energy U per anchor ─────────────
        # Ψ_yi(zi) = Σ_same ψ_att  +  λ * Σ_diff ψ_rep
        # Mean over valid sources (avoids scale dependency on batch/proxy count)
        n_same = same_no_self.sum(1).clamp(min=1)
        n_diff = diff.sum(1).clamp(min=1)

        attract = (att * same_no_self).sum(1) / n_same    # negative → minimising pulls same close
        repel   = (rep * diff).sum(1)        / n_diff     # positive → minimising pushes diff apart

        # U = Σ_i [attract_i + λ_rep * repel_i], averaged over batch
        return (attract + self.lambda_rep * repel).mean()
