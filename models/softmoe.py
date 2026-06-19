"""
Soft Mixture-of-Experts layer (Puigcerver et al., ICLR 2024).

Operates over a token set X [B, m, d]. Each of the S = n_experts * slots_per_expert
slots receives a SOFT (softmax-weighted) combination of all input tokens; each
expert processes its slots; outputs are recombined back into m output tokens via
a second softmax. Fully differentiable, no token dropping, no expert collapse,
no load-balance loss.

forward(X) -> (Y, C)
  Y : [B, m, d]      fused output tokens
  C : [B, m, S]      combine weights (per token, distribution over slots)
                     -> aggregated into the routing fingerprint rho downstream.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Expert(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):                # x: [B, p, d]
        return self.net(x)


class SoftMoE(nn.Module):
    def __init__(self, dim: int, n_experts: int = 8,
                 slots_per_expert: int = 4, hidden: int = 512):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.slots_per_expert = slots_per_expert
        self.num_slots = n_experts * slots_per_expert

        # slot parameter Phi: [d, S]
        self.phi = nn.Parameter(torch.randn(dim, self.num_slots) * (dim ** -0.5))
        self.experts = nn.ModuleList(
            [_Expert(dim, hidden) for _ in range(n_experts)])

    def forward(self, X: torch.Tensor):
        # X: [B, m, d]
        B, m, d = X.shape
        logits = torch.einsum('bmd,ds->bms', X, self.phi)   # [B, m, S]

        # Dispatch: softmax over tokens (each slot = distribution over tokens)
        dispatch = logits.softmax(dim=1)                     # [B, m, S]
        slots = torch.einsum('bms,bmd->bsd', dispatch, X)    # [B, S, d]

        # Experts process their own slots
        slots = slots.view(B, self.n_experts, self.slots_per_expert, d)
        outs = [self.experts[e](slots[:, e]) for e in range(self.n_experts)]
        expert_out = torch.stack(outs, dim=1).reshape(B, self.num_slots, d)  # [B, S, d]

        # Combine: softmax over slots (each token = distribution over slots)
        combine = logits.softmax(dim=2)                      # [B, m, S]
        Y = torch.einsum('bms,bsd->bmd', combine, expert_out)  # [B, m, d]

        return Y, combine
