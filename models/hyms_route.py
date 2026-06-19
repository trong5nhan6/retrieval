"""
HyMS-Route — full model.

  imgs ─► HybridEncoder ─► (ViT tokens, CNN maps)
        ─► project + TokenLearner + scale-embed ─► token set X [B, m, d]
        ─► Soft MoE ─► (Y [B,m,d], combine C [B,m,S])
        ─► z = pool(Y)            (retrieval embedding, L2)
        ─► rho = head(mean_t C)   (routing fingerprint, L2)

forward(imgs) -> (z, rho, combine)
  z       : [B, embed_dim]   L2-normalized retrieval embedding
  rho     : [B, route_dim]   L2-normalized routing descriptor
  combine : [B, m, S]        soft-routing weights (for analysis; optional)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import HCFG
from models.hybrid_encoder import HybridEncoder
from models.token_learner import TokenLearner
from models.softmoe import SoftMoE


class AttnPool(nn.Module):
    """Single learnable query attends over tokens -> one pooled vector."""
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.scale = dim ** -0.5

    def forward(self, Y):                       # Y: [B, m, d]
        B = Y.size(0)
        q = self.q.unsqueeze(0).expand(B, -1, -1)        # [B,1,d]
        attn = (q @ Y.transpose(1, 2)) * self.scale      # [B,1,m]
        attn = attn.softmax(dim=-1)
        return (attn @ Y).squeeze(1)                      # [B,d]


class HyMSRoute(nn.Module):
    def __init__(self, encoder: HybridEncoder, cfg=HCFG):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        d = cfg.token_dim
        self.cnn_stages = cfg.cnn_stages

        # ── ViT projection (768 -> d) ────────────────────────────────────
        self.vit_proj = nn.Linear(encoder.vit_dim, d)

        # ── per CNN stage: 1x1 conv (C_s -> d) + TokenLearner ────────────
        self.cnn_proj = nn.ModuleList()
        self.token_learners = nn.ModuleList()
        for s in cfg.cnn_stages:
            c_in = encoder.cnn_stage_dims[s]
            self.cnn_proj.append(nn.Conv2d(c_in, d, kernel_size=1))
            self.token_learners.append(
                TokenLearner(d, num_tokens=cfg.tokens_per_stage, num_heads=cfg.tl_heads))

        # ── scale / branch embeddings: 1 for ViT + 1 per CNN stage ───────
        n_sources = 1 + len(cfg.cnn_stages)
        self.scale_embed = nn.Parameter(torch.randn(n_sources, d) * 0.02)

        self.input_drop = nn.Dropout(cfg.dropout)

        # ── Soft MoE ──────────────────────────────────────────────────────
        self.softmoe = SoftMoE(d, n_experts=cfg.n_experts,
                               slots_per_expert=cfg.slots_per_expert,
                               hidden=cfg.expert_hidden)

        # ── heads ─────────────────────────────────────────────────────────
        self.pool = AttnPool(d)
        self.embed_proj = nn.Sequential(
            nn.Linear(d, cfg.embed_dim), nn.LayerNorm(cfg.embed_dim))
        self.route_head = nn.Sequential(
            nn.Linear(cfg.num_slots, cfg.route_dim), nn.LayerNorm(cfg.route_dim))

    def _assemble_tokens(self, vit_tokens, cnn_maps):
        d_emb = self.scale_embed
        toks = []

        # ViT branch
        v = self.vit_proj(vit_tokens) + d_emb[0]              # [B, P, d]
        toks.append(v)

        # CNN branches (only the configured stages)
        for j, s in enumerate(self.cnn_stages):
            m = self.cnn_proj[j](cnn_maps[s])                 # [B, d, H, W]
            t = self.token_learners[j](m)                     # [B, T, d]
            t = t + d_emb[1 + j]
            toks.append(t)

        return torch.cat(toks, dim=1)                          # [B, m, d]

    def forward(self, imgs):
        vit_tokens, cnn_maps = self.encoder(imgs)
        X = self._assemble_tokens(vit_tokens, cnn_maps)        # [B, m, d]
        if self.cfg.feat_noise > 0 and self.training:
            X = X + self.cfg.feat_noise * torch.randn_like(X)
        X = self.input_drop(X)

        Y, combine = self.softmoe(X)                           # [B,m,d], [B,m,S]

        z = F.normalize(self.embed_proj(self.pool(Y)), dim=-1)  # [B, embed_dim]

        usage = combine.mean(dim=1)                            # [B, S] slot-usage
        rho = F.normalize(self.route_head(usage), dim=-1)      # [B, route_dim]

        return z, rho, combine

    def head_parameters(self):
        """All trainable params except the (frozen) backbones."""
        backbone_ids = set(id(p) for p in self.encoder.parameters())
        return [p for p in self.parameters() if id(p) not in backbone_ids]
