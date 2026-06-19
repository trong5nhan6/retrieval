"""
TokenLearner (query-based / Perceiver-style).

Reduces a feature map of arbitrary spatial size to a FIXED number of summary
tokens. `num_tokens` learnable query vectors cross-attend over the flattened
feature map (keys/values), so each output token is a learned weighted pool of
pixels — keeps detail on large maps (s1) and avoids fake/interpolated tokens on
small maps (s4), unlike blind average pooling.

  in:  [B, C, H, W]   (a CNN stage feature map, already projected to d=C)
  out: [B, num_tokens, d]

Ref idea: TokenLearner (Ryoo et al., NeurIPS 2021); query-attention variant.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenLearner(nn.Module):
    def __init__(self, dim: int, num_tokens: int = 64, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_tokens = num_tokens
        self.num_heads  = num_heads
        self.head_dim   = dim // num_heads
        self.scale      = self.head_dim ** -0.5

        # learnable query tokens
        self.query = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat_map.shape
        x = feat_map.flatten(2).transpose(1, 2)          # [B, HW, C]
        N = x.size(1)

        q = self.query.unsqueeze(0).expand(B, -1, -1)    # [B, T, C]
        k = self.k_proj(x)                               # [B, HW, C]
        v = self.v_proj(x)                               # [B, HW, C]

        # split heads: [B, h, *, head_dim]
        def split(t, L):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        qh = split(q, self.num_tokens)                   # [B, h, T, hd]
        kh = split(k, N)                                 # [B, h, HW, hd]
        vh = split(v, N)                                 # [B, h, HW, hd]

        attn = (qh @ kh.transpose(-2, -1)) * self.scale  # [B, h, T, HW]
        attn = attn.softmax(dim=-1)                      # distribution over pixels
        out  = attn @ vh                                 # [B, h, T, hd]
        out  = out.transpose(1, 2).reshape(B, self.num_tokens, C)  # [B, T, C]
        out  = self.out_proj(out)
        return self.norm(out)                            # [B, num_tokens, dim]
