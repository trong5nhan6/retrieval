"""
HyMS-Route — full model.

  imgs ─► HybridEncoder ─► (ViT tokens, CNN maps)
        ─► project + TokenLearner + scale-embed ─► token set X [B, m, d]

moe_target = "cls"  (mặc định — Hướng A)
  CLS được NỐI VÀO tập token: X' = [vit_proj(CLS)+cls_type ; X]  [B, 1+m, d]
  Soft MoE định tuyến bằng chứng đa tỉ lệ *vào* CLS, phần tinh chỉnh là
    delta = Y'[:,0] - X'[:,0] = gamma_moe * MoE(LN(X'))[:,0]
  và cộng residual (bias=False) vào embedding:
    z = L2( BN( cls_proj(CLS) + moe_out_proj(delta) [+ local_gate*local] ) )
  gamma_moe = 0  =>  delta = 0  =>  z BẰNG ĐÚNG baseline CLS. Ablation lồng nhau.

moe_target = "tokens"  (hành vi cũ)
  Soft MoE xử lý X rồi kết quả đi vào nhánh local qua pool. Đo được: nhánh này
  đóng góp âm (xem config.use_local_branch), nên giữ để lấy bảng ablation.

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
from models.token_learner import LocalTokenizer
from models.softmoe import SoftMoE


class GeMPool(nn.Module):
    """Generalized-mean pooling trên trục token, p học được.

    p=1 là mean, p->inf là max. Lý do thay AttnPool: đo trên checkpoint cho thấy
    AttnPool phân bổ 0.00275/token (ViT) và 0.00231/token (CNN) so với uniform
    0.0026 — nó không chọn lọc gì. Zero-shot trên DINOv2-L: mean 35.11, GeM p=3
    42.08, GeM p=5 50.47, max 68.33.
    """
    def __init__(self, dim, p_init: float = 3.0):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))

    def forward(self, Y):                       # Y: [B, m, d]
        p = self.p.clamp(min=1.0, max=10.0)
        return Y.clamp(min=1e-6).pow(p).mean(dim=1).pow(1.0 / p)   # [B, d]


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

        # Effective component switches (encoder may have been built without a
        # branch, in which case force it off here too).
        self.use_vit = cfg.use_vit and encoder.vit is not None
        self.use_cnn = cfg.use_cnn and encoder.cnn is not None
        self.use_moe = cfg.use_moe
        self.cnn_stages = cfg.cnn_stages if self.use_cnn else []
        if not (self.use_vit or self.use_cnn):
            raise ValueError("HyMSRoute needs at least one of use_vit/use_cnn.")

        # Nơi đặt MoE. "cls" cần CLS của ViT, nên tự hạ về "tokens" nếu tắt ViT.
        self.moe_target = getattr(cfg, "moe_target", "tokens")
        if self.moe_target not in ("cls", "tokens"):
            raise ValueError(f"moe_target phải là 'cls' hoặc 'tokens', nhận {self.moe_target!r}")
        if self.moe_target == "cls" and not (self.use_vit and cfg.use_cls_skip):
            self.moe_target = "tokens"

        # ── ViT projection (768 -> d) ────────────────────────────────────
        self.vit_proj = nn.Linear(encoder.vit_dim, d) if self.use_vit else None

        # ── per CNN stage: 1x1 conv (C_s -> d) + LocalTokenizer ──────────
        # LocalTokenizer giữ tính cục bộ: mỗi token = một vùng lưới g×g của
        # feature map (không cross-attention/pool toàn cục như trước).
        self.cnn_proj = nn.ModuleList()
        self.token_learners = nn.ModuleList()
        for s in self.cnn_stages:
            c_in = encoder.cnn_stage_dims[s]
            self.cnn_proj.append(nn.Conv2d(c_in, d, kernel_size=1))
            self.token_learners.append(
                LocalTokenizer(d, num_tokens=cfg.tokens_per_stage))

        # ── scale / branch embeddings: 1 for ViT (if on) + 1 per CNN stage ─
        n_sources = (1 if self.use_vit else 0) + len(self.cnn_stages)
        self.scale_embed = nn.Parameter(torch.randn(n_sources, d) * 0.02)

        self.input_drop = nn.Dropout(cfg.dropout)

        # ── Soft MoE (optional) ───────────────────────────────────────────
        self.softmoe = SoftMoE(d, n_experts=cfg.n_experts,
                               slots_per_expert=cfg.slots_per_expert,
                               hidden=cfg.expert_hidden,
                               gate_init=getattr(cfg, "moe_gate_init", 0.1)
                               ) if self.use_moe else None
        # Hướng A: CLS tham gia tập token (một "nguồn" riêng) và phần MoE tinh
        # chỉnh nó được cộng residual vào embedding. bias=False là BẮT BUỘC —
        # có bias thì gamma_moe=0 vẫn cộng một hằng số, mất tính lồng nhau.
        self.moe_on_cls = self.use_moe and self.moe_target == "cls"
        if self.moe_on_cls:
            self.cls_type = nn.Parameter(torch.randn(d) * 0.02)
            # W giữ init MẶC ĐỊNH của Linear; gamma_moe là gate DUY NHẤT điều
            # khiển biên độ. Đã thử W init std=1e-3 và hỏng theo hai đường: đóng
            # góp tụt còn 0.05% của ||cls_proj(CLS)|| (MoE vô hình ở init), và
            # grad(gamma) ~ 3e-10 << eps=1e-8 của Adam nên bước bị bóp còn ~3%.
            # Nhân hai hệ số nhỏ với nhau là sai; chỉ một hệ số được nhỏ.
            self.moe_out_proj = nn.Linear(d, cfg.embed_dim, bias=False)

        # ── heads ─────────────────────────────────────────────────────────
        # global skip branch: CLS -> embed_dim (the data-efficient "floor").
        self.use_cls_skip = cfg.use_cls_skip and self.use_vit
        # nhánh local = pool(token) -> local_proj. Không có cls_skip thì nó là
        # đường DUY NHẤT ra embedding, nên bắt buộc bật.
        self.use_local_branch = getattr(cfg, "use_local_branch", True) or not self.use_cls_skip
        if not (self.use_cls_skip or self.use_local_branch):
            raise ValueError("HyMSRoute cần ít nhất một trong use_cls_skip/use_local_branch.")

        if self.use_local_branch:
            pool_type = getattr(cfg, "pool_type", "attn")
            self.pool = (GeMPool(d, getattr(cfg, "gem_p_init", 3.0))
                         if pool_type == "gem" else AttnPool(d))
            self.local_proj = nn.Linear(d, cfg.embed_dim)
        else:
            self.pool = None
            self.local_proj = None

        if self.use_cls_skip:
            self.cls_proj = nn.Linear(encoder.vit_dim, cfg.embed_dim)
            # gate trên nhánh local (LayerScale/ReZero).
            if self.use_local_branch:
                self.local_gate = nn.Parameter(torch.tensor(float(cfg.local_gate_init)))
        # final normalization before L2 (BNNeck, or LayerNorm fallback)
        self.embed_norm = (nn.BatchNorm1d(cfg.embed_dim) if cfg.bnneck
                           else nn.LayerNorm(cfg.embed_dim))
        # route_head only meaningful when Soft MoE produces combine weights.
        self.route_head = nn.Sequential(
            nn.Linear(cfg.num_slots, cfg.route_dim), nn.LayerNorm(cfg.route_dim)
        ) if self.use_moe else None

    def _assemble_tokens(self, vit_tokens, cnn_maps):
        d_emb = self.scale_embed
        toks = []
        src = 0   # running index into scale_embed (ViT first if present, then CNN)

        # ViT branch (optional)
        if self.use_vit:
            v = self.vit_proj(vit_tokens) + d_emb[src]         # [B, P, d]
            toks.append(v)
            src += 1

        # CNN branches (only the configured stages)
        for j, s in enumerate(self.cnn_stages):
            m = self.cnn_proj[j](cnn_maps[s])                 # [B, d, H, W]
            t = self.token_learners[j](m)                     # [B, T, d]
            t = t + d_emb[src]
            toks.append(t)
            src += 1

        return torch.cat(toks, dim=1)                          # [B, m, d]

    def forward(self, imgs):
        vit_tokens, cls, cnn_maps = self.encoder(imgs)
        X = self._assemble_tokens(vit_tokens, cnn_maps)        # [B, m, d]
        if self.cfg.feat_noise > 0 and self.training:
            X = X + self.cfg.feat_noise * torch.randn_like(X)
        X = self.input_drop(X)

        combine, cls_delta = None, None
        if self.moe_on_cls:
            # CLS tham gia tập token => MoE định tuyến bằng chứng đa tỉ lệ VÀO nó.
            cls_in = (self.vit_proj(cls) + self.cls_type).unsqueeze(1)   # [B,1,d]
            Xc = torch.cat([cls_in, X], dim=1)                 # [B, 1+m, d]
            Yc, combine = self.softmoe(Xc)
            cls_delta = Yc[:, 0] - cls_in[:, 0]                # = gamma_moe * MoE(...)[0]
            Y = Yc[:, 1:]                                      # token cho nhánh local
        elif self.use_moe:
            Y, combine = self.softmoe(X)                       # MoE trên token (cách cũ)
        else:
            Y = X                                              # bypass: pool raw tokens

        # ── fusion ────────────────────────────────────────────────────────
        if self.use_cls_skip:
            fused = self.cls_proj(cls)
            if self.use_local_branch:
                fused = fused + self.local_gate * self.local_proj(self.pool(Y))
            if cls_delta is not None:
                # bias=False => gamma_moe=0 cho đóng góp ĐÚNG BẰNG 0, tức nhánh
                # "+MoE" là mở rộng lồng nhau của baseline CLS.
                fused = fused + self.moe_out_proj(cls_delta)
        else:
            fused = self.local_proj(self.pool(Y))
        z = F.normalize(self.embed_norm(fused), dim=-1)        # [B, embed_dim]

        if combine is not None:
            usage = combine.mean(dim=1)                        # [B, S] slot-usage
            rho = F.normalize(self.route_head(usage), dim=-1)  # [B, route_dim]
        else:
            rho = None                                         # no routing fingerprint

        return z, rho, combine

    def gate_parameters(self):
        """Scalar gate (gamma_moe, GeM p) — cần LR RIÊNG.

        Đo trên local_gate: ở lr=1e-4 nó chỉ đi 0.5003 -> 0.5214 sau 10 epoch,
        tức trần dịch chuyển của Adam (~lr * n_step) quá nhỏ để một scalar gate
        học được gì. Nhóm này chạy ở cfg.moe_gate_lr và không chịu weight decay
        (weight decay kéo gate về 0 = tắt nhánh).
        """
        ps = []
        if self.softmoe is not None:
            ps.append(self.softmoe.gamma)
        if isinstance(self.pool, GeMPool):
            ps.append(self.pool.p)
        if getattr(self, "local_gate", None) is not None:
            ps.append(self.local_gate)
        return ps

    def head_parameters(self):
        """All trainable params except the (frozen) backbones and the gates."""
        backbone_ids = set(id(p) for p in self.encoder.parameters())
        gate_ids = set(id(p) for p in self.gate_parameters())
        return [p for p in self.parameters()
                if id(p) not in backbone_ids and id(p) not in gate_ids]
