"""
Configuration for HyMS-Route — Hybrid Multi-Scale Soft-MoE Retrieval
with Routing-aware Reranking.

Independent of the MedMNIST `config.py` (CFG) so the two subsystems never
clash. Import as:  from config import HCFG
"""
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class HyMSConfig:
    # ── Backbones ─────────────────────────────────────────────────────────
    vit_name:    str = "facebook/dinov2-large"      # HF DINOv2 ViT-B/14, 768-d
    cnn_name:    str = "convnext_base"             # torchvision CNN
    image_size:  int = 224

    # CNN stages to use (indices into the 4 ConvNeXt stages: 0=s1 .. 3=s4)
    # Default: all 4 stages. Ablate by e.g. [1, 2, 3] (drop the noisy s1).
    cnn_stages:  List[int] = field(default_factory=lambda: [0, 1, 2, 3])

    # ── Token assembly ────────────────────────────────────────────────────
    token_dim:        int = 768     # common token dim d (both branches projected to this)
    tokens_per_stage: int = 64      # LocalTokenizer tokens/stage (PHẢI là số chính phương: lưới g×g)
    tl_heads:         int = 4       # (deprecated) cũ dùng cho cross-attention TokenLearner; LocalTokenizer bỏ qua
    # ViT contributes its 256 patch tokens; CNN contributes len(cnn_stages)*tokens_per_stage.

    # ── Ablation switches ─────────────────────────────────────────────────
    # Master on/off for each component. Defaults all True => pipeline behaves
    # exactly like before. use_moe=False bypasses Soft MoE (Y=X), which also
    # disables rho/routerank/route-loss automatically (see train/eval).
    use_vit:  bool = True
    use_cnn:  bool = True
    use_moe:  bool = True

    # ── Soft MoE ──────────────────────────────────────────────────────────
    n_experts:        int = 8
    slots_per_expert: int = 2       # total slots S = n_experts * slots_per_expert
    expert_hidden:    int = 512

    # ── Heads ─────────────────────────────────────────────────────────────
    embed_dim:   int = 512          # retrieval embedding z (was 256 bottleneck)
    route_dim:   int = 64           # routing descriptor rho
    dropout:     float = 0.1

    # ── Embedding fusion (cải tiến) ───────────────────────────────────────
    # z = L2( norm( cls_proj(CLS) + gate * local_proj(pool(MoE tokens)) ) )
    # CLS skip = "sàn" data-efficient (≈ baseline). gate khởi tạo 0 => lúc bắt
    # đầu z ≈ CLS baseline; nhánh MoE chỉ được học tới mức nó thực sự giúp.
    use_cls_skip:    bool  = True   # đường CLS -> embedding trực tiếp (floor)
    local_gate_init: float = 0.5    # γ khởi tạo cho nhánh local (LayerScale/ReZero)
    bnneck:          bool  = True   # BatchNorm1d trước L2 (False -> LayerNorm như cũ)

    # ── Loss ──────────────────────────────────────────────────────────────
    # Main embedding loss on z: "supcon" (Supervised Contrastive) or "triplet"
    # (TripletMarginLoss, optionally with semihard mining) to match baselines
    # that report "triploss". The routing loss on rho stays SupCon either way.
    # MỘT loss chính trên z, chọn bằng loss_type:
    #   triplet | supcon | ms | nsoftmax | proxynca | softtriple | proxyanchor | ccl
    loss_type:         str   = "triplet"
    temperature:       float = 0.07   # SupCon tau for z
    triplet_margin:    float = 0.1    # margin for TripletMarginLoss (+ its miner)
    triplet_miner:     bool  = True   # mine semihard triplets (TripletMarginMiner)
    # Multi-Similarity (pair-based) + miner
    ms_alpha:      float = 2.0
    ms_beta:       float = 50.0
    ms_base:       float = 0.5
    ms_miner_eps:  float = 0.1
    # Normalized Softmax (classification proxy)
    nsoftmax_temp: float = 0.05
    # ProxyNCA++
    proxynca_scale: float = 3.0
    # SoftTriple (đa-center/lớp)
    softtriple_centers: int   = 2
    softtriple_la:      float = 20.0
    softtriple_gamma:   float = 0.1
    softtriple_margin:  float = 0.01
    # Center Contrastive Loss (CCL, arXiv 2308.00458)
    ccl_temp:   float = 0.1
    ccl_margin: float = 0.0
    # Potential Field (PFML, arXiv 2405.18560) — faithful reimplementation
    # Paper tunes: alpha∈[3,6] (best), delta∈[0.1,0.3], M=15 (CUB/Cars), M=2 (SOP)
    pf_alpha:             float = 4.0   # power-law exponent α (0=no decay → constant field)
    pf_delta:             float = 0.2   # flat-zone radius δ (margin-like: [0.1, 0.3])
    pf_lambda_rep:        float = 1.0   # weight of repulsion term λ_rep
    pf_proxies_per_class: int   = 15    # M proxies/class (M=15 CUB/Cars, M=2 SOP)
    pf_use_proxies:       bool  = True
    # Per-dataset override for proxies_per_class (paper: SOP uses M=2, CUB/Cars M=15)
    pf_ppc_per_dataset: Dict[str, int] = field(default_factory=lambda: {
        "sop":    2,
        "inshop": 2,
    })
    # LR riêng cho loss có THAM SỐ HỌC ĐƯỢC (proxy/center):
    #   nsoftmax | proxynca | softtriple | proxyanchor | ccl  (≫ head_lr)
    loss_lr:    float = 1e-2
    route_temperature: float = 0.1    # SupCon tau for rho (chỉ dùng nếu lambda_route>0)
    lambda_route:      float = 0.1    # weight routing-consistency loss (0 = TẮT, representation-first)

    # ── Proxy-Anchor (chạy SONG SONG với loss chính trên z) ───────────────
    # Khi bật, tổng loss = embed_loss(z) + lambda_proxy * ProxyAnchor(z)
    #                      + lambda_route * route(rho).
    # ProxyAnchor giữ một proxy học được cho MỖI lớp train (num_classes proxy),
    # nên các proxy được đưa vào optimizer với LR riêng (proxy_lr, thường lớn
    # hơn head_lr nhiều — theo paper gốc). Lớp test không cần proxy (zero-shot).
    use_proxy_anchor: bool  = False   # bật để cộng Proxy-Anchor vào loss
    lambda_proxy:     float = 1.0     # trọng số Proxy-Anchor trong tổng loss
    pa_margin:        float = 0.1     # δ (margin) của Proxy-Anchor
    pa_alpha:         float = 32.0    # α (scale/sharpening) của Proxy-Anchor
    proxy_lr:         float = 1e-2    # LR riêng cho các proxy (≫ head_lr)

    # ── RouteRank (inference) ─────────────────────────────────────────────
    # Defaults tuned for DENSE small-class sets (CUB/Cars: ~30–60 imgs/class).
    rr_beta:    float = 0.3      # weight of routing channel in fused score
    rr_topk:    int   = 10       # neighbours for test-time re-routing
    rr_alpha:   float = 3.0      # sharpening exponent
    rr_reroute: bool  = True
    # Bật/tắt RouteRank khi EVAL, độc lập với use_moe. Mặc định TẮT (chỉ đo base
    # embedding, đúng hướng representation-first). Bật lại bằng --eval_routerank.
    eval_routerank: bool = False
    # Per-dataset RouteRank overrides. SOP/In-Shop are SPARSE many-class sets
    # (~5 imgs/class): a large rr_topk drags negatives into query-expansion and
    # the coarse 16-slot rho channel (rr_beta) is mostly noise, so both are
    # reduced here. Empty dict for a dataset => use the global values above.
    rr_per_dataset: Dict[str, Dict] = field(default_factory=lambda: {
        "sop":    {"rr_topk": 4, "rr_beta": 0.1},
        "inshop": {"rr_topk": 4, "rr_beta": 0.1},
    })

    def rr_for(self, dataset: str) -> Dict:
        """RouteRank field overrides for a dataset (empty => use global rr_*)."""
        return self.rr_per_dataset.get(dataset.lower(), {})

    # ── Training ──────────────────────────────────────────────────────────
    batch_size:    int   = 128     # class-balanced sampler (see data loader)
    epochs:        int   = 10
    frozen_epochs: int   = 2       # Stage-1 warmup (backbones frozen)
    finetune_blocks: int = 4       # ViT blocks unfrozen in Stage-2 (0 = keep frozen)
    finetune_cnn_stages: int = 2   # ConvNeXt stages unfrozen in Stage-2 (0 = keep frozen, max 4)
    head_lr:       float = 1e-4
    backbone_lr:   float = 1e-5
    weight_decay:  float = 1e-4
    # LR schedule: "cosine" decays LR to 0 over the run (preserves head:backbone
    # ratio); "constant" keeps it flat (old behaviour). warmup_epochs applies a
    # linear warmup at the very start of Stage 1.
    lr_schedule:   str   = "cosine"   # "cosine" | "constant"
    warmup_epochs: int   = 0
    feat_noise:    float = 0.0     # optional gaussian noise on tokens (0 = off)
    grad_clip:     float = 1.0

    # class-balanced sampler: m images per class per batch
    classes_per_batch:    int = 20
    samples_per_class:    int = 6   # batch_size ~= classes_per_batch * samples_per_class

    # ── Eval / logging ────────────────────────────────────────────────────
    eval_every: int = 2     # run test every N train epochs (also always at last epoch)
    log_dir:    str = "results/logs"
    # Default K (CUB/Cars). Per-dataset K below follows each benchmark's SOTA tables.
    recall_k: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    recall_k_per_dataset: Dict[str, List[int]] = field(default_factory=lambda: {
        "cub":    [1, 2, 4, 8],
        "cars":   [1, 2, 4, 8],
        "sop":    [1, 10, 100, 1000],
        "inshop": [1, 10, 20, 30],
    })

    def recall_k_for(self, dataset: str) -> List[int]:
        return self.recall_k_per_dataset.get(dataset.lower(), self.recall_k)

    # ── Repro / paths ─────────────────────────────────────────────────────
    seed:           int = 42
    checkpoint_dir: str = "results/checkpoints"
    results_dir:    str = "results"

    # ── Dataset roots (override per environment / Kaggle) ────────────────
    data_roots: Dict[str, str] = field(default_factory=lambda: {
        "cub":    "datasets/CUB_200_2011",
        "cars":   "datasets/Cars196",
        "inshop": "datasets/In-shop Clothes Retrieval Benchmark",
        "sop":    "datasets/Stanford_Online_Products",
    })

    @property
    def num_slots(self) -> int:
        return self.n_experts * self.slots_per_expert


HCFG = HyMSConfig()
