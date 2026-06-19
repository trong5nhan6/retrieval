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
    vit_name:    str = "facebook/dinov2-base"      # HF DINOv2 ViT-B/14, 768-d
    cnn_name:    str = "convnext_tiny"             # torchvision CNN
    image_size:  int = 224

    # CNN stages to use (indices into the 4 ConvNeXt stages: 0=s1 .. 3=s4)
    # Default: all 4 stages. Ablate by e.g. [1, 2, 3] (drop the noisy s1).
    cnn_stages:  List[int] = field(default_factory=lambda: [0, 1, 2, 3])

    # ── Token assembly ────────────────────────────────────────────────────
    token_dim:        int = 256     # common token dim d (both branches projected to this)
    tokens_per_stage: int = 64      # TokenLearner output tokens per CNN stage
    tl_heads:         int = 4       # TokenLearner attention heads
    # ViT contributes its 256 patch tokens; CNN contributes len(cnn_stages)*tokens_per_stage.

    # ── Soft MoE ──────────────────────────────────────────────────────────
    n_experts:        int = 8
    slots_per_expert: int = 4       # total slots S = n_experts * slots_per_expert
    expert_hidden:    int = 512

    # ── Heads ─────────────────────────────────────────────────────────────
    embed_dim:   int = 256          # retrieval embedding z
    route_dim:   int = 64           # routing descriptor rho
    dropout:     float = 0.1

    # ── Loss ──────────────────────────────────────────────────────────────
    temperature:       float = 0.07   # SupCon tau for z
    route_temperature: float = 0.1    # SupCon tau for rho
    lambda_route:      float = 0.5    # weight of routing-consistency loss

    # ── RouteRank (inference) ─────────────────────────────────────────────
    rr_beta:    float = 0.3      # weight of routing channel in fused score
    rr_topk:    int   = 10       # neighbours for test-time re-routing
    rr_alpha:   float = 3.0      # sharpening exponent
    rr_reroute: bool  = True

    # ── Training ──────────────────────────────────────────────────────────
    batch_size:    int   = 128     # class-balanced sampler (see data loader)
    epochs:        int   = 60
    frozen_epochs: int   = 5       # Stage-1 warmup (backbones frozen)
    finetune_blocks: int = 2       # ViT blocks unfrozen in Stage-2 (0 = keep frozen)
    head_lr:       float = 1e-4
    backbone_lr:   float = 1e-5
    weight_decay:  float = 1e-4
    feat_noise:    float = 0.0     # optional gaussian noise on tokens (0 = off)
    grad_clip:     float = 1.0

    # class-balanced sampler: m images per class per batch
    classes_per_batch:    int = 20
    samples_per_class:    int = 6   # batch_size ~= classes_per_batch * samples_per_class

    # ── Eval / logging ────────────────────────────────────────────────────
    eval_every: int = 5     # run test every N train epochs (also always at last epoch)
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
    })

    @property
    def num_slots(self) -> int:
        return self.n_experts * self.slots_per_expert


HCFG = HyMSConfig()
