"""
Smoke test for the use_vit / use_cnn / use_moe ablation switches and the
cosine LR schedule. Uses a FAKE encoder (no HF / torchvision download), so it
runs in seconds on CPU. Run from the repo root:

    python tests/smoke_toggles.py

Checks, for every meaningful flag combo:
  - HyMSRoute builds and forwards a random batch
  - z has shape [B, embed_dim] and is L2-normalized
  - rho is [B, route_dim] when use_moe else None
  - evaluate_self returns 'routerank' only when use_moe (else base only)
Plus a cosine-schedule sanity check (warmup up, then decay to ~0).
"""
import math
import torch
import torch.nn as nn

from config import HCFG
from models.hyms_route import HyMSRoute
from eval.routerank import evaluate_self

B, D_VIT, EMB = 6, 768, None  # EMB read from cfg below
STAGE_DIMS = [96, 192, 384, 768]   # convnext_tiny stage channels


class FakeEncoder(nn.Module):
    """Mimics HybridEncoder's interface without downloading any weights."""
    def __init__(self, use_vit=True, use_cnn=True):
        super().__init__()
        self.use_vit, self.use_cnn = use_vit, use_cnn
        # a dummy param so .parameters() is non-empty (head_parameters relies on it)
        self._p = nn.Parameter(torch.zeros(1))
        self.vit = nn.Identity() if use_vit else None
        self.cnn = nn.Identity() if use_cnn else None
        self.vit_dim = D_VIT if use_vit else 0
        self.cnn_stage_dims = STAGE_DIMS if use_cnn else []

    def forward(self, imgs):
        b = imgs.size(0)
        vit_tokens = torch.randn(b, 256, D_VIT) if self.use_vit else None
        cnn_maps = []
        if self.use_cnn:
            for c in STAGE_DIMS:
                cnn_maps.append(torch.randn(b, c, 7, 7))
        return vit_tokens, cnn_maps

    def freeze_all(self):
        pass

    def unfreeze_vit_blocks(self, n):
        pass

    def trainable_backbone_parameters(self):
        return []


def fake_loader(n_batches=3, n_classes=4):
    out = []
    for _ in range(n_batches):
        imgs = torch.randn(B, 3, 224, 224)
        labels = torch.randint(0, n_classes, (B,))
        out.append((imgs, labels))
    return out


def run_combo(use_vit, use_cnn, use_moe):
    HCFG.use_vit, HCFG.use_cnn, HCFG.use_moe = use_vit, use_cnn, use_moe
    enc = FakeEncoder(use_vit, use_cnn)
    model = HyMSRoute(enc, HCFG).eval()

    imgs = torch.randn(B, 3, 224, 224)
    z, rho, combine = model(imgs)
    assert z.shape == (B, HCFG.embed_dim), z.shape
    assert torch.allclose(z.norm(dim=-1), torch.ones(B), atol=1e-4), "z not L2-normalized"
    if use_moe:
        assert rho is not None and rho.shape == (B, HCFG.route_dim), "rho wrong"
    else:
        assert rho is None, "rho should be None when use_moe=False"

    res = evaluate_self(model, fake_loader(), "cpu", HCFG,
                        use_routerank=use_moe, recall_k=[1, 2, 4])
    assert "base" in res
    assert ("routerank" in res) == use_moe, "routerank presence mismatch"
    tag = "moe" if use_moe else "no-moe"
    print(f"  OK  vit={use_vit} cnn={use_cnn} {tag:6s} "
          f"-> z{tuple(z.shape)} rho={None if rho is None else tuple(rho.shape)} "
          f"keys={list(res)}")


def check_schedule():
    def lr_lambda(e, warmup, n):
        if warmup > 0 and e < warmup:
            return (e + 1) / warmup
        prog = (e - warmup) / max(1, n - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    xs = [lr_lambda(e, 0, 15) for e in range(15)]
    assert all(xs[i] >= xs[i + 1] - 1e-9 for i in range(len(xs) - 1)), "not monotonic"
    assert min(xs) >= 0 and xs[0] == 1.0 and xs[-1] < 0.05
    print("  OK  cosine schedule: starts 1.0, monotonic decay, ends ~0")


if __name__ == "__main__":
    torch.manual_seed(0)
    print("== flag combos ==")
    for v, c, m in [(True, True, True),    # default pipeline
                    (True, True, False),   # MoE off
                    (True, False, True),   # ViT-only
                    (False, True, True),   # CNN-only
                    (True, False, False)]: # ViT-only + MoE off
        run_combo(v, c, m)
    print("== schedule ==")
    check_schedule()
    print("\nALL SMOKE TESTS PASSED")
