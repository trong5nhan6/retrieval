"""
Clean DINOv2 retrieval baseline (NO Soft MoE / NO CNN / NO 256-d bottleneck).

This reproduces the "DINOv2-ViT-B/14 + triploss + unfreeze last N blocks" row of
the baseline table. It is deliberately minimal so its number is a faithful
backbone reference, separate from the HyMS-Route head:

    img -> DINOv2 -> pool (cls | mean | cls+mean) -> Linear proj -> L2 -> z
    loss = Triplet(margin) + semihard mining   (or SupCon)

Two-stage: Stage-1 trains only the projection head (backbone frozen); Stage-2
unfreezes the last --finetune_blocks ViT blocks. Cosine LR + best-checkpoint.

Reuses the repo's data loaders and eval (evaluate_self / evaluate_query_gallery)
so metrics are directly comparable with train.py (base channel only).

Usage:
    python train_baseline.py --dataset cub --seed 42
    python train_baseline.py --dataset cub --pool cls --embed_dim 512 \
        --loss triplet --finetune_blocks 2 --epochs 30 --eval_every 2
"""
import os, argparse, json, datetime, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel
from pytorch_metric_learning.losses import SupConLoss, TripletMarginLoss
from pytorch_metric_learning.miners import TripletMarginMiner

from config import HCFG
from utils import set_seed
from data.dml_dataset import get_dml_loaders
from eval.routerank import evaluate_self, evaluate_query_gallery
from train import apply_overrides          # reuse notebook-overrides mechanism


# ── model ────────────────────────────────────────────────────────────────────
class DinoBaseline(nn.Module):
    """DINOv2 backbone + pooling + linear projection to an L2 retrieval embedding.

    forward(imgs) -> (z, None, None)   # (None, None) keep eval's (z, rho, C) API
    """
    def __init__(self, vit_name="facebook/dinov2-base", pool="cls", embed_dim=512):
        super().__init__()
        assert pool in ("cls", "mean", "cls_mean")
        self.pool = pool
        self.vit = AutoModel.from_pretrained(vit_name)
        d = self.vit.config.hidden_size                # 768 for ViT-B/14
        in_dim = d * (2 if pool == "cls_mean" else 1)
        # embed_dim<=0 -> identity (retrieve on raw pooled features)
        self.proj = nn.Identity() if embed_dim <= 0 else nn.Linear(in_dim, embed_dim)

    def freeze_backbone(self):
        for p in self.vit.parameters():
            p.requires_grad_(False)

    def unfreeze_last_blocks(self, n):
        if n <= 0:
            return
        for layer in self.vit.encoder.layer[-n:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        for p in self.vit.layernorm.parameters():
            p.requires_grad_(True)

    def head_parameters(self):
        return list(self.proj.parameters())

    def trainable_backbone_parameters(self):
        return [p for p in self.vit.parameters() if p.requires_grad]

    def forward(self, imgs):
        out = self.vit(pixel_values=imgs).last_hidden_state   # [B, 1+P, d]
        cls = out[:, 0]
        mean = out[:, 1:].mean(dim=1)
        if self.pool == "cls":
            feat = cls
        elif self.pool == "mean":
            feat = mean
        else:
            feat = torch.cat([cls, mean], dim=-1)
        z = F.normalize(self.proj(feat), dim=-1)
        return z, None, None


# ── helpers (mirror train.py) ────────────────────────────────────────────────
def make_scheduler(optimizer, n_epochs, warmup, kind):
    if kind != "cosine":
        return None

    def fn(e):
        if warmup > 0 and e < warmup:
            return (e + 1) / warmup
        prog = (e - warmup) / max(1, n_epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def build_loss(args):
    if args.loss == "triplet":
        loss = TripletMarginLoss(margin=args.margin)
        miner = TripletMarginMiner(margin=args.margin, type_of_triplets="semihard")
        return loss, miner
    return SupConLoss(temperature=0.07), None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["cub", "cars", "inshop"])
    p.add_argument("--vit_name", default=HCFG.vit_name)
    p.add_argument("--pool", default="cls", choices=["cls", "mean", "cls_mean"])
    p.add_argument("--embed_dim", type=int, default=512, help="<=0 => no projection")
    p.add_argument("--loss", default="triplet", choices=["triplet", "supcon"])
    p.add_argument("--margin", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--frozen_epochs", type=int, default=5)
    p.add_argument("--finetune_blocks", type=int, default=2)
    p.add_argument("--head_lr", type=float, default=1e-4)
    p.add_argument("--backbone_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lr_schedule", default="cosine", choices=["cosine", "constant"])
    p.add_argument("--warmup_epochs", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def evaluate(model, loaders, dataset, device):
    rk = HCFG.recall_k_for(dataset)
    # use_routerank=False: this baseline has no routing fingerprint
    if dataset in ("cub", "cars"):
        return evaluate_self(model, loaders["test"], device, HCFG,
                             use_routerank=False, recall_k=rk)
    return evaluate_query_gallery(model, loaders["query"], loaders["gallery"],
                                  device, HCFG, use_routerank=False, recall_k=rk)


def main():
    apply_overrides(HCFG)      # shared HCFG fields (sampler, etc.) from notebook
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(HCFG.checkpoint_dir, exist_ok=True)
    os.makedirs(HCFG.results_dir, exist_ok=True)
    os.makedirs(HCFG.log_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = f"baseline_{args.dataset}_seed{args.seed}"

    log_path = os.path.join(HCFG.log_dir, f"{run}.log")
    log_f = open(log_path, "a", encoding="utf-8")
    def log(msg=""):
        print(msg); log_f.write(str(msg) + "\n"); log_f.flush()

    log(f"\n===== {run} @ {datetime.datetime.now():%Y-%m-%d %H:%M:%S} =====")
    loaders = get_dml_loaders(args.dataset, HCFG)
    log(f"Train batches/epoch: {len(loaders['train'])}")

    model = DinoBaseline(args.vit_name, args.pool, args.embed_dim).to(device)
    model.freeze_backbone()
    total = sum(p.numel() for p in model.parameters())
    log("----- Baseline -----")
    log(f"  vit={args.vit_name} pool={args.pool} embed_dim={args.embed_dim}")
    log(f"  loss={args.loss}(margin={args.margin}) | epochs={args.epochs} "
        f"(frozen {args.frozen_epochs}, finetune_blocks {args.finetune_blocks})")
    log(f"  lr head {args.head_lr} backbone {args.backbone_lr} | sched {args.lr_schedule}")
    log(f"  total params {total:,}")
    log("--------------------")

    loss_fn, miner = build_loss(args)

    def make_optim(stage):
        if stage == 1:
            return torch.optim.AdamW(model.head_parameters(),
                                     lr=args.head_lr, weight_decay=args.weight_decay)
        return torch.optim.AdamW([
            {"params": model.head_parameters(), "lr": args.head_lr},
            {"params": model.trainable_backbone_parameters(), "lr": args.backbone_lr},
        ], weight_decay=args.weight_decay)

    optim = make_optim(1)
    sched = make_scheduler(optim, max(1, min(args.frozen_epochs, args.epochs)),
                           args.warmup_epochs, args.lr_schedule)
    best, stage = 0.0, 1
    train_log, test_log = [], []
    train_csv = os.path.join(HCFG.results_dir, f"train_{run}.csv")
    test_csv = os.path.join(HCFG.results_dir, f"test_{run}.csv")

    for epoch in range(1, args.epochs + 1):
        if stage == 1 and epoch > args.frozen_epochs:
            stage = 2
            model.unfreeze_last_blocks(args.finetune_blocks)
            optim = make_optim(2)
            sched = make_scheduler(optim, max(1, args.epochs - args.frozen_epochs),
                                   0, args.lr_schedule)
            log(f"--- Stage 2 (epoch {epoch}): unfroze last {args.finetune_blocks} blocks ---")

        model.train()
        if stage == 1:
            model.vit.eval()                    # keep frozen backbone in eval mode
        tot = 0.0
        for imgs, labels in tqdm(loaders["train"], desc=f"Ep{epoch:3d}[S{stage}]", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            z, _, _ = model(imgs)
            sc = loss_fn(z, labels, miner(z, labels)) if miner is not None else loss_fn(z, labels)
            optim.zero_grad(); sc.backward()
            torch.nn.utils.clip_grad_norm_(
                model.head_parameters() + model.trainable_backbone_parameters(), 1.0)
            optim.step()
            tot += sc.item()

        n = len(loaders["train"])
        train_log.append({"epoch": epoch, "stage": stage, "loss": round(tot / n, 4)})
        pd.DataFrame(train_log).to_csv(train_csv, index=False)
        log(f"Ep{epoch:3d}[S{stage}] train loss={tot/n:.4f}")
        if sched is not None:
            sched.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            res = evaluate(model, loaders, args.dataset, device)
            r1 = res["base"]["R@1"]
            row = {"epoch": epoch, "stage": stage}
            row.update({f"base.{m}": v for m, v in res["base"].items()})
            test_log.append(row)
            pd.DataFrame(test_log).to_csv(test_csv, index=False)
            log(f"   [test] base: " + " ".join(f"{m}={v}" for m, v in res["base"].items()))
            if r1 > best:
                best = r1
                ckpt = os.path.join(HCFG.checkpoint_dir, f"best_{run}.pt")
                torch.save({"model": model.state_dict(), "epoch": epoch, "R@1": best,
                            "args": vars(args)}, ckpt)
                log(f"   -> new best R@1={best:.2f}  saved {ckpt}")

    log(f"Done. Best R@1={best:.2f}")
    try:
        from plot_test_metrics import plot_test_metrics, plot_train_loss
        plot_test_metrics(test_csv, out=os.path.join(HCFG.results_dir, f"plot_test_{run}.png"),
                          channels=["base"], title=f"Baseline test — {run}")
        plot_train_loss(train_csv, out=os.path.join(HCFG.results_dir, f"plot_train_{run}.png"),
                        title=f"Baseline loss — {run}")
    except Exception as e:
        log(f"[plot] skipped ({type(e).__name__}: {e})")
    log_f.close()


if __name__ == "__main__":
    main()
