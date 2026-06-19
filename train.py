"""
Train HyMS-Route SEPARATELY on a single DML benchmark (CUB / Cars / In-Shop).

Two-stage schedule:
  Stage 1 (--frozen_epochs): both backbones frozen, only head + Soft MoE train.
  Stage 2:                   last --finetune_blocks DINOv2 blocks unfrozen
                             (backbone_lr << head_lr). CNN stays frozen.

Loss:  L = SupCon(z) + lambda_route * RoutingConsistency(rho)

Usage:
  python train_dml.py --dataset cub
  python train_dml.py --dataset cars  --epochs 80 --seed 0
  python train_dml.py --dataset inshop --finetune_blocks 0

Outputs:
  results/checkpoints/best_hyms_{dataset}_seed{seed}.pt
  results/history_hyms_{dataset}_seed{seed}.csv
"""
import os, argparse, json, dataclasses, datetime
import torch
import pandas as pd
from tqdm import tqdm
from pytorch_metric_learning.losses import SupConLoss

from config import HCFG
from utils import set_seed
from data.dml_dataset import get_dml_loaders
from models.hybrid_encoder import HybridEncoder
from models.hyms_route import HyMSRoute
from losses.routing_consistency import RoutingConsistencyLoss
from eval.routerank import evaluate_self, evaluate_query_gallery


def apply_overrides(cfg):
    """Áp config override do notebook ghi ra (ghi đè config.py cho lần chạy này).

    Đọc file JSON tại $HCFG_OVERRIDES nếu được set, mặc định
    'results/_notebook_overrides.json'. Mỗi key phải là một field hợp lệ của HCFG.
    Gọi TRƯỚC parse_args() để các default của argparse cũng nhận giá trị mới.
    """
    path = os.environ.get("HCFG_OVERRIDES",
                          os.path.join("results", "_notebook_overrides.json"))
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        overrides = json.load(f)
    applied, skipped = {}, []
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v); applied[k] = v
        else:
            skipped.append(k)
    if applied:
        print(f"[overrides] từ {path}: {applied}")
    if skipped:
        print(f"[overrides] BỎ QUA (không phải field của HCFG): {skipped}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["cub", "cars", "inshop"])
    p.add_argument("--epochs", type=int, default=HCFG.epochs)
    p.add_argument("--frozen_epochs", type=int, default=HCFG.frozen_epochs)
    p.add_argument("--finetune_blocks", type=int, default=HCFG.finetune_blocks)
    p.add_argument("--head_lr", type=float, default=HCFG.head_lr)
    p.add_argument("--backbone_lr", type=float, default=HCFG.backbone_lr)
    p.add_argument("--lambda_route", type=float, default=HCFG.lambda_route)
    p.add_argument("--seed", type=int, default=HCFG.seed)
    p.add_argument("--eval_every", type=int, default=HCFG.eval_every,
                   help="run test every N epochs (default from config: HCFG.eval_every)")
    return p.parse_args()


def evaluate(model, loaders, dataset, device):
    rk = HCFG.recall_k_for(dataset)            # CUB/Cars 1/2/4/8 · In-Shop 1/10/20/30
    if dataset in ("cub", "cars"):
        return evaluate_self(model, loaders["test"], device, HCFG, recall_k=rk)
    return evaluate_query_gallery(model, loaders["query"], loaders["gallery"],
                                  device, HCFG, recall_k=rk)


def main():
    apply_overrides(HCFG)          # notebook overrides -> trước parse_args để default lấy giá trị mới
    args = parse_args()
    HCFG.lambda_route = args.lambda_route
    set_seed(args.seed)
    os.makedirs(HCFG.checkpoint_dir, exist_ok=True)
    os.makedirs(HCFG.results_dir, exist_ok=True)
    os.makedirs(HCFG.log_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = f"hyms_{args.dataset}_seed{args.seed}"

    # ── logger: console + results/logs/{run}.log ──────────────────────────
    log_path = os.path.join(HCFG.log_dir, f"{run}.log")
    log_f = open(log_path, "a", encoding="utf-8")
    def log(msg=""):
        print(msg)
        log_f.write(str(msg) + "\n"); log_f.flush()

    log(f"\n===== {run} @ {datetime.datetime.now():%Y-%m-%d %H:%M:%S} =====")
    log(f"device={device} | dataset={args.dataset} | seed={args.seed} | eval_every={args.eval_every}")
    loaders = get_dml_loaders(args.dataset, HCFG)
    log(f"Train batches/epoch: {len(loaders['train'])}")

    encoder = HybridEncoder(HCFG.vit_name, HCFG.cnn_name, device)
    encoder.freeze_all()
    model = HyMSRoute(encoder, HCFG).to(device)

    n_head = sum(p.numel() for p in model.head_parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ── tóm tắt tham số trước khi train ───────────────────────────────────
    log("----- Tham số chạy -----")
    log(f"  total params   : {total_params:,}")
    log(f"  trainable params: {train_params:,} (head {n_head:,})")
    log(f"  epochs         : {args.epochs} (frozen {args.frozen_epochs}, finetune_blocks {args.finetune_blocks})")
    log(f"  lr             : head {args.head_lr} | backbone {args.backbone_lr}")
    log(f"  batch_size     : {HCFG.classes_per_batch * HCFG.samples_per_class} "
        f"(classes_per_batch {HCFG.classes_per_batch} x samples_per_class {HCFG.samples_per_class})")
    log(f"  n_experts      : {HCFG.n_experts} | slots_per_expert {HCFG.slots_per_expert} | num_slots {HCFG.num_slots}")
    log(f"  embed_dim      : {HCFG.embed_dim} | route_dim {HCFG.route_dim} | lambda_route {HCFG.lambda_route}")
    log("------------------------")

    # ── config snapshot tied to this run ──────────────────────────────────
    cfg_snapshot = dataclasses.asdict(HCFG)
    cfg_snapshot.update({"run": run, "dataset": args.dataset, "seed": args.seed,
                         "epochs": args.epochs, "frozen_epochs": args.frozen_epochs,
                         "finetune_blocks": args.finetune_blocks,
                         "head_lr": args.head_lr, "backbone_lr": args.backbone_lr,
                         "eval_every": args.eval_every,
                         "recall_k": HCFG.recall_k_for(args.dataset),
                         "n_head_params": n_head,
                         "timestamp": datetime.datetime.now().isoformat()})
    cfg_path = os.path.join(HCFG.log_dir, f"{run}_config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg_snapshot, f, indent=2)
    log(f"Config snapshot -> {cfg_path}")

    supcon = SupConLoss(temperature=HCFG.temperature)
    route_loss = RoutingConsistencyLoss(temperature=HCFG.route_temperature)

    def make_optim(stage):
        if stage == 1:
            return torch.optim.AdamW(model.head_parameters(),
                                     lr=args.head_lr, weight_decay=HCFG.weight_decay)
        return torch.optim.AdamW([
            {"params": model.head_parameters(), "lr": args.head_lr},
            {"params": encoder.trainable_backbone_parameters(), "lr": args.backbone_lr},
        ], weight_decay=HCFG.weight_decay)

    optim = make_optim(1)
    best, stage = 0.0, 1
    train_log, test_log = [], []                       # per-epoch / per-eval logs
    train_csv = os.path.join(HCFG.results_dir, f"train_{run}.csv")
    test_csv  = os.path.join(HCFG.results_dir, f"test_{run}.csv")

    for epoch in range(1, args.epochs + 1):
        if stage == 1 and epoch > args.frozen_epochs:
            stage = 2
            encoder.unfreeze_vit_blocks(args.finetune_blocks)
            optim = make_optim(2)
            log(f"--- Stage 2 (epoch {epoch}): unfroze last {args.finetune_blocks} ViT blocks ---")

        encoder.train() if stage == 2 else encoder.eval()
        model.train()
        tot = tot_sc = tot_rt = 0.0

        for imgs, labels in tqdm(loaders["train"], desc=f"Ep{epoch:3d}[S{stage}]", leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device)
            z, rho, _ = model(imgs)
            sc = supcon(z, labels)
            rt = route_loss(rho, labels)
            loss = sc + HCFG.lambda_route * rt

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.head_parameters() + encoder.trainable_backbone_parameters(),
                HCFG.grad_clip)
            optim.step()
            tot += loss.item(); tot_sc += sc.item(); tot_rt += rt.item()

        n = len(loaders["train"])
        # ── TRAIN log every epoch ─────────────────────────────────────────
        train_row = {"epoch": epoch, "stage": stage,
                     "loss": round(tot / n, 4), "sc": round(tot_sc / n, 4),
                     "route": round(tot_rt / n, 4)}
        train_log.append(train_row)
        pd.DataFrame(train_log).to_csv(train_csv, index=False)   # incremental
        log(f"Ep{epoch:3d}[S{stage}] train loss={tot/n:.4f} (sc={tot_sc/n:.4f} rt={tot_rt/n:.4f})")

        # ── TEST every eval_every epochs (and last) ───────────────────────
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            res = evaluate(model, loaders, args.dataset, device)
            r1 = res["routerank"]["R@1"]
            test_row = {"epoch": epoch, "stage": stage}
            for tag in ("base", "routerank"):
                for m, v in res[tag].items():
                    test_row[f"{tag}.{m}"] = v
            test_log.append(test_row)
            pd.DataFrame(test_log).to_csv(test_csv, index=False)  # incremental
            log(f"   [test] base     : " + " ".join(f"{m}={v}" for m, v in res["base"].items()))
            log(f"   [test] routerank: " + " ".join(f"{m}={v}" for m, v in res["routerank"].items()))

            if r1 > best:
                best = r1
                ckpt = os.path.join(HCFG.checkpoint_dir, f"best_{run}.pt")
                torch.save({"model": model.state_dict(), "epoch": epoch, "R@1": best,
                            "config": cfg_snapshot}, ckpt)
                log(f"   -> new best R@1={best:.2f}  saved {ckpt}")

    log(f"Done. Best R@1={best:.2f}")
    log(f"Logs: {log_path} | train: {train_csv} | test: {test_csv} | config: {cfg_path}")
    log_f.close()


if __name__ == "__main__":
    main()
