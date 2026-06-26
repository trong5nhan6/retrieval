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

Outputs (per run):
  results/{dataset}/{timestamp}_seed{N}_{flags}/best.pt
  results/{dataset}/{timestamp}_seed{N}_{flags}/train.csv
  results/{dataset}/{timestamp}_seed{N}_{flags}/test.csv
  results/{dataset}/{timestamp}_seed{N}_{flags}/config.json
  results/{dataset}/{timestamp}_seed{N}_{flags}/run.log
"""
import os, argparse, json, dataclasses, datetime, math
# Reduce CUDA fragmentation (must be set BEFORE torch initializes CUDA).
# Fixes the "reserved but unallocated" OOM that hits eval after Stage-2 unfreeze.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import pandas as pd
from tqdm import tqdm
from pytorch_metric_learning.losses import SupConLoss, TripletMarginLoss
from pytorch_metric_learning.miners import TripletMarginMiner

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
    p.add_argument("--dataset", required=True, choices=["cub", "cars", "inshop", "sop"])
    p.add_argument("--epochs", type=int, default=HCFG.epochs)
    p.add_argument("--frozen_epochs", type=int, default=HCFG.frozen_epochs)
    p.add_argument("--finetune_blocks", type=int, default=HCFG.finetune_blocks)
    p.add_argument("--finetune_cnn_stages", type=int, default=HCFG.finetune_cnn_stages)
    p.add_argument("--head_lr", type=float, default=HCFG.head_lr)
    p.add_argument("--backbone_lr", type=float, default=HCFG.backbone_lr)
    p.add_argument("--lambda_route", type=float, default=HCFG.lambda_route)
    p.add_argument("--seed", type=int, default=HCFG.seed)
    p.add_argument("--eval_every", type=int, default=HCFG.eval_every,
                   help="run test every N epochs (default from config: HCFG.eval_every)")
    p.add_argument("--run_id", type=str, default="",
                   help="custom run identifier (default: timestamp YYYYMMDD_HHMMSS)")
    return p.parse_args()


def evaluate(model, loaders, dataset, device):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()                # defrag before eval (avoid Stage-2 OOM)
    rk = HCFG.recall_k_for(dataset)            # CUB/Cars 1/2/4/8 · In-Shop 1/10/20/30
    use_rr = HCFG.use_moe                       # routerank needs rho (Soft MoE on)
    if dataset in ("cub", "cars", "sop"):
        return evaluate_self(model, loaders["test"], device, HCFG,
                             use_routerank=use_rr, recall_k=rk)
    return evaluate_query_gallery(model, loaders["query"], loaders["gallery"],
                                  device, HCFG, use_routerank=use_rr, recall_k=rk)


def build_embed_loss(cfg):
    """Main loss on the retrieval embedding z.

    Returns (loss_fn, miner). For "triplet" we optionally mine semihard triplets;
    both losses share the (embeddings, labels, indices_tuple) call signature."""
    if cfg.loss_type == "triplet":
        loss = TripletMarginLoss(margin=cfg.triplet_margin)
        miner = (TripletMarginMiner(margin=cfg.triplet_margin,
                                    type_of_triplets="semihard")
                 if cfg.triplet_miner else None)
        return loss, miner
    if cfg.loss_type == "supcon":
        return SupConLoss(temperature=cfg.temperature), None
    raise ValueError(f"unknown loss_type: {cfg.loss_type!r} (use 'supcon' | 'triplet')")


def make_scheduler(optimizer, n_epochs, warmup_epochs, kind):
    """Per-stage scheduler stepped once per epoch.

    "cosine": linear warmup for warmup_epochs, then cosine decay to 0 over the
    remaining epochs. Scales every param-group equally, so the head:backbone LR
    ratio is preserved. "constant": no scheduler (old behaviour)."""
    if kind != "cosine":
        return None

    def lr_lambda(e):                       # e = epoch index within this stage (0-based)
        if warmup_epochs > 0 and e < warmup_epochs:
            return (e + 1) / warmup_epochs
        prog = (e - warmup_epochs) / max(1, n_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    apply_overrides(HCFG)          # notebook overrides -> trước parse_args để default lấy giá trị mới
    args = parse_args()
    HCFG.lambda_route = args.lambda_route
    # Soft MoE off => no rho => no routing loss / no routerank.
    if not HCFG.use_moe:
        HCFG.lambda_route = 0.0
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── run directory: results/{dataset}/{timestamp}_seed{N}_{active_flags} ──
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    flags = "_".join(k for k, v in [("vit", HCFG.use_vit), ("cnn", HCFG.use_cnn), ("moe", HCFG.use_moe)] if v)
    run = f"{timestamp}_seed{args.seed}_{flags}"
    run_dir = os.path.join(HCFG.results_dir, args.dataset, run)
    os.makedirs(run_dir, exist_ok=True)

    # ── logger: console + {run_dir}/run.log ───────────────────────────────
    log_path = os.path.join(run_dir, "run.log")
    log_f = open(log_path, "w", encoding="utf-8")
    def log(msg=""):
        print(msg)
        log_f.write(str(msg) + "\n"); log_f.flush()

    log(f"\n===== {run} @ {datetime.datetime.now():%Y-%m-%d %H:%M:%S} =====")
    log(f"device={device} | dataset={args.dataset} | seed={args.seed} | eval_every={args.eval_every}")
    loaders = get_dml_loaders(args.dataset, HCFG)

    # ── thông tin dataset ──────────────────────────────────────────────────
    n_train_imgs    = len(loaders['train'].dataset)
    n_train_classes = len(set(lbl for _, lbl in loaders['train'].dataset.items))
    eval_key        = 'test' if args.dataset in ('cub', 'cars', 'sop') else 'query'
    n_eval_imgs     = len(loaders[eval_key].dataset)
    gallery_str     = (f", {len(loaders['gallery'].dataset)} gallery imgs"
                       if 'gallery' in loaders else "")
    log(f"Dataset      : {args.dataset.upper()} | root: {HCFG.data_roots[args.dataset]}")
    log(f"  train      : {n_train_classes} classes, {n_train_imgs} imgs")
    log(f"  {eval_key:<8}: {n_eval_imgs} imgs{gallery_str}")
    log(f"Train batches/epoch: {len(loaders['train'])}")

    encoder = HybridEncoder(HCFG.vit_name, HCFG.cnn_name, device,
                            use_vit=HCFG.use_vit, use_cnn=HCFG.use_cnn)
    encoder.freeze_all()
    model = HyMSRoute(encoder, HCFG).to(device)

    n_head = sum(p.numel() for p in model.head_parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ── tóm tắt tham số trước khi train ───────────────────────────────────
    log("----- Tham số chạy -----")
    log(f"  total params   : {total_params:,}")
    log(f"  trainable params: {train_params:,} (head {n_head:,})")
    log(f"  epochs         : {args.epochs} (frozen {args.frozen_epochs}, finetune_blocks {args.finetune_blocks}, finetune_cnn_stages {args.finetune_cnn_stages})")
    log(f"  lr             : head {args.head_lr} | backbone {args.backbone_lr}")
    log(f"  batch_size     : {HCFG.classes_per_batch * HCFG.samples_per_class} "
        f"(classes_per_batch {HCFG.classes_per_batch} x samples_per_class {HCFG.samples_per_class})")
    log(f"  n_experts      : {HCFG.n_experts} | slots_per_expert {HCFG.slots_per_expert} | num_slots {HCFG.num_slots}")
    log(f"  embed_dim      : {HCFG.embed_dim} | route_dim {HCFG.route_dim} | lambda_route {HCFG.lambda_route}")
    log(f"  fusion         : cls_skip={HCFG.use_cls_skip} gate_init={HCFG.local_gate_init} bnneck={HCFG.bnneck}")
    log(f"  switches       : use_vit={HCFG.use_vit} use_cnn={HCFG.use_cnn} use_moe={HCFG.use_moe}")
    log(f"  lr_schedule    : {HCFG.lr_schedule} (warmup {HCFG.warmup_epochs})")
    loss_desc = (f"triplet(margin={HCFG.triplet_margin}, miner={HCFG.triplet_miner})"
                 if HCFG.loss_type == "triplet" else f"supcon(tau={HCFG.temperature})")
    log(f"  embed loss     : {loss_desc}")
    log("------------------------")

    # ── config snapshot tied to this run ──────────────────────────────────
    cfg_snapshot = dataclasses.asdict(HCFG)
    cfg_snapshot.update({"run": run, "run_dir": run_dir, "dataset": args.dataset, "seed": args.seed,
                         "epochs": args.epochs, "frozen_epochs": args.frozen_epochs,
                         "finetune_blocks": args.finetune_blocks,
                         "finetune_cnn_stages": args.finetune_cnn_stages,
                         "head_lr": args.head_lr, "backbone_lr": args.backbone_lr,
                         "eval_every": args.eval_every,
                         "recall_k": HCFG.recall_k_for(args.dataset),
                         "n_head_params": n_head,
                         "data_root": HCFG.data_roots[args.dataset],
                         "n_train_classes": n_train_classes,
                         "n_train_imgs": n_train_imgs,
                         f"n_{eval_key}_imgs": n_eval_imgs,
                         **({"n_gallery_imgs": len(loaders['gallery'].dataset)}
                            if 'gallery' in loaders else {}),
                         "timestamp": datetime.datetime.now().isoformat()})
    cfg_path = os.path.join(run_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg_snapshot, f, indent=2)
    log(f"Config snapshot -> {cfg_path}")

    embed_loss, miner = build_embed_loss(HCFG)        # SupCon or Triplet on z
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
    # Stage-1 scheduler spans the frozen warmup phase; rebuilt for Stage 2 below.
    stage1_epochs = min(args.frozen_epochs, args.epochs)
    sched = make_scheduler(optim, max(1, stage1_epochs),
                           HCFG.warmup_epochs, HCFG.lr_schedule)
    best, stage = 0.0, 1
    train_log, test_log = [], []                       # per-epoch / per-eval logs
    train_csv = os.path.join(run_dir, "train.csv")
    test_csv  = os.path.join(run_dir, "test.csv")

    for epoch in range(1, args.epochs + 1):
        if stage == 1 and epoch > args.frozen_epochs:
            stage = 2
            encoder.unfreeze_vit_blocks(args.finetune_blocks)
            encoder.unfreeze_cnn_stages(args.finetune_cnn_stages)
            optim = make_optim(2)
            # Fresh cosine over the remaining (Stage-2) epochs, no warmup.
            sched = make_scheduler(optim, max(1, args.epochs - args.frozen_epochs),
                                   0, HCFG.lr_schedule)
            log(f"--- Stage 2 (epoch {epoch}): unfroze last {args.finetune_blocks} ViT blocks "
                f"+ {args.finetune_cnn_stages} CNN stages ---")

        encoder.train() if stage == 2 else encoder.eval()
        model.train()
        tot = tot_sc = tot_rt = 0.0

        for imgs, labels in tqdm(loaders["train"], desc=f"Ep{epoch:3d}[S{stage}]", leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device)
            z, rho, _ = model(imgs)
            if miner is not None:
                sc = embed_loss(z, labels, miner(z, labels))
            else:
                sc = embed_loss(z, labels)
            if rho is not None and HCFG.lambda_route > 0:
                rt = route_loss(rho, labels)
                loss = sc + HCFG.lambda_route * rt
            else:
                rt = torch.zeros((), device=z.device)
                loss = sc

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.head_parameters() + encoder.trainable_backbone_parameters(),
                HCFG.grad_clip)
            optim.step()
            tot += loss.item(); tot_sc += sc.item(); tot_rt += rt.item()

        n = len(loaders["train"])
        # γ = local_gate (sức nặng nhánh MoE). None nếu tắt cls_skip.
        gp = getattr(model, "local_gate", None)
        gate = float(gp.detach().cpu()) if gp is not None else None
        # ── TRAIN log every epoch ─────────────────────────────────────────
        train_row = {"epoch": epoch, "stage": stage,
                     "loss": round(tot / n, 4), "sc": round(tot_sc / n, 4),
                     "route": round(tot_rt / n, 4),
                     "gate": round(gate, 4) if gate is not None else None}
        train_log.append(train_row)
        pd.DataFrame(train_log).to_csv(train_csv, index=False)   # incremental
        gate_str = f" gate={gate:+.4f}" if gate is not None else ""
        log(f"Ep{epoch:3d}[S{stage}] train loss={tot/n:.4f} "
            f"(sc={tot_sc/n:.4f} rt={tot_rt/n:.4f}){gate_str}")

        if sched is not None:
            sched.step()

        # ── TEST every eval_every epochs (and last) ───────────────────────
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            res = evaluate(model, loaders, args.dataset, device)
            # Model-selection metric: routerank R@1 when available, else base R@1.
            r1 = res["routerank"]["R@1"] if "routerank" in res else res["base"]["R@1"]
            test_row = {"epoch": epoch, "stage": stage}
            for tag in ("base", "routerank"):
                if tag in res:
                    for m, v in res[tag].items():
                        test_row[f"{tag}.{m}"] = v
            test_log.append(test_row)
            pd.DataFrame(test_log).to_csv(test_csv, index=False)  # incremental
            log(f"   [test] base     : " + " ".join(f"{m}={v}" for m, v in res["base"].items()))
            if "routerank" in res:
                log(f"   [test] routerank: " + " ".join(f"{m}={v}" for m, v in res["routerank"].items()))

            if r1 > best:
                best = r1
                ckpt = os.path.join(run_dir, "best.pt")
                torch.save({"model": model.state_dict(), "epoch": epoch, "R@1": best,
                            "config": cfg_snapshot}, ckpt)
                log(f"   -> new best R@1={best:.2f}  saved {ckpt}")

    log(f"Done. Best R@1={best:.2f}")
    gp = getattr(model, "local_gate", None)
    if gp is not None:
        log(f"Final local_gate γ = {float(gp.detach().cpu()):+.4f}  "
            f"(γ≈0 -> MoE branch không đóng góp; |γ| lớn -> MoE có ích)")

    # ── auto-plot test metrics + train loss -> results/plot_*_{run}.png ────
    try:
        from plot_test_metrics import plot_test_metrics, plot_train_loss
        test_png = os.path.join(run_dir, "plot_test.png")
        plot_test_metrics(test_csv, out=test_png, title=f"Test-metric trends — {run}")
        loss_png = os.path.join(run_dir, "plot_train.png")
        plot_train_loss(train_csv, out=loss_png, title=f"Training loss — {run}")
        log(f"Plots: {test_png} | {loss_png}")
    except Exception as e:                       # never let plotting break a finished run
        log(f"[plot] skipped ({type(e).__name__}: {e})")

    log(f"Run dir: {run_dir}")
    log(f"Logs: {log_path} | train: {train_csv} | test: {test_csv} | config: {cfg_path}")
    log_f.close()


if __name__ == "__main__":
    main()
