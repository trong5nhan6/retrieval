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
from pytorch_metric_learning.losses import (SupConLoss, TripletMarginLoss,
                                            ProxyAnchorLoss, MultiSimilarityLoss,
                                            NormalizedSoftmaxLoss, ProxyNCALoss,
                                            SoftTripleLoss)
from pytorch_metric_learning.miners import (TripletMarginMiner,
                                            MultiSimilarityMiner)
from losses.center_contrastive import CenterContrastiveLoss
from losses.potential_field import PotentialFieldLoss

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
    # Loss chính: default=None => không ghi đè HCFG/overrides.
    p.add_argument("--loss_type", type=str, default=None,
                   choices=["triplet", "supcon", "ms", "nsoftmax", "proxynca",
                            "softtriple", "proxyanchor", "ccl", "pfml"],
                   help="loss chính trên embedding z")
    p.add_argument("--loss_lr", type=float, default=None,
                   help="LR cho tham số loss (proxy/center); mặc định HCFG.loss_lr")
    # Proxy-Anchor (song song với triplet). default=None => không ghi đè HCFG/overrides.
    p.add_argument("--use_proxy_anchor", action="store_true", default=None,
                   help="bật Proxy-Anchor loss chạy song song với loss chính trên z")
    p.add_argument("--lambda_proxy", type=float, default=None,
                   help="trọng số Proxy-Anchor (mặc định lấy từ HCFG.lambda_proxy)")
    p.add_argument("--proxy_lr", type=float, default=None,
                   help="LR riêng cho proxy (mặc định lấy từ HCFG.proxy_lr)")
    # PFML-specific hyperparameters (only used when --loss_type pfml)
    p.add_argument("--pf_alpha", type=float, default=None,
                   help="PFML power-law exponent α (paper best range [3,6]; 0=no decay)")
    p.add_argument("--pf_delta", type=float, default=None,
                   help="PFML flat-zone radius δ (paper range [0.1,0.3])")
    p.add_argument("--pf_lambda_rep", type=float, default=None,
                   help="PFML repulsion weight λ_rep (default 1.0)")
    p.add_argument("--pf_ppc", type=int, default=None,
                   help="PFML proxies-per-class M (overrides per-dataset default)")
    p.add_argument("--seed", type=int, default=HCFG.seed)
    p.add_argument("--eval_every", type=int, default=HCFG.eval_every,
                   help="run test every N epochs (default from config: HCFG.eval_every)")
    p.add_argument("--run_id", type=str, default="",
                   help="custom run identifier (default: timestamp YYYYMMDD_HHMMSS)")
    p.add_argument("--eval_routerank", action="store_true", default=None,
                   help="bật RouteRank khi eval (mặc định TẮT)")
    return p.parse_args()


def evaluate(model, loaders, dataset, device):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()                # defrag before eval (avoid Stage-2 OOM)
    rk = HCFG.recall_k_for(dataset)            # CUB/Cars 1/2/4/8 · In-Shop 1/10/20/30
    use_rr = HCFG.use_moe and HCFG.eval_routerank   # cần MoE (rho) + bật eval_routerank
    # Per-dataset RouteRank params (SOP/In-Shop use a smaller rr_topk/rr_beta).
    # dataclasses.replace gives a copy with the overrides — HCFG itself is untouched.
    rr_over = HCFG.rr_for(dataset)
    cfg_eval = dataclasses.replace(HCFG, **rr_over) if rr_over else HCFG
    if dataset in ("cub", "cars", "sop"):
        return evaluate_self(model, loaders["test"], device, cfg_eval,
                             use_routerank=use_rr, recall_k=rk)
    return evaluate_query_gallery(model, loaders["query"], loaders["gallery"],
                                  device, cfg_eval, use_routerank=use_rr, recall_k=rk)


def build_embed_loss(cfg, num_classes, embed_dim, device, dataset: str = ""):
    """Main loss on the retrieval embedding z. Returns (loss_fn, miner).

    One main loss, selected by cfg.loss_type. Some losses keep LEARNABLE params
    (proxies/centers) — nsoftmax, proxynca, softtriple, proxyanchor, ccl, pfml — and
    are moved to `device`; their params are added to the optimizer at cfg.loss_lr
    (see make_optim). All losses share the (embeddings, labels, indices_tuple)
    call signature so the training loop is loss-agnostic."""
    t = cfg.loss_type
    if t == "triplet":
        miner = (TripletMarginMiner(margin=cfg.triplet_margin,
                                    type_of_triplets="semihard")
                 if cfg.triplet_miner else None)
        return TripletMarginLoss(margin=cfg.triplet_margin), miner
    if t == "supcon":
        return SupConLoss(temperature=cfg.temperature), None
    if t == "ms":
        loss = MultiSimilarityLoss(alpha=cfg.ms_alpha, beta=cfg.ms_beta, base=cfg.ms_base)
        return loss, MultiSimilarityMiner(epsilon=cfg.ms_miner_eps)
    if t == "nsoftmax":
        return NormalizedSoftmaxLoss(num_classes=num_classes, embedding_size=embed_dim,
                                     temperature=cfg.nsoftmax_temp).to(device), None
    if t == "proxynca":
        return ProxyNCALoss(num_classes=num_classes, embedding_size=embed_dim,
                            softmax_scale=cfg.proxynca_scale).to(device), None
    if t == "softtriple":
        return SoftTripleLoss(num_classes=num_classes, embedding_size=embed_dim,
                              centers_per_class=cfg.softtriple_centers, la=cfg.softtriple_la,
                              gamma=cfg.softtriple_gamma, margin=cfg.softtriple_margin
                              ).to(device), None
    if t == "proxyanchor":
        return ProxyAnchorLoss(num_classes=num_classes, embedding_size=embed_dim,
                               margin=cfg.pa_margin, alpha=cfg.pa_alpha).to(device), None
    if t == "ccl":
        return CenterContrastiveLoss(num_classes=num_classes, embedding_size=embed_dim,
                                     temperature=cfg.ccl_temp, margin=cfg.ccl_margin
                                     ).to(device), None
    if t == "pfml":
        # Per-dataset proxy count override (paper: M=15 CUB/Cars, M=2 SOP/In-Shop)
        ppc = cfg.pf_ppc_per_dataset.get(dataset.lower(), cfg.pf_proxies_per_class)
        return PotentialFieldLoss(num_classes=num_classes, embedding_size=embed_dim,
                                  proxies_per_class=ppc,
                                  alpha=cfg.pf_alpha, delta=cfg.pf_delta,
                                  lambda_rep=cfg.pf_lambda_rep,
                                  use_proxies=cfg.pf_use_proxies).to(device), None
    raise ValueError(f"unknown loss_type: {t!r} "
                     "(triplet|supcon|ms|nsoftmax|proxynca|softtriple|proxyanchor|ccl|pfml)")


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
    # Chỉ ghi đè HCFG khi flag được truyền trên CLI (default=None).
    if args.loss_type is not None:
        HCFG.loss_type = args.loss_type
    if args.loss_lr is not None:
        HCFG.loss_lr = args.loss_lr
    if args.eval_routerank is not None:
        HCFG.eval_routerank = args.eval_routerank
    if args.use_proxy_anchor is not None:
        HCFG.use_proxy_anchor = args.use_proxy_anchor
    if args.lambda_proxy is not None:
        HCFG.lambda_proxy = args.lambda_proxy
    if args.proxy_lr is not None:
        HCFG.proxy_lr = args.proxy_lr
    # PFML overrides (only meaningful when loss_type=pfml)
    if args.pf_alpha is not None:
        HCFG.pf_alpha = args.pf_alpha
    if args.pf_delta is not None:
        HCFG.pf_delta = args.pf_delta
    if args.pf_lambda_rep is not None:
        HCFG.pf_lambda_rep = args.pf_lambda_rep
    if args.pf_ppc is not None:
        # Override both the global default and all per-dataset values
        HCFG.pf_proxies_per_class = args.pf_ppc
        HCFG.pf_ppc_per_dataset = {}
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
    raw_model = model      # keep reference before DataParallel wrap
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        log(f"DataParallel: using {torch.cuda.device_count()} GPUs")

    n_head = sum(p.numel() for p in raw_model.head_parameters() if p.requires_grad)
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
    log(f"  embed loss     : {HCFG.loss_type} | lambda_route {HCFG.lambda_route}")
    if HCFG.loss_type == "pfml":
        ppc_eff = HCFG.pf_ppc_per_dataset.get(args.dataset.lower(), HCFG.pf_proxies_per_class)
        log(f"  pfml           : α={HCFG.pf_alpha} δ={HCFG.pf_delta} "
            f"λ_rep={HCFG.pf_lambda_rep} M={ppc_eff} proxies/class "
            f"(total proxies: {n_train_classes * ppc_eff:,})")
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

    # MỘT loss chính trên z (chọn bằng HCFG.loss_type). Một số loss giữ tham số
    # học được (proxy/center) -> sẽ được đưa vào optimizer ở make_optim.
    embed_loss, miner = build_embed_loss(HCFG, n_train_classes, HCFG.embed_dim, device,
                                         dataset=args.dataset)
    route_loss = RoutingConsistencyLoss(temperature=HCFG.route_temperature)
    loss_params = list(embed_loss.parameters())        # rỗng với triplet/ms/supcon
    if loss_params:
        n_lp = sum(p.numel() for p in loss_params)
        log(f"  loss params    : {n_lp:,} (proxy/center) | loss_lr={HCFG.loss_lr}")

    # Proxy-Anchor song song (tuỳ chọn) — chạy cùng lúc với loss chính trên z.
    proxy_loss = None
    proxy_params = []
    if HCFG.use_proxy_anchor:
        proxy_loss = ProxyAnchorLoss(num_classes=n_train_classes,
                                     embedding_size=HCFG.embed_dim,
                                     margin=HCFG.pa_margin,
                                     alpha=HCFG.pa_alpha).to(device)
        proxy_params = list(proxy_loss.parameters())
        log(f"  proxy-anchor   : ON | {n_train_classes} proxies x {HCFG.embed_dim}d "
            f"| margin={HCFG.pa_margin} alpha={HCFG.pa_alpha} "
            f"lambda={HCFG.lambda_proxy} proxy_lr={HCFG.proxy_lr}")

    def make_optim(stage):
        if stage == 1:
            groups = [{"params": raw_model.head_parameters(), "lr": args.head_lr}]
        else:
            groups = [
                {"params": raw_model.head_parameters(), "lr": args.head_lr},
                {"params": encoder.trainable_backbone_parameters(), "lr": args.backbone_lr},
            ]
        if loss_params:                                # proxy/center với LR riêng
            groups.append({"params": loss_params, "lr": HCFG.loss_lr})
        if proxy_params:                               # proxy-anchor song song
            groups.append({"params": proxy_params, "lr": HCFG.proxy_lr})
        return torch.optim.AdamW(groups, weight_decay=HCFG.weight_decay)

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
            loss = sc
            if proxy_loss is not None:
                loss = loss + HCFG.lambda_proxy * proxy_loss(z, labels)
            # Loss phụ routing-consistency (chỉ khi lambda_route>0; mặc định 0).
            if rho is not None and HCFG.lambda_route > 0:
                rt = route_loss(rho, labels)
                loss = loss + HCFG.lambda_route * rt
            else:
                rt = torch.zeros((), device=z.device)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                raw_model.head_parameters() + encoder.trainable_backbone_parameters()
                + loss_params + proxy_params,
                HCFG.grad_clip)
            optim.step()
            tot += loss.item(); tot_sc += sc.item(); tot_rt += rt.item()

        n = len(loaders["train"])
        # γ = local_gate (sức nặng nhánh MoE). None nếu tắt cls_skip.
        gp = getattr(raw_model, "local_gate", None)
        gate = float(gp.detach().cpu()) if gp is not None else None
        # ── TRAIN log every epoch ─────────────────────────────────────────
        train_row = {"epoch": epoch, "stage": stage, "loss_type": HCFG.loss_type,
                     "loss": round(tot / n, 4), "sc": round(tot_sc / n, 4),
                     "route": round(tot_rt / n, 4),
                     "gate": round(gate, 4) if gate is not None else None}
        train_log.append(train_row)
        pd.DataFrame(train_log).to_csv(train_csv, index=False)   # incremental
        gate_str = f" gate={gate:+.4f}" if gate is not None else ""
        rt_str = f" rt={tot_rt/n:.4f}" if HCFG.lambda_route > 0 else ""
        log(f"Ep{epoch:3d}[S{stage}] train loss={tot/n:.4f} "
            f"(sc={tot_sc/n:.4f}{rt_str}){gate_str}")

        if sched is not None:
            sched.step()

        # ── TEST every eval_every epochs (and last) ───────────────────────
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            res = evaluate(model, loaders, args.dataset, device)
            # Model-selection metric: routerank R@1 when available, else base R@1.
            r1 = res["routerank"]["R@1"] if "routerank" in res else res["base"]["R@1"]
            test_row = {"epoch": epoch, "stage": stage, "loss_type": HCFG.loss_type}
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
                torch.save({"model": raw_model.state_dict(), "epoch": epoch, "R@1": best,
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
