# SoftMoE-Retrieval (HyMS-Route)

Hybrid **CNN + Transformer** multi-scale encoder → **Soft MoE** fusion →
**two retrieval channels** (`z` embedding + `ρ` routing fingerprint) →
**RouteRank** training-free reranking.

Standalone project (independent of `moe_medir`). Each DML benchmark is trained
**separately** following the standard zero-shot retrieval protocol.

## Idea in one line
The Soft-MoE routing distribution `ρ` is a near-orthogonal similarity channel:
trained class-consistent (`L_route`) and fused with embedding cosine at search
time (`S = cos(z) + β·cos(ρ)`). An MLP head has no routing → cannot do RouteRank,
so the MoE is *necessary*, not just extra capacity.

## Pipeline
```
img ─┬─ DINOv2 ViT-B/14 (frozen)  ── patch tokens [B,256,768] ─ proj ─┐
     └─ ConvNeXt-tiny (4 stages) ─ 1x1 conv + TokenLearner(64) ───────┤  concat → X [B,512,256]
                                                                       ┘
X ─ Soft MoE (dispatch→experts→combine) ─→ Y [B,512,256], C [B,512,32]
   ├─ pool(Y) → z  [B,256]   (retrieval embedding)
   └─ mean_t(C) → route_head → ρ [B,64]  (routing fingerprint)
train:  L = SupCon(z) + λ·SupCon(ρ)
infer:  S = cos(z_q,z_g) + β·cos(ρ_q,ρ_g)  (+ test-time re-routing)
```

## Layout
```
config.py                 all hyperparameters (HCFG)
utils.py                  set_seed
models/
  token_learner.py        query-attention token reducer (fixed 64 tokens/stage)
  softmoe.py              Soft MoE (no collapse, no load-balance)
  hybrid_encoder.py       DINOv2 + ConvNeXt token-level features
  hyms_route.py           full model -> (z, rho, combine)
losses/routing_consistency.py   SupCon on rho (L_route)
eval/routerank.py         RouteRank rerank + Recall@K / mAP@R (self & query-gallery)
data/dml_dataset.py       CUB / Cars / In-Shop loaders + zero-shot split
train.py                  train ONE dataset separately
notebooks/train.ipynb     local / Colab
notebooks/kaggle.ipynb    Kaggle (auto-detect dataset paths)
```

## Install
```
pip install -r requirements.txt
```

## Data layout
```
data/CUB_200_2011/{images.txt, image_class_labels.txt, images/...}
data/cars196/{cars_annos.mat, car_ims/...}
data/inshop/{list_eval_partition.txt, img/...}
```
Set roots in `config.py` → `HCFG.data_roots`.

## Run
```
python train.py --dataset cub  --seed 42
python train.py --dataset cars --seed 42
python train.py --dataset inshop --finetune_blocks 0
```
Reports **base** (embedding only) vs **RouteRank** every `HCFG.eval_every` epochs
(set in `config.py`). Protocol: CUB 100/100, Cars 98/98 class split (self-retrieval);
In-Shop query–gallery.

### Logs & outputs (per run = `hyms_{dataset}_seed{seed}`)
| File | Content |
|---|---|
| `results/train_{run}.csv` | per-epoch train loss (loss / sc / route) — saved every epoch |
| `results/test_{run}.csv` | metrics every `eval_every` epochs (base + routerank: R@k, P@k, R-Precision, mAP@R) |
| `results/logs/{run}.log` | full console log (text) |
| `results/logs/{run}_config.json` | exact config snapshot tied to this run (HCFG + args + timestamp) |
| `results/checkpoints/best_{run}.pt` | best checkpoint (model + embedded config) |

`--eval_every N` overrides the config default for test frequency.

## Ablations (via config)
| Toggle | Effect |
|---|---|
| `HCFG.cnn_stages = [1,2,3]` | drop the low-level s1 stage |
| `HCFG.lambda_route = 0` | remove routing-consistency loss |
| `HCFG.rr_beta = 0` | disable RouteRank (embedding only) |
| `HCFG.rr_reroute = False` | RouteRank fusion without test-time re-routing |

## Notes
- Backbones frozen (only last 2 ViT blocks unfrozen in Stage 2) → light & fast on small data.
- Run ≥3 seeds and report mean ± std (required for a top CV venue).
- Compare reranking against α-QE / k-reciprocal using published numbers.
