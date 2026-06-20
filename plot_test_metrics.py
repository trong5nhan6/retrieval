"""
Plot test-metric trends across evaluation epochs from results/test_{run}.csv
(the CSV that train.py writes every eval_every epochs).

All selected metrics are drawn on ONE figure so you can read the trend at a
glance: each metric is a colour; the `base` channel is a solid line and
`routerank` is dashed. A vertical line marks the Stage-2 (backbone unfreeze)
start, and the best routerank R@1 (the model-selection metric) is annotated.

Usage:
    python plot_test_metrics.py --dataset cub --seed 42
    python plot_test_metrics.py --csv results/test_hyms_cub_seed42.csv
    python plot_test_metrics.py --dataset cub --seed 42 \
        --metrics "R@1,R@2,mAP@R,R-Precision" --channels base,routerank

Output: results/plot_test_{run}.png  (override with --out)
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_METRICS = ["R@1", "R@2", "R@4", "R@8", "R-Precision", "mAP@R"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None, help="path to test_{run}.csv (overrides dataset/seed)")
    p.add_argument("--dataset", default="cub")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--metrics", default=",".join(DEFAULT_METRICS),
                   help="comma-separated metric names to plot")
    p.add_argument("--channels", default="base,routerank",
                   help="comma-separated channels: base and/or routerank")
    p.add_argument("--out", default=None, help="output PNG path")
    return p.parse_args()


def plot_test_metrics(csv_path, out=None, metrics=None, channels=None, title=None):
    """Render test-metric trends from a test_{run}.csv to a PNG. Returns out path."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"test CSV not found: {csv_path}")
    metrics = metrics or list(DEFAULT_METRICS)
    channels = channels or ["base", "routerank"]
    if isinstance(metrics, str):
        metrics = [m.strip() for m in metrics.split(",") if m.strip()]
    if isinstance(channels, str):
        channels = [c.strip() for c in channels.split(",") if c.strip()]

    df = pd.read_csv(csv_path).sort_values("epoch")

    cmap = plt.get_cmap("tab10")
    color = {m: cmap(i % 10) for i, m in enumerate(metrics)}
    style = {"base": "-", "routerank": "--"}
    marker = {"base": "o", "routerank": "s"}

    fig, ax = plt.subplots(figsize=(11, 6.5))
    plotted = 0
    for ch in channels:
        for m in metrics:
            col = f"{ch}.{m}"
            if col not in df.columns or df[col].isna().all():
                continue
            ax.plot(df["epoch"], df[col], style.get(ch, "-"),
                    marker=marker.get(ch, "o"), ms=4, lw=1.8,
                    color=color[m], alpha=0.65 if ch == "base" else 1.0,
                    label=f"{ch} {m}")
            plotted += 1
    if plotted == 0:
        raise SystemExit("No matching metric columns found to plot.")

    # Stage-2 (unfreeze) start marker
    if "stage" in df.columns and (df["stage"] == 2).any():
        s2 = df.loc[df["stage"] == 2, "epoch"].min()
        ax.axvline(s2, color="grey", ls=":", lw=1)
        ax.text(s2, ax.get_ylim()[1], " Stage 2", va="top", ha="left",
                fontsize=8, color="grey")

    # annotate best routerank R@1 (model-selection metric), fall back to base R@1
    sel_col = "routerank.R@1" if "routerank.R@1" in df.columns else "base.R@1"
    if sel_col in df.columns and not df[sel_col].isna().all():
        bi = df[sel_col].idxmax()
        be, bv = df.loc[bi, "epoch"], df.loc[bi, sel_col]
        ax.scatter([be], [bv], s=120, facecolors="none", edgecolors="black", zorder=5)
        ax.annotate(f"best {sel_col}={bv:.2f} @ep{int(be)}",
                    (be, bv), textcoords="offset points", xytext=(8, 8), fontsize=9)

    ax.set_xlabel("epoch")
    ax.set_ylabel("metric (%)")
    ax.set_title(title or "Test-metric trends")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8, loc="lower left")
    fig.tight_layout()

    if out is None:
        out = os.path.splitext(csv_path)[0] + ".png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_train_loss(csv_path, out=None, title=None):
    """Render training-loss curves from a train_{run}.csv to a PNG.

    Plots total loss and its components: sc (embedding loss on z) and route
    (routing-consistency loss on rho). Returns out path."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"train CSV not found: {csv_path}")
    df = pd.read_csv(csv_path).sort_values("epoch")

    series = [("loss", "total loss", "tab:red"),
              ("sc", "sc  (embedding loss)", "tab:blue"),
              ("route", "route  (routing loss)", "tab:green")]

    fig, ax = plt.subplots(figsize=(11, 6))
    for col, lbl, c in series:
        if col in df.columns and not df[col].isna().all():
            ax.plot(df["epoch"], df[col], "-", lw=1.8, color=c, label=lbl)

    if "stage" in df.columns and (df["stage"] == 2).any():
        s2 = df.loc[df["stage"] == 2, "epoch"].min()
        ax.axvline(s2, color="grey", ls=":", lw=1)
        ax.text(s2, ax.get_ylim()[1], " Stage 2", va="top", ha="left",
                fontsize=8, color="grey")

    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss")
    ax.set_title(title or "Training loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()

    if out is None:
        out = os.path.splitext(csv_path)[0] + "_loss.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    args = parse_args()
    run = f"hyms_{args.dataset}_seed{args.seed}"
    csv_path = args.csv or os.path.join(args.results_dir, f"test_{run}.csv")
    out = args.out or os.path.join(args.results_dir, f"plot_test_{run}.png")
    saved = plot_test_metrics(csv_path, out=out, metrics=args.metrics,
                              channels=args.channels, title=f"Test-metric trends — {run}")
    print(f"saved {saved}")

    # also plot the training-loss curve if the matching train CSV exists
    train_csv = os.path.join(args.results_dir, f"train_{run}.csv")
    if os.path.exists(train_csv):
        saved_loss = plot_train_loss(train_csv,
                                     out=os.path.join(args.results_dir, f"plot_train_{run}.png"),
                                     title=f"Training loss — {run}")
        print(f"saved {saved_loss}")


if __name__ == "__main__":
    main()
