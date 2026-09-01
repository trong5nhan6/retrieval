"""
RouteRank — reranking + evaluation for HyMS-Route.

Fuses two channels:  S = cos(z_q, z_g) + beta * cos(rho_q, rho_g)
with optional test-time re-routing (neighbour expansion on BOTH channels).

Supports two evaluation protocols:
  - self-retrieval (CUB / Cars): query set == gallery set, self excluded.
  - query/gallery (In-Shop):     disjoint query and gallery sets.

Metrics (cosine, percentages): Recall@K, Precision@K, R-Precision, mAP@R.
K values follow each benchmark's SOTA tables (CUB/Cars: 1/2/4/8; In-Shop: 1/10/20/30).
"""
import torch
import torch.nn.functional as F
import numpy as np


# ── feature extraction ──────────────────────────────────────────────────────
@torch.no_grad()
def extract(model, loader, device):
    """Returns Z [N,De], R [N,Dr] (or None if model has no routing head), labels [N]."""
    model.eval()
    Z, Rr, Y = [], [], []
    has_rho = True
    for batch in loader:
        imgs, labels = batch[0], batch[1]
        z, rho, _ = model(imgs.to(device))
        Z.append(z.cpu())
        if rho is None:
            has_rho = False
        else:
            Rr.append(rho.cpu())
        Y.append(labels if isinstance(labels, torch.Tensor) else torch.tensor(labels))
    R = torch.cat(Rr) if has_rho else None
    return torch.cat(Z), R, torch.cat(Y)


# ── core fusion ─────────────────────────────────────────────────────────────
@torch.no_grad()
def _qe(feats, sim_for_nn, top_k, alpha):
    """Query-expansion: refine each row's feature with its top-k neighbours."""
    vals, idx = sim_for_nn.topk(top_k, dim=1)
    w = vals.clamp(min=0).pow(alpha)
    w = w / (w.sum(dim=1, keepdim=True) + 1e-9)            # [N, K]
    nb = (w.unsqueeze(-1) * feats[idx]).sum(dim=1)         # [N, D]
    return F.normalize(feats + nb, dim=-1)


@torch.no_grad()
def routerank_sim(Zq, Rq, Zg, Rg, beta=0.3, top_k=10, alpha=3.0,
                  reroute=True, self_retrieval=False):
    """
    Returns fused similarity S [Nq, Ng].
    For self-retrieval, pass the same tensors for q and g and set self_retrieval=True.
    """
    Sz = Zq @ Zg.T
    if reroute:
        # neighbours found on the gallery side via base cosine
        Sg = Zg @ Zg.T
        if self_retrieval:
            Sg.fill_diagonal_(-1e9)
        Zg = _qe(Zg, Sg, top_k, alpha)
        Rg = _qe(Rg, Sg, top_k, alpha)
        if self_retrieval:
            Zq, Rq = Zg, Rg
        else:
            Sq = Zq @ Zq.T
            Zq = _qe(Zq, Sq, top_k, alpha)
            Rq = _qe(Rq, Sq, top_k, alpha)
        Sz = Zq @ Zg.T

    Sr = Rq @ Rg.T
    S = Sz + beta * Sr
    if self_retrieval:
        S.fill_diagonal_(-1e9)
    return S


# ── metrics from a similarity matrix ────────────────────────────────────────
def _metrics_from_sim(S, q_labels, g_labels, recall_k, exclude_self=False):
    """
    S [Nq,Ng]; returns dict (percentages):
      R@k          Recall@k    — >=1 correct in top-k
      P@k          Precision@k — fraction of top-k that are correct
      R-Precision  precision at R (R = #relevant for the query)
      mAP@R        mean average precision at R
    """
    if exclude_self:
        S = S.clone(); S.fill_diagonal_(-1e9)
    order = S.argsort(dim=1, descending=True)              # [Nq, Ng]
    sorted_labels = g_labels[order]                        # [Nq, Ng]
    is_correct = (sorted_labels == q_labels.unsqueeze(1))  # [Nq, Ng] bool

    out = {}
    for k in recall_k:
        out[f"R@{k}"] = round(is_correct[:, :k].any(dim=1).float().mean().item() * 100, 2)
        out[f"P@{k}"] = round(is_correct[:, :k].float().mean(dim=1).mean().item() * 100, 2)

    # R-Precision and mAP@R (both depend on per-query R = #relevant).
    # In self-retrieval the query's own item shares its label (ranked last via the
    # -inf diagonal); exclude it from the relevant count to match the standard DML protocol.
    R_per = is_correct.sum(dim=1)
    if exclude_self:
        R_per = R_per - 1
    rprec, aps = [], []
    for i in range(is_correct.size(0)):
        R = int(R_per[i].item())
        if R == 0:
            continue
        top = is_correct[i, :R].float()
        rprec.append((top.sum() / R).item())
        cum = top.cumsum(0)
        ranks = torch.arange(1, R + 1, dtype=torch.float32)
        aps.append(((cum / ranks * top).sum() / R).item())
    out["R-Precision"] = round(float(np.mean(rprec)) * 100, 2) if rprec else 0.0
    out["mAP@R"] = round(float(np.mean(aps)) * 100, 2) if aps else 0.0
    return out


# ── memory-safe (chunked) evaluation for very large galleries ────────────────
# A full Ng×Ng similarity + argsort is ~O(Ng²) memory (e.g. SOP test = 60k → 14GB
# for the float matrix alone, 29GB for argsort). For large galleries we instead
# stream query rows in chunks and keep only the top-K neighbours per query, which
# is enough for every metric (R-Precision/mAP@R need top-R, and R ≤ class size).
_LARGE_N = 20000          # switch to the chunked path above this gallery size
_CHUNK = 2048             # query rows processed per block


def _qe_chunked(Zg, Rg, top_k, alpha, chunk, device):
    """Chunked query-expansion (self excluded), matching `_qe` over Zg@Zg.T.

    Neighbours are found on the z-channel (cosine) only, then BOTH Zg and Rg are
    refined with the same top-k weights — identical to the full routerank_sim
    rerouting step, but never materialising the Ng×Ng matrix.
    """
    N = Zg.size(0)
    Zg = Zg.to(device)
    Rg = Rg.to(device) if Rg is not None else None
    Z2 = torch.empty_like(Zg)
    R2 = torch.empty_like(Rg) if Rg is not None else None
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        sg = Zg[s:e] @ Zg.T                                  # [b, N]
        sg[torch.arange(e - s, device=device),
           torch.arange(s, e, device=device)] = -1e9         # exclude self
        vals, idx = sg.topk(top_k, dim=1)                    # [b, k]
        w = vals.clamp(min=0).pow(alpha)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-9)
        Z2[s:e] = F.normalize(Zg[s:e] + (w.unsqueeze(-1) * Zg[idx]).sum(dim=1), dim=-1)
        if Rg is not None:
            R2[s:e] = F.normalize(Rg[s:e] + (w.unsqueeze(-1) * Rg[idx]).sum(dim=1), dim=-1)
    return Z2, R2


def _metrics_chunked(Zq, Rq, Yq, Zg, Rg, Yg, beta, recall_k, mask_self, r_offset,
                     chunk, device):
    """Same metrics as `_metrics_from_sim`, computed from per-query top-K only.

    score = Zq@Zg.T (+ beta·Rq@Rg.T). We gather top-K neighbours per query where
    K = max(max(recall_k), max relevant-per-query) so Recall/Precision@k and
    R-Precision/mAP@R are all exact.

    `mask_self` drops the diagonal (self-retrieval). `r_offset` is subtracted from
    each query's #relevant — kept separate to mirror the existing full-matrix
    paths exactly: base self uses r_offset=1, the routerank channel uses 0 (it
    masks self in ranking but still counts it in R, as the original code does).
    """
    Nq, Ng = Zq.size(0), Zg.size(0)
    Zq, Zg = Zq.to(device), Zg.to(device)
    if Rq is not None:
        Rq, Rg = Rq.to(device), Rg.to(device)
    Yg_dev = Yg.to(device)

    # per-query #relevant from the gallery label distribution
    cmax = int(max(int(Yg.max()), int(Yq.max()))) + 1
    cnt = torch.zeros(cmax, dtype=torch.long)
    cnt.scatter_add_(0, Yg.long(), torch.ones_like(Yg.long()))
    R_per = cnt[Yq.long()] - r_offset                        # [Nq]
    k_max = int(max(max(recall_k), int(R_per.max().item()), 1))
    k_max = min(k_max, Ng)

    is_correct = torch.empty(Nq, k_max, dtype=torch.bool)
    for s in range(0, Nq, chunk):
        e = min(s + chunk, Nq)
        sim = Zq[s:e] @ Zg.T
        if Rq is not None:
            sim = sim + beta * (Rq[s:e] @ Rg.T)
        if mask_self:                                        # query i == gallery i
            sim[torch.arange(e - s, device=device),
                torch.arange(s, e, device=device)] = -1e9
        _, idx = sim.topk(k_max, dim=1)                      # [b, k_max]
        corr = Yg_dev[idx] == Yq[s:e].to(device).unsqueeze(1)
        is_correct[s:e] = corr.cpu()

    out = {}
    for k in recall_k:
        out[f"R@{k}"] = round(is_correct[:, :k].any(dim=1).float().mean().item() * 100, 2)
        out[f"P@{k}"] = round(is_correct[:, :k].float().mean(dim=1).mean().item() * 100, 2)
    rprec, aps = [], []
    for i in range(Nq):
        R = int(R_per[i].item())
        if R <= 0:
            continue
        top = is_correct[i, :R].float()
        rprec.append((top.sum() / R).item())
        cum = top.cumsum(0)
        ranks = torch.arange(1, R + 1, dtype=torch.float32)
        aps.append(((cum / ranks * top).sum() / R).item())
    out["R-Precision"] = round(float(np.mean(rprec)) * 100, 2) if rprec else 0.0
    out["mAP@R"] = round(float(np.mean(aps)) * 100, 2) if aps else 0.0
    return out


# ── high-level evaluation ───────────────────────────────────────────────────
@torch.no_grad()
def evaluate_self(model, loader, device, cfg, use_routerank=True, recall_k=None):
    """CUB / Cars / SOP: query == gallery == test set."""
    rk = recall_k or cfg.recall_k
    Z, R, Y = extract(model, loader, device)

    # Large gallery (e.g. SOP, 60k) -> chunked path to avoid an Ng×Ng blow-up.
    if Z.size(0) > _LARGE_N:
        dev = device
        base = _metrics_chunked(Z, None, Y, Z, None, Y, 0.0, rk,
                                mask_self=True, r_offset=1, chunk=_CHUNK, device=dev)
        if not use_routerank or R is None:
            return {"base": base}
        if cfg.rr_reroute:
            Zg, Rg = _qe_chunked(Z, R, cfg.rr_topk, cfg.rr_alpha, _CHUNK, dev)
        else:
            Zg, Rg = Z, R
        # BUG FIX (self-retrieval R-count): the query's own item is masked out of
        # ranking (mask_self=True) and can NEVER be retrieved, so it must NOT be
        # counted in the relevant total R. r_offset=1 makes R = (#same-label − 1),
        # matching the base path. The old r_offset=0 inflated R by 1, capping
        # R-Precision/mAP@R at (R−1)/R — negligible on CUB/Cars (~30–60 imgs/class)
        # but a large artificial deflation on SOP (~5 imgs/class).
        rr = _metrics_chunked(Zg, Rg, Y, Zg, Rg, Y, cfg.rr_beta, rk,
                              mask_self=True, r_offset=1, chunk=_CHUNK, device=dev)
        return {"base": base, "routerank": rr}

    base = _metrics_from_sim(Z @ Z.T, Y, Y, rk, exclude_self=True)
    if not use_routerank or R is None:      # no routing fingerprint -> base only
        return {"base": base}
    S = routerank_sim(Z, R, Z, R, cfg.rr_beta, cfg.rr_topk, cfg.rr_alpha,
                      cfg.rr_reroute, self_retrieval=True)
    # BUG FIX: exclude_self=True so the query's own item is dropped from the
    # relevant count R (it is already ranked last via the −1e9 diagonal). This
    # matches the base path; the old exclude_self=False over-counted R by 1,
    # deflating R-Precision/mAP@R most on small-class datasets like SOP.
    rr = _metrics_from_sim(S, Y, Y, rk, exclude_self=True)
    return {"base": base, "routerank": rr}


@torch.no_grad()
def evaluate_query_gallery(model, query_loader, gallery_loader, device, cfg,
                           use_routerank=True, recall_k=None):
    """In-Shop: disjoint query / gallery sets."""
    rk = recall_k or cfg.recall_k
    Zq, Rq, Yq = extract(model, query_loader, device)
    Zg, Rg, Yg = extract(model, gallery_loader, device)
    base = _metrics_from_sim(Zq @ Zg.T, Yq, Yg, rk, exclude_self=False)
    if not use_routerank or Rq is None or Rg is None:   # no fingerprint -> base only
        return {"base": base}
    S = routerank_sim(Zq, Rq, Zg, Rg, cfg.rr_beta, cfg.rr_topk, cfg.rr_alpha,
                      cfg.rr_reroute, self_retrieval=False)
    rr = _metrics_from_sim(S, Yq, Yg, rk, exclude_self=False)
    return {"base": base, "routerank": rr}
