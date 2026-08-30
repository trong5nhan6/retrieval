"""Chẩn đoán Soft-MoE: nó ĐANG THỰC SỰ LÀM GÌ, chứ không chỉ nhìn accuracy.

Trả lời 3 câu mà bảng ablation không trả lời được:
  1. Các gate scalar hội tụ về đâu? (gamma_moe ~ 0 => model tự tắt nhánh MoE)
  2. Dispatch có chọn lọc không, hay Soft-MoE = mean-pool trá hình?
     H(dispatch)/ln(m) -> 1.0 nghĩa là ĐỀU TUYỆT ĐỐI, mọi slot = token trung bình.
  3. Expert có chuyên biệt theo NGUỒN token (ViT / CNN-s2 / CNN-s3) không?
     Đây chính là claim trung tâm của bài — nếu ma trận dưới phẳng thì claim rỗng.

Dùng:
    python tools/diag_moe.py --dataset cub                    # model init
    python tools/diag_moe.py --dataset cub --ckpt results/cub/<run>/best.pt
    python tools/diag_moe.py --dataset cub --no_norm_dispatch  # so với bản cũ

Chạy trước/sau khi sửa để chứng minh fix có tác dụng.
"""
import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import HCFG
from data.dml_dataset import get_dml_loaders
from models.hybrid_encoder import HybridEncoder
from models.hyms_route import HyMSRoute


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cub", choices=["cub", "cars", "inshop", "sop"])
    p.add_argument("--ckpt", default="", help="đường dẫn best.pt (bỏ trống = model init)")
    p.add_argument("--batches", type=int, default=4, help="số batch để lấy thống kê")
    p.add_argument("--no_norm_dispatch", action="store_true",
                   help="tắt chuẩn hoá dispatch (tái hiện hành vi cũ để so sánh)")
    p.add_argument("--random_input", action="store_true",
                   help="dùng ảnh ngẫu nhiên thay vì dataset (không cần data root)")
    return p.parse_args()


def token_sources(model):
    """Nhãn nguồn cho từng token trong X, đúng thứ tự _assemble_tokens ghép."""
    src = []
    if getattr(model, "moe_on_cls", False):
        src.append(("CLS", 1))
    if model.use_vit:
        src.append(("ViT-patch", 256))          # cập nhật bên dưới theo shape thật
    for s in model.cnn_stages:
        src.append((f"CNN-s{s}", HCFG.tokens_per_stage))
    return src


def main():
    args = parse()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.no_norm_dispatch:
        HCFG.moe_norm_dispatch = False
    if not HCFG.use_moe:
        print("[!] HCFG.use_moe=False — bật lên để chẩn đoán MoE.")
        HCFG.use_moe = True

    enc = HybridEncoder(HCFG.vit_name, HCFG.cnn_name, device=dev,
                        use_vit=HCFG.use_vit, use_cnn=HCFG.use_cnn)
    model = HyMSRoute(enc, HCFG).to(dev).eval()

    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck) if isinstance(ck, dict) else ck
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f"[ckpt] {args.ckpt}\n       missing={len(miss)} unexpected={len(unexp)}")
        for k in miss:
            if "softmoe" in k or "scale" in k:
                print(f"       ! thiếu (dùng giá trị init): {k}")
    else:
        print("[ckpt] không có — đang chẩn đoán model ở INIT")

    # ── 1. gate scalar ────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("1. GATE SCALAR ĐÃ HỌC")
    print("=" * 68)
    moe = model.softmoe
    print(f"  gamma_moe            = {moe.gamma.item():+.5f}"
          f"   {'← ~0: model TỰ TẮT nhánh MoE' if abs(moe.gamma.item()) < 1e-2 else ''}")
    scale = moe.log_scale.exp().item()
    print(f"  dispatch scale       = {scale:.3f}  (log_scale={moe.log_scale.item():+.4f})")
    print(f"    init sqrt(d)={HCFG.token_dim ** 0.5:.2f} | vùng chọn lọc bắt đầu ~50")
    print(f"  norm_dispatch        = {moe.norm_dispatch}")
    if getattr(model, "local_gate", None) is not None:
        print(f"  local_gate           = {model.local_gate.item():+.5f}")
    if hasattr(model.pool, "p"):
        print(f"  GeM p                = {model.pool.p.item():.4f}")
    print(f"  |phi| cột trung bình = {moe.phi.norm(dim=0).mean().item():.4f}")

    # ── lấy dữ liệu ───────────────────────────────────────────────────────
    if args.random_input:
        batches = [torch.randn(8, 3, HCFG.image_size, HCFG.image_size) * 0.5
                   for _ in range(args.batches)]
    else:
        loaders = get_dml_loaders(args.dataset, HCFG)
        key = "test" if "test" in loaders else "query"
        batches = []
        for i, (imgs, _) in enumerate(loaders[key]):
            if i >= args.batches:
                break
            batches.append(imgs)

    combines, rms_by_src, m_total = [], None, None
    with torch.no_grad():
        for imgs in batches:
            imgs = imgs.to(dev)
            vit_tokens, cls, cnn_maps = enc(imgs)
            X = model._assemble_tokens(vit_tokens, cnn_maps)
            if model.moe_on_cls:
                cls_in = (model.vit_proj(cls) + model.cls_type).unsqueeze(1)
                X = torch.cat([cls_in, X], dim=1)
            m_total = X.shape[1]

            # RMS/token theo nguồn, tại đúng điểm concat
            if rms_by_src is None:
                rms_by_src = []
                off = 0
                if model.moe_on_cls:
                    rms_by_src.append(("CLS", X[:, :1]))
                    off = 1
                if model.use_vit:
                    n = vit_tokens.shape[1]
                    rms_by_src.append(("ViT-patch", X[:, off:off + n])); off += n
                for s in model.cnn_stages:
                    n = HCFG.tokens_per_stage
                    rms_by_src.append((f"CNN-s{s}", X[:, off:off + n])); off += n

            Xn = moe.norm(X)
            if moe.norm_dispatch:
                lg = moe.log_scale.exp() * torch.einsum(
                    'bmd,ds->bms', F.normalize(Xn, dim=-1), F.normalize(moe.phi, dim=0))
            else:
                lg = torch.einsum('bmd,ds->bms', Xn, moe.phi)
            combines.append((lg.softmax(1).cpu(), lg.softmax(2).cpu(), Xn.cpu(), lg.cpu()))

    # ── 2. scale mỗi nhánh tại điểm concat ────────────────────────────────
    print("\n" + "=" * 68)
    print("2. SCALE TOKEN MỖI NHÁNH (tại điểm concat trong _assemble_tokens)")
    print("=" * 68)
    for name, t in rms_by_src:
        r = t.pow(2).mean(-1).sqrt()
        print(f"  {name:12s} RMS/token = {r.mean():.4f}  "
              f"(min {r.min():.4f}  max {r.max():.4f})")
    se = model.scale_embed
    print(f"  scale_embed RMS/nguồn = "
          f"{[round(v, 4) for v in se.pow(2).mean(-1).sqrt().tolist()]}")
    print(f"    -> nhãn nguồn bằng ~{se.pow(2).mean().sqrt().item():.1%} biên độ token")

    # ── 3. dispatch có chọn lọc không ─────────────────────────────────────
    print("\n" + "=" * 68)
    print("3. DISPATCH — Soft-MoE có chọn lọc, hay là mean-pool?")
    print("=" * 68)
    disp = torch.cat([c[0] for c in combines], 0)     # [N, m, S] softmax trên TOKEN
    Xn_all = torch.cat([c[2] for c in combines], 0)
    lg_all = torch.cat([c[3] for c in combines], 0)
    H = -(disp * (disp + 1e-12).log()).sum(1).mean()
    Hn = (H / math.log(m_total)).item()
    print(f"  m = {m_total} token, S = {moe.num_slots} slot, uniform = {1/m_total:.5f}")
    print(f"  logits std          = {lg_all.std():.4f}")
    print(f"  max dispatch weight = {disp.max():.5f}  ({disp.max()*m_total:.1f}x uniform)")
    print(f"  H(dispatch)/ln(m)   = {Hn:.4f}   "
          f"{'← GẦN ĐỀU: Soft-MoE ≈ mean-pool' if Hn > 0.90 else '← có chọn lọc'}")
    slots = torch.einsum('bms,bmd->bsd', disp, Xn_all)
    print(f"  ||slot||/||token||  = "
          f"{(slots.pow(2).mean(-1).sqrt().mean() / Xn_all.pow(2).mean(-1).sqrt().mean()):.4f}"
          f"   (<<1 = dấu hiệu lấy trung bình)")
    sn = F.normalize(slots, dim=-1)
    S = moe.num_slots
    cos = (sn @ sn.transpose(1, 2))[:, ~torch.eye(S, dtype=torch.bool)].mean()
    print(f"  cos(slot_i, slot_j) = {cos:.4f}   (→1.0 = mọi slot giống hệt nhau)")

    # ── 4. expert có chuyên biệt theo nguồn không ─────────────────────────
    print("\n" + "=" * 68)
    print("4. CHUYÊN BIỆT CỦA EXPERT THEO NGUỒN TOKEN  (claim trung tâm của bài)")
    print("=" * 68)
    # khối lượng dispatch mỗi slot hút từ mỗi nhóm token, chuẩn hoá theo slot
    mass = disp.mean(0)                                # [m, S]
    rows, off = [], 0
    names = []
    if model.moe_on_cls:
        names.append(("CLS", 1))
    if model.use_vit:
        names.append(("ViT-patch", rms_by_src[1 if model.moe_on_cls else 0][1].shape[1]))
    for s in model.cnn_stages:
        names.append((f"CNN-s{s}", HCFG.tokens_per_stage))
    for name, n in names:
        rows.append((name, n, mass[off:off + n].sum(0)))   # [S]
        off += n
    tot = sum(r[2] for r in rows)
    hdr = "  " + " " * 13 + "".join(f"  e{e}s{s}" for e in range(moe.n_experts)
                                    for s in range(moe.slots_per_expert))
    print(hdr[:120])
    print(f"  {'nguồn':12s} {'#tok':>5s}   tỉ lệ khối lượng dispatch mỗi slot hút từ nguồn này")
    for name, n, v in rows:
        share = (v / tot)
        bar = "".join(f" {x:5.3f}" for x in share.tolist()[:12])
        print(f"  {name:12s} {n:5d}  {bar}")
    print(f"\n  (nếu MỌI CỘT gần bằng tỉ lệ #token thì expert KHÔNG chuyên biệt "
          f"theo nguồn)")
    frac = torch.stack([r[2] / tot for r in rows])       # [n_src, S]
    base = torch.tensor([r[1] / m_total for r in rows]).unsqueeze(1)
    print(f"  độ lệch khỏi tỉ lệ đều |share - #tok/m| trung bình = "
          f"{(frac - base).abs().mean():.4f}   (0 = hoàn toàn không chuyên biệt)")


if __name__ == "__main__":
    main()
