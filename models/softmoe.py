"""
Soft Mixture-of-Experts layer (Puigcerver et al., ICLR 2024).

Operates over a token set X [B, m, d]. Each of the S = n_experts * slots_per_expert
slots receives a SOFT (softmax-weighted) combination of all input tokens; each
expert processes its slots; outputs are recombined back into m output tokens via
a second softmax. Fully differentiable, no token dropping, no expert collapse,
no load-balance loss.

RESIDUAL (quan trọng — sửa 2026-08-28)
--------------------------------------
Trong paper, Soft-MoE thay thế MLP *bên trong* một block ViT, nên nó luôn nằm
trong một residual: x = x + SoftMoE(LN(x)). Bản cũ ở đây trả thẳng Y ra, không
có `X +`, và đo được là ở init nó xoá cấu trúc token:

    ||x_t||   H(dispatch)/ln m   cos(Y_i,Y_j)   cos(X_i,X_j)   ||Y||/||X||
      26.8          0.920            +0.622         ~0.000         0.15

tức dispatch gần như đều -> mọi slot ≈ token trung bình -> các token đầu ra gần
trùng nhau, norm bị bóp ~7 lần. Ở đây khối MoE được đưa về đúng dạng residual có
gate (LayerScale/ReZero):

    Y = X + gamma * MoE(LN(X))

gamma = 0 => Y ≡ X, tức nhánh "+MoE" là mở rộng LỒNG NHAU của nhánh "no MoE"
(bằng nhau chính xác, không phải xấp xỉ). gamma khởi tạo NHỎ nhưng KHÁC 0: đo
trên `local_gate` cho thấy scalar gate ở lr=1e-4 gần như bất động (0.5003 ->
0.5214 sau 10 epoch), nên gamma cần LR riêng (cfg.moe_gate_lr) và không chịu
weight decay — xem make_optim trong train.py.

CHUẨN HOÁ DISPATCH (quan trọng — sửa 2026-08-30)
------------------------------------------------
Soft-MoE paper §2.3 ("Normalization") quy định: khi d và m lớn thì phải L2-chuẩn
hoá CẢ X lẫn Phi rồi nhân một scale HỌC ĐƯỢC, nếu không dispatch thoái hoá. Bản
cũ ở đây dùng thẳng `logits = LN(X) @ phi`, và đo trên DINOv2-L + ConvNeXt-B thật
(m = 385 token, S = 8 slot) cho thấy nó rơi đúng vào chế độ thoái hoá đó:

    logits std   H(dispatch)/ln m   max/uniform   cos(slot_i, slot_j)   ||slot||/||token||
       0.944          0.948            7.1x              0.773                0.566

tức mọi slot ~ token trung bình => Soft-MoE = mean-pool + MLP, và nhánh "+MoE"
không thể tách khỏi nhánh "n_experts=1".

Nguyên nhân là số học: |LN(X)_t| = sqrt(d) = 27.7, |phi_s| ~ 1, hướng ngẫu nhiên
=> cos ~ 1/sqrt(d) => logit std ~ 1.0. Softmax trên 385 token với spread 1.0 thì
gần như phẳng. Scale bị KHOÁ CỨNG ở sqrt(d), model không có đường thoát ra.

    scale     10      27.7 (cũ)      50       100
    H/ln m   0.989      0.915      0.738     0.351

Sweep trên cho thấy vùng routing thật sự chọn lọc bắt đầu từ ~50. Nên ở đây:

    logits = exp(log_scale) * <normalize(LN(X)), normalize(phi)>

log_scale khởi tạo ln(sqrt(d)) => Ở INIT phân bố logit TRÙNG KHỚP bản cũ (nested,
không phá gì), nhưng giờ scale học được nên model tự đi ra vùng chọn lọc. Tham số
hoá trên trục LOG vì scalar này cần đi xa: ở lr=1e-2 Adam đi được ~30 đơn vị
tuyệt đối (đủ 27.7->50, chật vật tới 100), còn trên trục log một bước 0.7 là gấp
đôi. log_scale phải nằm trong gate_parameters() để nhận moe_gate_lr và
weight_decay=0 — xem HyMSRoute.gate_parameters.

`norm_dispatch=False` khôi phục hành vi cũ (dùng để lấy dòng ablation
"Soft-MoE naive vs + paper normalization").

forward(X) -> (Y, C)
  Y : [B, m, d]      X + gamma * MoE(LN(X))
  C : [B, m, S]      combine weights (per token, distribution over slots)
                     -> aggregated into the routing fingerprint rho downstream.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Expert(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), 
            nn.LayerNorm(hidden), 
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):                # x: [B, p, d]
        return self.net(x)


class SoftMoE(nn.Module):
    def __init__(self, dim: int, n_experts: int = 8,
                 slots_per_expert: int = 4, hidden: int = 512,
                 gate_init: float = 0.1, norm_dispatch: bool = True,
                 logit_scale_init: float = 0.0):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.slots_per_expert = slots_per_expert
        self.num_slots = n_experts * slots_per_expert

        # LN trước khối MoE (như trong block ViT của paper): định tuyến dựa trên
        # token đã chuẩn hoá, nên entropy dispatch không phụ thuộc scale của X.
        self.norm = nn.LayerNorm(dim)
        # slot parameter Phi: [d, S]
        self.phi = nn.Parameter(torch.randn(dim, self.num_slots) * (dim ** -0.5))
        # Scale HỌC ĐƯỢC cho logits sau khi L2-chuẩn hoá X và phi (paper §2.3).
        # logit_scale_init<=0 => sqrt(d), tức TRÙNG bản cũ ở init.
        self.norm_dispatch = norm_dispatch
        s0 = logit_scale_init if logit_scale_init > 0 else dim ** 0.5
        self.log_scale = nn.Parameter(torch.tensor(math.log(float(s0))))
        self.experts = nn.ModuleList(
            [_Expert(dim, hidden) for _ in range(n_experts)])
        # gamma: gate residual (LayerScale). gamma=0 => lớp này là identity.
        self.gamma = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, X: torch.Tensor):
        # X: [B, m, d]
        B, m, d = X.shape
        Xn = self.norm(X)
        if self.norm_dispatch:
            # <normalize(X), normalize(phi)> in [-1,1], biên độ do log_scale quyết định.
            logits = self.log_scale.exp() * torch.einsum(
                'bmd,ds->bms', F.normalize(Xn, dim=-1), F.normalize(self.phi, dim=0))
        else:
            logits = torch.einsum('bmd,ds->bms', Xn, self.phi)   # [B, m, S] (đường cũ)

        # Dispatch: softmax over tokens (each slot = distribution over tokens)
        dispatch = logits.softmax(dim=1)                     # [B, m, S]
        slots = torch.einsum('bms,bmd->bsd', dispatch, Xn)   # [B, S, d]

        # Experts process their own slots
        slots = slots.view(B, self.n_experts, self.slots_per_expert, d)
        outs = [self.experts[e](slots[:, e]) for e in range(self.n_experts)]
        expert_out = torch.stack(outs, dim=1).reshape(B, self.num_slots, d)  # [B, S, d]

        # Combine: softmax over slots (each token = distribution over slots)
        combine = logits.softmax(dim=2)                      # [B, m, S]
        moe = torch.einsum('bms,bsd->bmd', combine, expert_out)  # [B, m, d]

        return X + self.gamma * moe, combine
