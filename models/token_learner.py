"""
LocalTokenizer — biến feature map CNN thành SỐ TOKEN CỐ ĐỊNH, GIỮ TÍNH CỤC BỘ.

Khác cách cross-attention (mỗi token = pool có trọng số trên TOÀN BỘ pixel =>
mất liên kết không gian), ở đây mỗi token tương ứng MỘT VÙNG cục bộ của feature
map. Ta adaptive-pool map về lưới g x g (g = round(sqrt(num_tokens))) để số token
cố định giữa các stage khác độ phân giải, rồi phát mỗi ô lưới thành 1 token =>
giữ đúng chi tiết cục bộ mà ConvNeXt sinh ra.

  vào:  [B, C, H, W]   (đã được 1x1 conv chiếu về d=C ở upstream)
  ra:   [B, g*g, d]    (g*g == num_tokens)

Ví dụ s1 (56x56) -> mỗi token gộp một vùng ~7x7 pixel LIỀN KỀ (vẫn cục bộ),
không bị trộn toàn cục như cross-attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalTokenizer(nn.Module):
    def __init__(self, dim: int, num_tokens: int = 64, local_mix: bool = True,
                 **_unused):                      # nuốt num_heads cũ cho tương thích
        super().__init__()
        g = int(round(num_tokens ** 0.5))
        assert g * g == num_tokens, \
            f"num_tokens phải là số chính phương để xếp lưới g x g, nhận {num_tokens}"
        self.grid = g
        self.num_tokens = num_tokens

        # depthwise 3x3: trộn lân cận cục bộ trước khi pool (rẻ, vẫn cục bộ).
        # Đặt local_mix=False để pool thuần (không trộn lân cận).
        self.local = (nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
                      if local_mix else nn.Identity())
        # pointwise 1x1 = "Conv1d biến filter thành token" (mix theo kênh,
        # áp dụng độc lập từng vị trí không gian).
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        x = self.local(feat_map)                  # [B, d, H, W]
        x = F.adaptive_avg_pool2d(x, self.grid)   # [B, d, g, g]  ← pool CỤC BỘ
        x = self.proj(x)                          # [B, d, g, g]
        x = x.flatten(2).transpose(1, 2)          # [B, g*g, d]   mỗi ô = 1 token
        return self.norm(x)                       # [B, num_tokens, dim]
