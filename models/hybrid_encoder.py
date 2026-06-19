"""
Hybrid CNN + Transformer encoder for HyMS-Route.

Two parallel branches, token-level outputs (NOT pooled):
  - ViT branch  : DINOv2 ViT-B/14 (HF)  -> patch tokens [B, 256, 768]
  - CNN branch  : ConvNeXt-tiny (tv)    -> 4 stage maps  s1..s4

Returns the raw branch outputs; projection / TokenLearner / scale-embed are done
in the full model (models/hyms_route.py).

Freezing:
  freeze_all()                      -> both backbones frozen
  unfreeze_vit_blocks(n)            -> last n DINOv2 blocks + final norm trainable
"""
import torch
import torch.nn as nn


def _set_requires_grad(module: nn.Module, value: bool):
    for p in module.parameters():
        p.requires_grad_(value)


class HybridEncoder(nn.Module):
    # torchvision convnext `features` Sequential layout:
    #   0 stem, 1 stage1(96), 2 down, 3 stage2(192), 4 down, 5 stage3(384), 6 down, 7 stage4(768)
    _CONVNEXT_STAGE_IDX = [1, 3, 5, 7]

    def __init__(self, vit_name: str = "facebook/dinov2-base",
                 cnn_name: str = "convnext_tiny", device="cpu"):
        super().__init__()
        from transformers import AutoModel
        import torchvision.models as tvm

        self.vit = AutoModel.from_pretrained(vit_name)
        cnn_fn = getattr(tvm, cnn_name)
        self.cnn = cnn_fn(weights="DEFAULT")
        self.cnn_features = self.cnn.features      # Sequential

        # channel dims of the 4 stages (convnext_tiny): 96,192,384,768
        self.cnn_stage_dims = self._infer_stage_dims()
        self.vit_dim = self.vit.config.hidden_size  # 768

        self.to(device)

    @torch.no_grad()
    def _infer_stage_dims(self):
        dims, x = [], torch.zeros(1, 3, 224, 224)
        for i, layer in enumerate(self.cnn_features):
            x = layer(x)
            if i in self._CONVNEXT_STAGE_IDX:
                dims.append(x.shape[1])
        return dims                                 # [96, 192, 384, 768]

    # ── freezing ──────────────────────────────────────────────────────────
    def freeze_all(self):
        _set_requires_grad(self.vit, False)
        _set_requires_grad(self.cnn, False)

    def unfreeze_vit_blocks(self, n_blocks: int = 2):
        if n_blocks <= 0:
            return
        for layer in self.vit.encoder.layer[-n_blocks:]:
            _set_requires_grad(layer, True)
        _set_requires_grad(self.vit.layernorm, True)

    def trainable_backbone_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ── forward ─────────────────────────────────────────────────────────────
    def forward(self, imgs: torch.Tensor):
        # ViT patch tokens (drop CLS at index 0)
        vit_out = self.vit(pixel_values=imgs).last_hidden_state   # [B, 1+P, 768]
        vit_tokens = vit_out[:, 1:, :]                            # [B, P, 768]

        # CNN 4 stage maps
        cnn_maps, x = [], imgs
        for i, layer in enumerate(self.cnn_features):
            x = layer(x)
            if i in self._CONVNEXT_STAGE_IDX:
                cnn_maps.append(x)                                # [B, C_s, H, W]

        return vit_tokens, cnn_maps
