"""Stage 3 encoders for ordered multi-view MRI and multilingual reports."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoImageProcessor, ViTModel, XLMRobertaModel

VISION_BACKBONE = "google/vit-base-patch16-224-in21k"
TEXT_BACKBONE = "xlm-roberta-base"


class AttentionBlock(nn.Module):
    """Pre-norm transformer block using PyTorch scaled dot-product attention."""

    def __init__(self, hidden_size=768, num_heads=8, dropout=0.1):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.dropout = dropout
        self.norm1 = nn.LayerNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size), nn.Dropout(dropout),
        )

    def forward(self, tokens):
        batch, length, hidden = tokens.shape
        qkv = self.qkv(self.norm1(tokens)).reshape(
            batch, length, 3, self.num_heads, hidden // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attended = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout if self.training else 0.0
        ).transpose(1, 2).reshape(batch, length, hidden)
        tokens = tokens + F.dropout(self.output(attended), self.dropout, self.training)
        return tokens + self.mlp(self.norm2(tokens))


class MultiViewVisionEncoder(nn.Module):
    """Shared slice ViT -> ordered depth attention -> plane-aware view attention.

    Inputs are [batch, sagittal/coronal/axial, depth, RGB, height, width] in
    [0, 1]. ImageNet/pretrained normalization is applied inside this encoder,
    so the standalone Stage 4 handoff uses exactly the same preprocessing.
    """

    def __init__(
        self, backbone, num_slices=24, num_views=3, num_heads=8,
        slice_batch_size=32, gradient_checkpointing=True,
        image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5),
    ):
        super().__init__()
        if min(num_slices, num_views, slice_batch_size) < 1:
            raise ValueError("Slice, view and slice chunk counts must be positive")
        if len(image_mean) != 3 or len(image_std) != 3 or min(image_std) <= 0:
            raise ValueError("Normalization requires three means and positive standard deviations")
        self.backbone = backbone
        self.hidden_size = backbone.config.hidden_size
        self.num_slices = num_slices
        self.num_views = num_views
        self.slice_batch_size = slice_batch_size
        self.gradient_checkpointing = gradient_checkpointing
        self.register_buffer("image_mean", torch.tensor(image_mean).reshape(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(image_std).reshape(1, 3, 1, 1))
        self.depth_position = nn.Parameter(torch.empty(1, num_slices, self.hidden_size))
        self.view_position = nn.Parameter(torch.empty(1, num_views, self.hidden_size))
        nn.init.trunc_normal_(self.depth_position, std=0.02)
        nn.init.trunc_normal_(self.view_position, std=0.02)
        self.depth_attention = AttentionBlock(self.hidden_size, num_heads)
        self.view_attention = AttentionBlock(self.hidden_size, num_heads)
        self.output_norm = nn.LayerNorm(self.hidden_size)
        if gradient_checkpointing:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def _slice_features(self, slices):
        pixels = (slices - self.image_mean) / self.image_std
        return self.backbone(pixel_values=pixels).last_hidden_state[:, 0]

    def forward(self, images):
        if images.ndim != 6 or tuple(images.shape[1:4]) != (self.num_views, self.num_slices, 3):
            raise ValueError("Expected images [B, views, slices, 3, H, W] matching encoder configuration")
        batch, views, depth, channels, height, width = images.shape
        slices = images.reshape(-1, channels, height, width)
        features = []
        for chunk in slices.split(self.slice_batch_size):
            if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
                features.append(checkpoint(self._slice_features, chunk, use_reentrant=False))
            else:
                features.append(self._slice_features(chunk))
        tokens = torch.cat(features).reshape(batch * views, depth, self.hidden_size)
        tokens = self.depth_attention(tokens + self.depth_position).mean(dim=1)
        tokens = tokens.reshape(batch, views, self.hidden_size) + self.view_position
        return self.output_norm(self.view_attention(tokens).mean(dim=1))


class MultilingualTextEncoder(nn.Module):
    """XLM-R token features pooled with an attention-mask-weighted mean."""

    def __init__(self, backbone, max_length=384, gradient_checkpointing=True):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = backbone.config.hidden_size
        self.max_length = max_length
        if gradient_checkpointing:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def forward(self, input_ids, attention_mask):
        if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
            raise ValueError("Text IDs and attention mask must have matching [B, length] shapes")
        if input_ids.shape[1] > self.max_length:
            raise ValueError(f"Reports must be truncated to {self.max_length} tokens")
        tokens = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        weights = attention_mask.unsqueeze(-1).float()
        return (tokens.float() * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


class ContrastiveModel(nn.Module):
    """Normalized joint embeddings with a learned, capped inverse temperature."""

    def __init__(self, visual_encoder, text_encoder, projection_dim=256):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_encoder.hidden_size, visual_encoder.hidden_size), nn.GELU(),
            nn.Linear(visual_encoder.hidden_size, projection_dim),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_encoder.hidden_size, text_encoder.hidden_size), nn.GELU(),
            nn.Linear(text_encoder.hidden_size, projection_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    @classmethod
    def from_pretrained(
        cls, vision_name=VISION_BACKBONE, text_name=TEXT_BACKBONE,
        slice_batch_size=32, gradient_checkpointing=True,
    ):
        processor = AutoImageProcessor.from_pretrained(vision_name)
        visual = MultiViewVisionEncoder(
            ViTModel.from_pretrained(vision_name, add_pooling_layer=False),
            slice_batch_size=slice_batch_size, gradient_checkpointing=gradient_checkpointing,
            image_mean=processor.image_mean, image_std=processor.image_std,
        )
        text = MultilingualTextEncoder(
            XLMRobertaModel.from_pretrained(text_name, add_pooling_layer=False),
            gradient_checkpointing=gradient_checkpointing,
        )
        return cls(visual, text)

    def temperature_scale(self):
        return self.logit_scale.float().clamp(max=math.log(100)).exp()

    def forward(self, images, input_ids, attention_mask):
        visual = self.visual_projection(self.visual_encoder(images))
        text = self.text_projection(self.text_encoder(input_ids, attention_mask))
        return F.normalize(visual.float(), dim=-1), F.normalize(text.float(), dim=-1)