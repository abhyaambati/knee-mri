"""Stage 4 report generation: Stage-3 vision encoder + visual prefix + LoRA LM."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, ViTConfig, ViTModel

from models import MultiViewVisionEncoder

DEFAULT_DECODER = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_STAGE3 = Path("/workspace/knee-mri-rrg/checkpoints/stage3")
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load_visual_encoder(
    stage3_dir: str | Path = DEFAULT_STAGE3,
    *,
    checkpoint: str = "best_visual_encoder.pt",
    slice_batch_size: int | None = None,
    gradient_checkpointing: bool = False,
    device: torch.device | str | None = None,
) -> MultiViewVisionEncoder:
    """Rebuild MultiViewVisionEncoder from Stage 3 artifacts."""
    stage3_dir = Path(stage3_dir)
    config = json.loads((stage3_dir / "visual_encoder_config.json").read_text())
    vision_config = config.pop("vision_config")
    config.pop("view_order", None)
    if slice_batch_size is not None:
        config["slice_batch_size"] = slice_batch_size
    config["gradient_checkpointing"] = gradient_checkpointing
    backbone = ViTModel(ViTConfig.from_dict(vision_config), add_pooling_layer=False)
    encoder = MultiViewVisionEncoder(backbone, **config)

    weights_path = stage3_dir / checkpoint
    payload = torch.load(weights_path, map_location="cpu", weights_only=False)
    if checkpoint == "best_pretraining.pt":
        state = payload["model"]
        prefix = "visual_encoder."
        state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    else:
        state = payload
    missing, unexpected = encoder.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Vision load mismatch missing={missing} unexpected={unexpected}")
    if device is not None:
        encoder = encoder.to(device)
    return encoder


class VisualPrefixProjector(nn.Module):
    """Map a study embedding [B, vision_dim] to LM prefix tokens [B, num_tokens, lm_dim]."""

    def __init__(self, vision_dim: int, lm_dim: int, num_tokens: int = 32, dropout: float = 0.1):
        super().__init__()
        if num_tokens < 1:
            raise ValueError("num_tokens must be positive")
        self.num_tokens = num_tokens
        self.lm_dim = lm_dim
        self.norm = nn.LayerNorm(vision_dim)
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, lm_dim * num_tokens),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(lm_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("Expected visual features [B, vision_dim]")
        tokens = self.proj(self.norm(features)).view(features.shape[0], self.num_tokens, self.lm_dim)
        return self.output_norm(tokens)


class ReportGenerationModel(nn.Module):
    """Causal report LM conditioned on multi-view MRI via a visual prefix."""

    def __init__(
        self,
        visual_encoder: MultiViewVisionEncoder,
        language_model: nn.Module,
        projector: VisualPrefixProjector,
        *,
        freeze_vision: bool = True,
    ):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.language_model = language_model
        self.projector = projector
        self.num_prefix_tokens = projector.num_tokens
        self.set_vision_trainable(not freeze_vision)

    def set_vision_trainable(self, trainable: bool) -> None:
        for parameter in self.visual_encoder.parameters():
            parameter.requires_grad = trainable
        self.visual_encoder.train(trainable)

    def vision_parameters(self):
        return (parameter for parameter in self.visual_encoder.parameters() if parameter.requires_grad)

    def adapter_parameters(self):
        for module in (self.projector, self.language_model):
            for parameter in module.parameters():
                if parameter.requires_grad:
                    yield parameter

    def encode_prefix(self, images: torch.Tensor) -> torch.Tensor:
        if any(parameter.requires_grad for parameter in self.visual_encoder.parameters()):
            features = self.visual_encoder(images)
        else:
            with torch.no_grad():
                features = self.visual_encoder(images)
        return self.projector(features)

    def _embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        embed = self.language_model.get_input_embeddings()
        return embed(input_ids)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        prefix = self.encode_prefix(images)
        text_embeds = self._embed_text(input_ids)
        inputs_embeds = torch.cat([prefix, text_embeds], dim=1)
        prefix_mask = attention_mask.new_ones(attention_mask.shape[0], self.num_prefix_tokens)
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        full_labels = None
        if labels is not None:
            ignore = labels.new_full((labels.shape[0], self.num_prefix_tokens), -100)
            full_labels = torch.cat([ignore, labels], dim=1)
        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=full_labels,
        )

    @torch.no_grad()
    def generate(self, images: torch.Tensor, **generate_kwargs):
        self.eval()
        prefix = self.encode_prefix(images)
        batch = prefix.shape[0]
        attention_mask = torch.ones(batch, self.num_prefix_tokens, device=prefix.device, dtype=torch.long)
        return self.language_model.generate(
            inputs_embeds=prefix,
            attention_mask=attention_mask,
            **generate_kwargs,
        )


def build_lora_language_model(
    model_name: str = DEFAULT_DECODER,
    *,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: tuple[str, ...] = LORA_TARGETS,
    gradient_checkpointing: bool = True,
    torch_dtype: torch.dtype | None = torch.bfloat16,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    language_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    if gradient_checkpointing:
        language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        language_model.enable_input_require_grads()
    language_model.config.use_cache = False

    lora = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(target_modules),
    )
    language_model = get_peft_model(language_model, lora)
    return language_model, tokenizer


def build_report_model(
    stage3_dir: str | Path = DEFAULT_STAGE3,
    *,
    decoder_name: str = DEFAULT_DECODER,
    vision_checkpoint: str = "best_visual_encoder.pt",
    freeze_vision: bool = True,
    num_prefix_tokens: int = 32,
    slice_batch_size: int = 32,
    vision_gradient_checkpointing: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    torch_dtype: torch.dtype | None = torch.bfloat16,
) -> tuple[ReportGenerationModel, AutoTokenizer]:
    visual = load_visual_encoder(
        stage3_dir,
        checkpoint=vision_checkpoint,
        slice_batch_size=slice_batch_size,
        gradient_checkpointing=vision_gradient_checkpointing and not freeze_vision,
    )
    language_model, tokenizer = build_lora_language_model(
        decoder_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        torch_dtype=torch_dtype,
    )
    hidden = language_model.config.hidden_size
    projector = VisualPrefixProjector(visual.hidden_size, hidden, num_tokens=num_prefix_tokens)
    model = ReportGenerationModel(visual, language_model, projector, freeze_vision=freeze_vision)
    return model, tokenizer


def save_rrg_checkpoint(model: ReportGenerationModel, tokenizer, path: str | Path, **metadata) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "projector": model.projector.state_dict(),
            "vision_requires_grad": any(p.requires_grad for p in model.visual_encoder.parameters()),
            **metadata,
        },
        path / "rrg_projector.pt",
    )
    if isinstance(model.language_model, PeftModel):
        model.language_model.save_pretrained(path / "lora")
    else:
        model.language_model.save_pretrained(path / "language_model")
    tokenizer.save_pretrained(path / "tokenizer")
    if any(parameter.requires_grad for parameter in model.visual_encoder.parameters()):
        torch.save(model.visual_encoder.state_dict(), path / "visual_encoder.pt")


def load_rrg_checkpoint(
    checkpoint_dir: str | Path,
    *,
    stage3_dir: str | Path = DEFAULT_STAGE3,
    device: torch.device | str | None = None,
    torch_dtype: torch.dtype | None = torch.bfloat16,
    slice_batch_size: int = 32,
    freeze_vision: bool | None = None,
    vision_checkpoint: str | None = None,
    vision_gradient_checkpointing: bool = False,
    for_training: bool = False,
) -> tuple[ReportGenerationModel, AutoTokenizer]:
    """Load a Stage-4 best/ directory (projector + LoRA + optional vision).

    Pass ``freeze_vision`` / ``vision_checkpoint`` to override metadata stored in
    the Phase-1 projector payload (used when starting Phase-2 joint fine-tuning).
    """
    checkpoint_dir = Path(checkpoint_dir)
    payload = torch.load(checkpoint_dir / "rrg_projector.pt", map_location="cpu", weights_only=False)
    decoder = payload.get("decoder", DEFAULT_DECODER)
    resolved_vision_checkpoint = vision_checkpoint or payload.get("vision_checkpoint", "best_visual_encoder.pt")
    resolved_freeze = bool(payload.get("freeze_vision", True)) if freeze_vision is None else freeze_vision
    weight = payload["projector"]["proj.0.weight"]
    vision_dim = weight.shape[1]

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir / "tokenizer", use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    language_model = AutoModelForCausalLM.from_pretrained(
        decoder,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    if for_training:
        language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        language_model.enable_input_require_grads()
        language_model.config.use_cache = False
    else:
        language_model.config.use_cache = True
    language_model = PeftModel.from_pretrained(language_model, checkpoint_dir / "lora")
    if for_training:
        language_model.train()
        for name, parameter in language_model.named_parameters():
            if "lora_" in name:
                parameter.requires_grad = True
    else:
        language_model.eval()

    lm_dim = language_model.config.hidden_size
    num_prefix = weight.shape[0] // lm_dim
    if num_prefix < 1 or weight.shape[0] != lm_dim * num_prefix:
        raise RuntimeError(f"Cannot infer prefix length from projector weight {tuple(weight.shape)}")
    if vision_dim <= 0:
        raise RuntimeError("Invalid projector vision dimension")

    visual = load_visual_encoder(
        stage3_dir,
        checkpoint=resolved_vision_checkpoint,
        slice_batch_size=slice_batch_size,
        gradient_checkpointing=vision_gradient_checkpointing and not resolved_freeze,
    )
    vision_path = checkpoint_dir / "visual_encoder.pt"
    # Prefer checkpoint-local vision only when not explicitly overriding the Stage-3 source.
    if vision_path.is_file() and vision_checkpoint is None:
        missing, unexpected = visual.load_state_dict(
            torch.load(vision_path, map_location="cpu", weights_only=False), strict=True
        )
        if missing or unexpected:
            raise RuntimeError(f"Vision load mismatch missing={missing} unexpected={unexpected}")

    projector = VisualPrefixProjector(visual.hidden_size, lm_dim, num_tokens=num_prefix)
    missing, unexpected = projector.load_state_dict(payload["projector"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Projector load mismatch missing={missing} unexpected={unexpected}")

    model = ReportGenerationModel(visual, language_model, projector, freeze_vision=resolved_freeze)
    if not for_training:
        model.eval()
    if device is not None:
        model = model.to(device)
    return model, tokenizer
