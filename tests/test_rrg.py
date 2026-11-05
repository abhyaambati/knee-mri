import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel, ViTConfig, ViTModel

from models import MultiViewVisionEncoder
from rrg_models import ReportGenerationModel, VisualPrefixProjector, load_visual_encoder
from train_rrg import collate_rrg, prepare_batch


def tiny_vision(num_slices=2, num_views=3):
    backbone = ViTModel(
        ViTConfig(
            image_size=8, patch_size=4, hidden_size=32, num_hidden_layers=1,
            num_attention_heads=4, intermediate_size=64,
        ),
        add_pooling_layer=False,
    )
    return MultiViewVisionEncoder(
        backbone, num_slices=num_slices, num_views=num_views, num_heads=4,
        slice_batch_size=4, gradient_checkpointing=False,
        image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5),
    )


def tiny_lm():
    config = GPT2Config(
        n_layer=1, n_head=2, n_embd=32, vocab_size=64, n_positions=64,
        n_ctx=64, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(config)
    lora = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["c_attn"],
    )
    return get_peft_model(model, lora)


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, texts, truncation=True, max_length=16, padding=True, return_tensors="pt"):
        rows = []
        for text in texts:
            ids = [min(3 + (ord(ch) % 50), 63) for ch in text][: max_length - 1] + [self.eos_token_id]
            rows.append(ids)
        width = max(len(row) for row in rows)
        input_ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row)
            attention_mask[index, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_prefix_shapes_and_freeze():
    vision = tiny_vision()
    lm = tiny_lm()
    projector = VisualPrefixProjector(32, lm.config.hidden_size, num_tokens=4)
    model = ReportGenerationModel(vision, lm, projector, freeze_vision=True)
    assert not any(parameter.requires_grad for parameter in model.visual_encoder.parameters())
    assert any(parameter.requires_grad for parameter in model.adapter_parameters())

    images = torch.rand(2, 3, 2, 3, 8, 8)
    input_ids = torch.randint(0, 64, (2, 5))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    outputs = model(images, input_ids, attention_mask, labels)
    assert outputs.loss.ndim == 0
    assert torch.isfinite(outputs.loss)

    model.set_vision_trainable(True)
    assert any(parameter.requires_grad for parameter in model.vision_parameters())


def test_collate_and_prepare_batch():
    tokenizer = TinyTokenizer()
    samples = [
        {"images": torch.zeros(3, 2, 3, 8, 8), "report": "normal knee", "is_valid": True, "StudyInstanceUID": "a"},
        {"images": torch.ones(3, 2, 3, 8, 8), "report": "", "is_valid": True, "StudyInstanceUID": "b"},
    ]
    batch = collate_rrg(samples, tokenizer, max_length=16)
    assert batch["labels"].shape == batch["input_ids"].shape
    assert (batch["labels"][0] != -100).any()
    prepared = prepare_batch(batch, torch.device("cpu"))
    assert prepared is not None
    assert prepared["images"].shape[0] == 1


def test_load_visual_encoder_from_stage3(tmp_path):
    vision = tiny_vision()
    config = {
        "vision_config": vision.backbone.config.to_dict(),
        "num_slices": 2,
        "num_views": 3,
        "num_heads": 4,
        "slice_batch_size": 4,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "view_order": ["sagittal", "coronal", "axial"],
    }
    (tmp_path / "visual_encoder_config.json").write_text(json.dumps(config))
    torch.save(vision.state_dict(), tmp_path / "best_visual_encoder.pt")
    loaded = load_visual_encoder(tmp_path, gradient_checkpointing=False)
    vision.eval()
    loaded.eval()
    images = torch.rand(1, 3, 2, 3, 8, 8)
    with torch.no_grad():
        expected = vision(images)
        actual = loaded(images)
    assert actual.shape == expected.shape == (1, 32)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
