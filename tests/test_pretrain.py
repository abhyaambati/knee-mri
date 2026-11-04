import json
import math
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import h5py
import pandas as pd
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    PreTrainedTokenizerFast, ViTConfig, ViTImageProcessor, ViTModel,
    XLMRobertaConfig, XLMRobertaModel,
)

from dataset import PLANES, VOLUME_SHAPE
from models import ContrastiveModel, MultilingualTextEncoder, MultiViewVisionEncoder
from pretrain import evaluate, make_scheduler, retrieval_metrics, save_best, symmetric_info_nce, train_epoch


def tiny_model(checkpointing=True):
    vision = MultiViewVisionEncoder(
        ViTModel(ViTConfig(
            image_size=8, patch_size=4, hidden_size=32, num_hidden_layers=1,
            num_attention_heads=4, intermediate_size=64,
        ), add_pooling_layer=False),
        num_slices=2, slice_batch_size=2, gradient_checkpointing=checkpointing,
    )
    text = MultilingualTextEncoder(
        XLMRobertaModel(XLMRobertaConfig(
            vocab_size=32, hidden_size=32, num_hidden_layers=1,
            num_attention_heads=4, intermediate_size=64, max_position_embeddings=32,
        ), add_pooling_layer=False), gradient_checkpointing=checkpointing,
    )
    return ContrastiveModel(vision, text)


def batch(valid=(True, True)):
    count = len(valid)
    return {
        "images": torch.rand(count, 3, 2, 3, 8, 8),
        "input_ids": torch.tensor([[0, 4, 2, 1]] * count),
        "attention_mask": torch.tensor([[1, 1, 1, 0]] * count),
        "is_valid": torch.tensor(valid), "report": ["normal"] * count,
    }


def test_embeddings_backward_and_temperature_cap():
    model = tiny_model()
    sample = batch()
    visual, text = model(**{key: sample[key] for key in ("images", "input_ids", "attention_mask")})
    assert visual.shape == text.shape == (2, 256)
    torch.testing.assert_close(visual.norm(dim=-1), torch.ones(2))
    torch.testing.assert_close(text.norm(dim=-1), torch.ones(2))
    assert model.temperature_scale().item() == pytest.approx(1 / 0.07)
    symmetric_info_nce(visual, text, model.temperature_scale()).backward()
    for parameter in (
        model.visual_encoder.depth_position, model.visual_encoder.view_position,
        model.visual_encoder.backbone.embeddings.patch_embeddings.projection.weight,
        model.text_encoder.backbone.embeddings.word_embeddings.weight, model.logit_scale,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    with torch.no_grad():
        model.logit_scale.fill_(1000)
    assert model.temperature_scale().item() == pytest.approx(100, rel=1e-5)


def test_text_pooling_ignores_padding():
    encoder = tiny_model(False).text_encoder.eval()
    ids = torch.tensor([[0, 4, 2, 1]])
    mask = torch.tensor([[1, 1, 1, 0]])
    with torch.no_grad():
        tokens = encoder.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        torch.testing.assert_close(encoder(ids, mask), tokens[:, :3].mean(dim=1))
        torch.testing.assert_close(encoder(ids, mask), encoder(ids[:, :3], mask[:, :3]))


def test_loss_and_full_pool_retrieval():
    embeddings = torch.eye(6)
    scale = torch.tensor(10.0)
    expected = symmetric_info_nce(embeddings, embeddings, scale).item()
    metrics = retrieval_metrics(embeddings, embeddings, scale, chunk_size=2)
    assert metrics["loss"] == pytest.approx(expected)
    for direction in ("image_to_text", "text_to_image"):
        assert metrics[f"{direction}_top1"] == 1
        assert metrics[f"{direction}_top5"] == 1
    reversed_metrics = retrieval_metrics(embeddings, embeddings.roll(1, 0), scale, 2)
    assert reversed_metrics["image_to_text_top1"] == 0
    with pytest.raises(ValueError, match="at least two"):
        symmetric_info_nce(embeddings[:1], embeddings[:1], scale)


def test_training_accumulation_validation_and_handoff(tmp_path):
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = make_scheduler(optimizer, total_steps=2, warmup_fraction=0)
    metrics = train_epoch(
        model, [batch(), batch(), batch(), batch(), batch(), batch((False, True))],
        optimizer, scheduler, torch.device("cpu"), "fp32", accumulation_steps=4,
    )
    assert metrics["updates"] == 2
    assert metrics["pairs"] == 10
    assert metrics["skipped_batches"] == 1
    assert scheduler.last_epoch == 2
    validation = evaluate(model, [batch(), batch((True, False))], torch.device("cpu"), "fp32", 2)
    assert validation["pairs"] == 3
    assert math.isfinite(validation["loss"])
    save_best(model, optimizer, scheduler, 1, validation, Namespace(output_dir=tmp_path))
    restored = tiny_model(False).visual_encoder
    restored.load_state_dict(torch.load(tmp_path / "best_visual_encoder.pt", weights_only=True))
    model.eval()
    restored.eval()
    images = batch()["images"]
    with torch.no_grad():
        torch.testing.assert_close(restored(images), model.visual_encoder(images))
    config = json.loads((tmp_path / "visual_encoder_config.json").read_text())
    assert config["view_order"] == ["sagittal", "coronal", "axial"]
    assert (tmp_path / "best_pretraining.pt").is_file()


def test_warmup_and_cosine_schedule():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    scheduler = make_scheduler(optimizer, total_steps=10)
    assert optimizer.param_groups[0]["lr"] == 0
    rates = []
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates[0] == pytest.approx(1e-4)
    assert rates[-1] == 0
    assert rates == sorted(rates, reverse=True)


def test_offline_training_cli(tmp_path):
    vision_dir, text_dir = tmp_path / "vision", tmp_path / "text"
    ViTModel(ViTConfig(
        image_size=224, patch_size=112, hidden_size=32, num_hidden_layers=1,
        num_attention_heads=4, intermediate_size=64,
    ), add_pooling_layer=False).save_pretrained(vision_dir)
    ViTImageProcessor().save_pretrained(vision_dir)
    XLMRobertaModel(XLMRobertaConfig(
        vocab_size=8, hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
        intermediate_size=64, max_position_embeddings=514,
    ), add_pooling_layer=False).save_pretrained(text_dir)
    backend = Tokenizer(WordLevel({"<unk>": 0, "<pad>": 1, "normal": 2, "knee": 3}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", model_max_length=384,
    ).save_pretrained(text_dir)
    for split in ("train", "val"):
        directory = tmp_path / split
        directory.mkdir()
        studies = [f"{split}.0", f"{split}.1"]
        pd.DataFrame({"StudyInstanceUID": studies, "Report": ["normal knee", "knee"]}).to_csv(
            tmp_path / f"{split}.csv", index=False,
        )
        for study_uid in studies:
            with h5py.File(directory / f"{study_uid}.h5", "w") as handle:
                for plane in PLANES:
                    handle.create_dataset(plane, shape=VOLUME_SHAPE, dtype="float32", fillvalue=0.5)
    output = tmp_path / "checkpoints"
    result = subprocess.run(
        [
            sys.executable, str(Path(__file__).resolve().parents[1] / "pretrain.py"),
            "--train-csv", str(tmp_path / "train.csv"), "--val-csv", str(tmp_path / "val.csv"),
            "--h5-root", str(tmp_path), "--output-dir", str(output),
            "--vision-backbone", str(vision_dir), "--text-backbone", str(text_dir),
            "--device", "cpu", "--precision", "fp32", "--num-workers", "0",
            "--batch-size", "2", "--epochs", "1", "--slice-batch-size", "8",
        ],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Epoch 1" in result.stderr
    assert (output / "tokenizer/tokenizer.json").is_file()
    checkpoint = torch.load(output / "best_pretraining.pt", map_location="cpu", weights_only=True)
    assert checkpoint["metrics"]["pairs"] == 2
    assert checkpoint["args"]["max_length"] == 384
    assert (output / "best_visual_encoder.pt").is_file()