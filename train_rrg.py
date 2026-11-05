"""Stage 4 radiology report generation training.

Phase 1 (default): freeze Stage-3 vision, train visual prefix + LoRA decoder.
Phase 2: --unfreeze-vision --init-from <phase1/best> --vision-checkpoint best_pretraining.pt

Example (Phase 1):
  python train_rrg.py \\
    --stage3-dir /workspace/knee-mri-rrg/checkpoints/stage3 \\
    --h5-root /workspace/knee-mri-rrg/processed_h5 \\
    --output-dir /workspace/knee-mri-rrg/checkpoints/stage4 \\
    --decoder Qwen/Qwen2.5-1.5B-Instruct \\
    --batch-size 1 --accumulation-steps 8 --precision bf16

Example (Phase 2):
  python train_rrg.py \\
    --init-from /workspace/knee-mri-rrg/checkpoints/stage4/best \\
    --unfreeze-vision --vision-checkpoint best_pretraining.pt \\
    --vision-lr 1e-6 --lr 1e-4 \\
    --output-dir /workspace/knee-mri-rrg/checkpoints/stage4-phase2
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from dataset import KneeMRIDataset
from rrg_models import (
    DEFAULT_DECODER,
    DEFAULT_STAGE3,
    build_report_model,
    load_rrg_checkpoint,
    save_rrg_checkpoint,
)

LOGGER = logging.getLogger(__name__)


def precision_context(device: torch.device, precision: str):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def collate_rrg(samples: list[dict], tokenizer, max_length: int) -> dict:
    reports = [sample["report"] for sample in samples]
    encoded = tokenizer(
        reports,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100
    return {
        "images": torch.stack([sample["images"] for sample in samples]),
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
        "is_valid": torch.tensor([sample["is_valid"] for sample in samples], dtype=torch.bool),
        "StudyInstanceUID": [sample["StudyInstanceUID"] for sample in samples],
        "report": reports,
    }


def make_loader(csv_path, h5_dir, tokenizer, batch_size, shuffle, num_workers, max_length, pin_memory):
    dataset = KneeMRIDataset(csv_path, h5_dir, tokenizer=None, max_length=max_length)

    def _collate(samples):
        return collate_rrg(samples, tokenizer, max_length)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=pin_memory,
        **({"prefetch_factor": 1} if num_workers > 0 else {}),
    )


def prepare_batch(batch, device):
    valid = batch["is_valid"].bool() & torch.tensor(
        [bool(report.strip()) for report in batch["report"]], dtype=torch.bool
    )
    if not valid.any():
        return None
    return {
        key: batch[key][valid].to(device, non_blocking=True)
        for key in ("images", "input_ids", "attention_mask", "labels")
    }


def build_optimizer(model, lr: float, vision_lr: float, weight_decay: float):
    vision = [parameter for parameter in model.vision_parameters()]
    adapters = [parameter for parameter in model.adapter_parameters()]
    groups = [{"params": adapters, "lr": lr}]
    if vision:
        groups.append({"params": vision, "lr": vision_lr})
    if not any(group["params"] for group in groups):
        raise RuntimeError("No trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def train_epoch(model, loader, optimizer, scheduler, device, precision, accumulation_steps):
    model.train()
    if not any(parameter.requires_grad for parameter in model.visual_encoder.parameters()):
        model.visual_encoder.eval()
    optimizer.zero_grad(set_to_none=True)
    pending = updates = pairs = skipped = 0
    loss_sum = 0.0
    progress = tqdm(loader, desc="train", leave=False, dynamic_ncols=True)

    def step_optimizer(count):
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(count)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            1.0,
            error_if_nonfinite=True,
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(progress, start=1):
        inputs = prepare_batch(batch, device)
        if inputs is None:
            skipped += 1
            continue
        with precision_context(device, precision):
            outputs = model(**inputs)
            loss = outputs.loss
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError("Non-finite report-generation loss")
        loss.backward()
        count = len(inputs["images"])
        loss_sum += loss.detach().item() * count
        pairs += count
        pending += 1
        if pending == accumulation_steps:
            step_optimizer(pending)
            updates += 1
            pending = 0
        mean_loss = loss_sum / max(pairs, 1)
        progress.set_postfix(loss=f"{mean_loss:.4f}", updates=updates, skipped=skipped)
        if step % 20 == 0 or step == len(loader):
            LOGGER.info(
                "train step %d/%d loss=%.4f updates=%d pairs=%d skipped=%d",
                step, len(loader), mean_loss, updates, pairs, skipped,
            )
    if pending:
        step_optimizer(pending)
        updates += 1
    if not updates:
        raise RuntimeError("No trainable batches this epoch")
    return {"loss": loss_sum / pairs, "pairs": pairs, "updates": updates, "skipped_batches": skipped}


@torch.no_grad()
def evaluate(model, loader, device, precision):
    model.eval()
    loss_sum = pairs = 0
    for batch in tqdm(loader, desc="val", leave=False, dynamic_ncols=True):
        inputs = prepare_batch(batch, device)
        if inputs is None:
            continue
        with precision_context(device, precision):
            loss = model(**inputs).loss
        count = len(inputs["images"])
        loss_sum += loss.item() * count
        pairs += count
    if not pairs:
        raise RuntimeError("Validation has no valid report pairs")
    return {"loss": loss_sum / pairs, "pairs": pairs}


def atomic_json(payload, path: Path):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage3-dir", type=Path, default=DEFAULT_STAGE3)
    parser.add_argument("--vision-checkpoint", default="best_visual_encoder.pt",
                        choices=("best_visual_encoder.pt", "best_pretraining.pt"))
    parser.add_argument("--h5-root", type=Path, default=Path("/workspace/knee-mri-rrg/processed_h5"))
    parser.add_argument("--train-csv", type=Path, default=Path(__file__).resolve().parent / "Split/train.csv")
    parser.add_argument("--val-csv", type=Path, default=Path(__file__).resolve().parent / "Split/val.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/knee-mri-rrg/checkpoints/stage4"))
    parser.add_argument("--decoder", default=DEFAULT_DECODER)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--vision-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--num-prefix-tokens", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--slice-batch-size", type=int, default=32)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--unfreeze-vision", action="store_true",
                        help="Phase 2: train vision with --vision-lr (default freezes vision)")
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="Resume projector+LoRA from a Phase-1 best/ directory (required path for Phase 2)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.batch_size < 1 or args.accumulation_steps < 1 or args.epochs < 1:
        parser.error("batch-size, accumulation-steps and epochs must be positive")
    if args.lr <= 0 or args.vision_lr <= 0 or args.weight_decay < 0:
        parser.error("Need positive learning rates and nonnegative weight decay")
    if not 0 <= args.warmup_fraction < 1:
        parser.error("warmup-fraction must be in [0, 1)")
    if args.init_from is not None and not (args.init_from / "rrg_projector.pt").is_file():
        parser.error(f"--init-from missing rrg_projector.pt: {args.init_from}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu --precision fp32")
    if args.precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        parser.error("BF16 requires a supported CUDA device")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    freeze_vision = not args.unfreeze_vision
    if args.init_from is not None:
        LOGGER.info("Loading Phase checkpoint from %s (freeze_vision=%s)", args.init_from, freeze_vision)
        model, tokenizer = load_rrg_checkpoint(
            args.init_from,
            stage3_dir=args.stage3_dir,
            device=None,
            torch_dtype=dtype,
            slice_batch_size=args.slice_batch_size,
            freeze_vision=freeze_vision,
            vision_checkpoint=args.vision_checkpoint,
            vision_gradient_checkpointing=args.unfreeze_vision,
            for_training=True,
        )
    else:
        model, tokenizer = build_report_model(
            args.stage3_dir,
            decoder_name=args.decoder,
            vision_checkpoint=args.vision_checkpoint,
            freeze_vision=freeze_vision,
            num_prefix_tokens=args.num_prefix_tokens,
            slice_batch_size=args.slice_batch_size,
            vision_gradient_checkpointing=args.unfreeze_vision,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            torch_dtype=dtype,
        )
    model = model.to(device)

    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        args.train_csv, args.h5_root / "train", tokenizer, args.batch_size, True,
        args.num_workers, args.max_length, pin_memory,
    )
    val_loader = make_loader(
        args.val_csv, args.h5_root / "val", tokenizer, args.batch_size, False,
        args.num_workers, args.max_length, pin_memory,
    )
    if len(train_loader.dataset) < 1 or len(val_loader.dataset) < 1:
        parser.error("Train and validation splits must be nonempty")

    optimizer = build_optimizer(model, args.lr, args.vision_lr, args.weight_decay)
    total_steps = math.ceil(len(train_loader) / args.accumulation_steps) * args.epochs
    warmup = max(1, int(total_steps * args.warmup_fraction))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    LOGGER.info(
        "freeze_vision=%s trainable=%s / %s (%.2f%%); microbatch=%d accum=%d",
        freeze_vision, f"{trainable:,}", f"{total:,}", 100 * trainable / max(total, 1),
        args.batch_size, args.accumulation_steps,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        training = train_epoch(
            model, train_loader, optimizer, scheduler, device, args.precision, args.accumulation_steps,
        )
        validation = evaluate(model, val_loader, device, args.precision)
        record = {"epoch": epoch, "train": training, "validation": validation}
        history.append(record)
        LOGGER.info("Epoch %d train=%s validation=%s", epoch, json.dumps(training), json.dumps(validation))
        atomic_json({"history": history, "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}},
                    args.output_dir / "history.json")
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            save_rrg_checkpoint(
                model, tokenizer, args.output_dir / "best",
                epoch=epoch, metrics=validation, decoder=args.decoder,
                vision_checkpoint=args.vision_checkpoint, freeze_vision=freeze_vision,
            )
            LOGGER.info("Saved best Stage 4 checkpoint to %s", args.output_dir / "best")


if __name__ == "__main__":
    main()
