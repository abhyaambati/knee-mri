"""Stage 3 single-device contrastive pretraining.

Example:
    python pretrain.py --h5-root /opt/scratchspace/aambat3/knee-mri-rrg/processed_h5

Defaults: BF16 CUDA, 16 pairs/microbatch, four accumulation steps (nominal
update size 64), 384 text tokens, AdamW and 10% warmup/cosine decay. Negatives
are within each microbatch, not across accumulated microbatches. Invalid scans
and blank reports are excluded. Fewer than two valid pairs skip a microbatch.
Validation uses the full split as the retrieval candidate pool, with chunked
similarity computation. CPU use requires --precision fp32 --device cpu.

best_visual_encoder.pt is a plain MultiViewVisionEncoder state_dict for Stage
4; best_pretraining.pt additionally stores both encoders, projections, optimizer,
scheduler, configuration, metrics, and backbone configs for reconstruction.
"""

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
from torch.nn import functional as F
from transformers import AutoTokenizer

from dataset import get_dataloader
from models import ContrastiveModel, TEXT_BACKBONE, VISION_BACKBONE

LOGGER = logging.getLogger(__name__)


def symmetric_info_nce(visual, text, scale):
    if visual.ndim != 2 or visual.shape != text.shape or len(visual) < 2:
        raise ValueError("InfoNCE requires at least two matching image/text embeddings")
    logits = scale.float() * (visual.float() @ text.float().T)
    targets = torch.arange(len(visual), device=visual.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2


def make_scheduler(optimizer, total_steps, warmup_fraction=0.1):
    if total_steps < 1 or not 0 <= warmup_fraction < 1:
        raise ValueError("Need positive total_steps and warmup_fraction in [0, 1)")
    warmup = math.ceil(total_steps * warmup_fraction)

    def multiplier(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def precision_context(device, precision):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16) if precision == "bf16" else nullcontext()


def prepare_batch(batch, device):
    valid = batch["is_valid"].bool() & torch.tensor([bool(report.strip()) for report in batch["report"]])
    return {
        key: batch[key][valid].to(device, non_blocking=True)
        for key in ("images", "input_ids", "attention_mask")
    }


def train_epoch(model, loader, optimizer, scheduler, device, precision="bf16", accumulation_steps=4):
    """Average gradients over actual microbatches, including partial windows."""
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    pending = updates = pairs = skipped = 0
    loss_sum = 0.0

    def update(count):
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(count)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            model.logit_scale.clamp_(max=math.log(100))
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    for batch in loader:
        inputs = prepare_batch(batch, device)
        count = len(inputs["images"])
        if count < 2:
            skipped += 1
            continue
        with precision_context(device, precision):
            visual, text = model(**inputs)
        loss = symmetric_info_nce(visual, text, model.temperature_scale())
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        loss_sum += loss.detach().item() * count
        pairs += count
        pending += 1
        if pending == accumulation_steps:
            update(pending)
            updates += 1
            pending = 0
    if pending:
        update(pending)
        updates += 1
    if not updates:
        raise RuntimeError("No trainable batches: need at least two valid scans with nonblank reports per batch")
    return {"loss": loss_sum / pairs, "pairs": pairs, "updates": updates, "skipped_batches": skipped}


@torch.no_grad()
def retrieval_metrics(visual, text, scale, chunk_size=256):
    """Exact full-split symmetric loss and bidirectional recall, without NxN storage."""
    if visual.shape != text.shape or len(visual) < 2 or chunk_size < 1:
        raise ValueError("Retrieval requires at least two paired embeddings and a positive chunk size")
    result = {}
    losses = []
    for name, queries, candidates in (("image_to_text", visual, text), ("text_to_image", text, visual)):
        total_loss = 0.0
        hits = {1: 0, 5: 0}
        for start in range(0, len(queries), chunk_size):
            query = queries[start:start + chunk_size].float()
            logits = scale * (query @ candidates.float().T)
            targets = torch.arange(start, start + len(query), device=logits.device)
            total_loss += F.cross_entropy(logits, targets, reduction="sum").item()
            ranked = logits.topk(min(5, len(candidates)), dim=1).indices
            for top in hits:
                hits[top] += (ranked[:, :min(top, len(candidates))] == targets[:, None]).any(dim=1).sum().item()
        losses.append(total_loss / len(queries))
        result.update({f"{name}_top{top}": hits[top] / len(queries) for top in hits})
    result["loss"] = sum(losses) / 2
    result["pairs"] = len(visual)
    return result


@torch.no_grad()
def evaluate(model, loader, device, precision="bf16", chunk_size=256):
    model.eval()
    visual_batches, text_batches = [], []
    for batch in loader:
        inputs = prepare_batch(batch, device)
        if not len(inputs["images"]):
            continue
        with precision_context(device, precision):
            visual, text = model(**inputs)
        visual_batches.append(visual.float().cpu())
        text_batches.append(text.float().cpu())
    if not visual_batches:
        raise RuntimeError("Validation contains no valid scan/report pairs")
    return retrieval_metrics(
        torch.cat(visual_batches), torch.cat(text_batches),
        model.temperature_scale().cpu(), chunk_size,
    )


def atomic_save(payload, path):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_best(model, optimizer, scheduler, epoch, metrics, args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_save(model.visual_encoder.state_dict(), output / "best_visual_encoder.pt")
    atomic_save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "epoch": epoch, "metrics": metrics,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "vision_config": model.visual_encoder.backbone.config.to_dict(),
        "text_config": model.text_encoder.backbone.config.to_dict(),
    }, output / "best_pretraining.pt")
    metadata = {
        "vision_config": model.visual_encoder.backbone.config.to_dict(),
        "num_slices": model.visual_encoder.num_slices,
        "num_views": model.visual_encoder.num_views,
        "num_heads": model.visual_encoder.depth_attention.num_heads,
        "slice_batch_size": model.visual_encoder.slice_batch_size,
        "image_mean": model.visual_encoder.image_mean.flatten().tolist(),
        "image_std": model.visual_encoder.image_std.flatten().tolist(),
        "view_order": ["sagittal", "coronal", "axial"],
    }
    (output / "visual_encoder_config.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-csv", type=Path, default=root / "Split/train.csv")
    parser.add_argument("--val-csv", type=Path, default=root / "Split/val.csv")
    parser.add_argument("--h5-root", type=Path, default=root / "processed_h5", help="Root containing train/ and val/")
    parser.add_argument("--train-h5-dir", type=Path, help="Override for a flat or custom training HDF5 directory")
    parser.add_argument("--val-h5-dir", type=Path, help="Override for a flat or custom validation HDF5 directory")
    parser.add_argument("--output-dir", type=Path, default=root / "checkpoints/stage3")
    parser.add_argument("--vision-backbone", default=VISION_BACKBONE)
    parser.add_argument("--text-backbone", default=TEXT_BACKBONE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--slice-batch-size", type=int, default=32)
    parser.add_argument("--retrieval-chunk-size", type=int, default=256)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.batch_size < 2 or args.accumulation_steps < 1 or args.epochs < 1:
        parser.error("batch-size >= 2, accumulation-steps >= 1 and epochs >= 1 are required")
    if not 1 <= args.max_length <= 384:
        parser.error("max-length must be between 1 and 384")
    if args.num_workers < 0 or args.slice_batch_size < 1 or args.retrieval_chunk_size < 1:
        parser.error("Worker count must be nonnegative and chunk sizes positive")
    if args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.warmup_fraction < 1:
        parser.error("Need positive lr, nonnegative weight-decay and warmup-fraction in [0, 1)")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; for CPU debugging use --device cpu --precision fp32")
    if args.precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        parser.error("BF16 requires a supported CUDA device; use --precision fp32 otherwise")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.text_backbone)
    loaders = []
    for split, csv_path, override in (
        ("train", args.train_csv, args.train_h5_dir), ("val", args.val_csv, args.val_h5_dir)
    ):
        h5_dir = override if override is not None else args.h5_root / split
        if not h5_dir.is_dir():
            parser.error(f"HDF5 directory does not exist: {h5_dir}")
        loader = get_dataloader(
            csv_path, h5_dir, tokenizer, args.batch_size, shuffle=split == "train",
            num_workers=args.num_workers, max_length=args.max_length, pin_memory=device.type == "cuda",
        )
        if len(loader.dataset) < 2:
            parser.error(f"{split} split needs at least two studies")
        if loader.dataset.data["StudyInstanceUID"].duplicated().any():
            parser.error(f"{split} split contains duplicate study IDs, which invalidate diagonal-only contrastive targets")
        loaders.append(loader)
    if set(loaders[0].dataset.data["StudyInstanceUID"]) & set(loaders[1].dataset.data["StudyInstanceUID"]):
        parser.error("Training and validation splits overlap")
    model = ContrastiveModel.from_pretrained(
        args.vision_backbone, args.text_backbone, args.slice_batch_size,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(loaders[0]) / args.accumulation_steps) * args.epochs
    scheduler = make_scheduler(optimizer, total_steps, args.warmup_fraction)
    LOGGER.info("Microbatch=%d accumulation=%d nominal update=%d; negatives remain within each microbatch",
                args.batch_size, args.accumulation_steps, args.batch_size * args.accumulation_steps)
    best_loss = float("inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.output_dir / "tokenizer")
    for epoch in range(1, args.epochs + 1):
        training = train_epoch(model, loaders[0], optimizer, scheduler, device, args.precision, args.accumulation_steps)
        validation = evaluate(model, loaders[1], device, args.precision, args.retrieval_chunk_size)
        LOGGER.info("Epoch %d train=%s validation=%s", epoch, json.dumps(training), json.dumps(validation))
        if validation["loss"] < best_loss:
            save_best(model, optimizer, scheduler, epoch, validation, args)
            best_loss = validation["loss"]
            LOGGER.info("Saved best visual encoder to %s", args.output_dir / "best_visual_encoder.pt")


if __name__ == "__main__":
    main()