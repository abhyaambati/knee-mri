"""Phase 1 Stage-4 test-set report generation + export.

Loads the active Phase-1 best checkpoint (projector + LoRA + frozen Stage-3
vision), generates a report for every test study, writes CSV/XLSX, and logs
corpus BLEU-1..4 and ROUGE-L.

Example:
  python evaluate_phase1_test.py \\
    --checkpoint-dir /workspace/knee-mri-rrg/checkpoints/stage4/best \\
    --test-csv /root/knee-mri/Split/test.csv \\
    --h5-dir /workspace/knee-mri-rrg/processed_h5/test \\
    --output-dir /workspace/knee-mri-rrg/results
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from tqdm import tqdm

from dataset import KneeMRIDataset
from rrg_models import DEFAULT_STAGE3, load_rrg_checkpoint

LOGGER = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = Path("/workspace/knee-mri-rrg/checkpoints/stage4/best")
DEFAULT_TEST_CSV = Path(__file__).resolve().parent / "Split" / "test.csv"
DEFAULT_H5_DIR = Path("/workspace/knee-mri-rrg/processed_h5/test")
DEFAULT_OUTPUT_DIR = Path("/workspace/knee-mri-rrg/results")


@torch.inference_mode()
def generate_batch(model, images, tokenizer, device, *, max_new_tokens, num_beams, repetition_penalty):
    images = images.to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        token_ids = model.generate(
            images,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.batch_decode(token_ids, skip_special_tokens=True)


def collate_eval(samples: list[dict]) -> dict:
    return {
        "images": torch.stack([sample["images"] for sample in samples]),
        "StudyInstanceUID": [sample["StudyInstanceUID"] for sample in samples],
        "reference_report": [sample["report"] for sample in samples],
        "is_valid": torch.tensor([sample["is_valid"] for sample in samples], dtype=torch.bool),
    }


def compute_metrics(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    if len(references) != len(hypotheses) or not references:
        raise ValueError("Need nonempty equal-length reference and hypothesis lists")
    metrics = {}
    for order in range(1, 5):
        score = BLEU(max_ngram_order=order, effective_order=True).corpus_score(hypotheses, [references])
        metrics[f"bleu_{order}"] = float(score.score)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_l = [
        scorer.score(reference, hypothesis)["rougeL"].fmeasure
        for reference, hypothesis in zip(references, hypotheses)
    ]
    metrics["rouge_l"] = float(sum(rouge_l) / len(rouge_l))
    return metrics


def run_inference(args) -> pd.DataFrame:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --device cpu")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 requires a supported CUDA device")

    model, tokenizer = load_rrg_checkpoint(
        args.checkpoint_dir,
        stage3_dir=args.stage3_dir,
        device=device,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        slice_batch_size=args.slice_batch_size,
    )
    model.eval()
    model.language_model.config.use_cache = True

    dataset = KneeMRIDataset(args.test_csv, args.h5_dir, tokenizer=None)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_eval,
        pin_memory=device.type == "cuda",
        **({"prefetch_factor": 1} if args.num_workers > 0 else {}),
    )
    if len(dataset) != args.expected_n:
        LOGGER.warning("Expected %d test studies, found %d", args.expected_n, len(dataset))

    rows = []
    progress = tqdm(loader, desc="phase1-test", dynamic_ncols=True)
    for batch in progress:
        valid = batch["is_valid"].bool()
        if not valid.all():
            invalid = [uid for uid, ok in zip(batch["StudyInstanceUID"], valid.tolist()) if not ok]
            raise RuntimeError(f"Invalid / missing HDF5 for studies: {invalid}")
        generated = generate_batch(
            model,
            batch["images"],
            tokenizer,
            device,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            repetition_penalty=args.repetition_penalty,
        )
        for uid, reference, hypothesis in zip(
            batch["StudyInstanceUID"], batch["reference_report"], generated
        ):
            rows.append(
                {
                    "StudyInstanceUID": uid,
                    "reference_report": reference,
                    "generated_report_phase1": hypothesis,
                }
            )
        progress.set_postfix(n=len(rows))

    return pd.DataFrame(rows, columns=["StudyInstanceUID", "reference_report", "generated_report_phase1"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT,
                        help="Phase-1 best dir with lora/, rrg_projector.pt, tokenizer/")
    parser.add_argument("--stage3-dir", type=Path, default=DEFAULT_STAGE3)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--h5-dir", type=Path, default=DEFAULT_H5_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--slice-batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-n", type=int, default=434)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.batch_size < 1 or args.num_beams < 1 or args.max_new_tokens < 1:
        parser.error("batch-size, num-beams, and max-new-tokens must be positive")
    if args.repetition_penalty <= 0:
        parser.error("repetition-penalty must be positive")

    frame = run_inference(args)
    if len(frame) != args.expected_n:
        LOGGER.warning("Wrote %d rows; expected %d", len(frame), args.expected_n)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "phase1_test_predictions.csv"
    xlsx_path = args.output_dir / "phase1_test_predictions.xlsx"
    frame.to_csv(csv_path, index=False)
    frame.to_excel(xlsx_path, index=False)

    metrics = compute_metrics(
        frame["reference_report"].tolist(),
        frame["generated_report_phase1"].tolist(),
    )
    metrics_path = args.output_dir / "phase1_test_metrics.json"
    metrics_path.write_text(json.dumps({"n": len(frame), **metrics}, indent=2) + "\n")

    LOGGER.info("Saved %d predictions to %s and %s", len(frame), csv_path, xlsx_path)
    LOGGER.info(
        "Metrics n=%d BLEU-1=%.2f BLEU-2=%.2f BLEU-3=%.2f BLEU-4=%.2f ROUGE-L=%.4f",
        len(frame),
        metrics["bleu_1"],
        metrics["bleu_2"],
        metrics["bleu_3"],
        metrics["bleu_4"],
        metrics["rouge_l"],
    )


if __name__ == "__main__":
    main()
