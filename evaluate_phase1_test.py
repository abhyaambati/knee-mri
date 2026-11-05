"""Phase 1 Stage-4 test-set report generation + export.

Loads the active Phase-1 best checkpoint (projector + LoRA + frozen Stage-3
vision), generates a report for every test study, writes CSV/XLSX, and logs
corpus BLEU-1..4 plus mean ROUGE-1/2/L. Each row stores sentence BLEU and
ROUGE on a 0-1 scale. Pass --score-only to rescore an existing predictions CSV.

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


REPORT_METRIC_COLUMNS = (
    "bleu_1",
    "bleu_2",
    "bleu_3",
    "bleu_4",
    "rouge_1",
    "rouge_2",
    "rouge_l",
)
_BLEU = {
    order: BLEU(max_ngram_order=order, effective_order=True) for order in range(1, 5)
}
_ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)


def _check_pairs(references: list[str], hypotheses: list[str]) -> None:
    if len(references) != len(hypotheses) or not references:
        raise ValueError("Need nonempty equal-length reference and hypothesis lists")


def _unit_interval(score: float) -> float:
    """sacreBLEU reports 0-100. Store BLEU on the same 0-1 scale as ROUGE."""
    return min(1.0, max(0.0, float(score) / 100.0))


def score_report(reference: str, hypothesis: str) -> dict[str, float]:
    """Sentence BLEU-1..4 and ROUGE-1/2/L F1, each in [0, 1]."""
    reference = "" if reference is None else str(reference)
    hypothesis = "" if hypothesis is None else str(hypothesis)
    rouge = _ROUGE.score(reference, hypothesis)
    return {
        "bleu_1": _unit_interval(_BLEU[1].sentence_score(hypothesis, [reference]).score),
        "bleu_2": _unit_interval(_BLEU[2].sentence_score(hypothesis, [reference]).score),
        "bleu_3": _unit_interval(_BLEU[3].sentence_score(hypothesis, [reference]).score),
        "bleu_4": _unit_interval(_BLEU[4].sentence_score(hypothesis, [reference]).score),
        "rouge_1": float(rouge["rouge1"].fmeasure),
        "rouge_2": float(rouge["rouge2"].fmeasure),
        "rouge_l": float(rouge["rougeL"].fmeasure),
    }


def attach_report_metrics(
    frame: pd.DataFrame,
    reference_column: str = "reference_report",
    hypothesis_column: str = "generated_report_phase1",
) -> pd.DataFrame:
    """Add per-report BLEU and ROUGE columns for every generated report."""
    missing = {reference_column, hypothesis_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Predictions are empty")
    scored = frame.copy()
    rows = [
        score_report(reference, hypothesis)
        for reference, hypothesis in zip(
            scored[reference_column].tolist(),
            scored[hypothesis_column].tolist(),
        )
    ]
    for column in REPORT_METRIC_COLUMNS:
        scored[column] = [row[column] for row in rows]
    return scored


def compute_metrics(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """Corpus BLEU-1..4 and mean sentence ROUGE-1/2/L F1, each in [0, 1]."""
    _check_pairs(references, hypotheses)
    metrics = {}
    for order in range(1, 5):
        score = _BLEU[order].corpus_score(hypotheses, [references])
        metrics[f"bleu_{order}"] = _unit_interval(score.score)
    totals = {name: 0.0 for name in ("rouge_1", "rouge_2", "rouge_l")}
    for reference, hypothesis in zip(references, hypotheses):
        row = score_report(reference, hypothesis)
        for name in totals:
            totals[name] += row[name]
    count = float(len(references))
    metrics.update({name: value / count for name, value in totals.items()})
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
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Score an existing predictions CSV instead of generating reports",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "phase1_test_predictions.csv",
        help="CSV with reference_report and generated_report_phase1, used by --score-only",
    )
    parser.add_argument("--green", action="store_true", help="Add per-report GREEN scores from a local model")
    parser.add_argument("--green-model", default="StanfordAIMI/GREEN-RadLlama2-7b")
    parser.add_argument("--green-batch-size", type=int, default=4)
    parser.add_argument(
        "--bertscore",
        action="store_true",
        help="Add per-report BERTScore from emilyalsentzer/Bio_ClinicalBERT",
    )
    parser.add_argument("--bertscore-model", default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--bertscore-batch-size", type=int, default=16)
    parser.add_argument(
        "--crimson",
        action="store_true",
        help="Add per-report CRIMSON scores from rajpurkarlab/medgemma-4b-it-crimson",
    )
    parser.add_argument("--crimson-model", default="rajpurkarlab/medgemma-4b-it-crimson")
    parser.add_argument("--crimson-batch-size", type=int, default=2)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.batch_size < 1 or args.num_beams < 1 or args.max_new_tokens < 1:
        parser.error("batch-size, num-beams, and max-new-tokens must be positive")
    if args.repetition_penalty <= 0:
        parser.error("repetition-penalty must be positive")

    if args.score_only:
        frame = pd.read_csv(args.predictions, dtype=str, keep_default_na=False)
    else:
        frame = run_inference(args)
    if not args.score_only and len(frame) != args.expected_n:
        LOGGER.warning("Wrote %d rows; expected %d", len(frame), args.expected_n)

    frame = attach_report_metrics(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.green:
        if args.green_batch_size < 1:
            parser.error("green-batch-size must be positive")
        from metrics.green_eval import GreenEvaluator

        frame = GreenEvaluator(
            args.green_model,
            device=args.device,
            batch_size=args.green_batch_size,
        ).attach(
            frame,
            checkpoint_path=args.output_dir / "green_partial.csv",
        )
    if args.bertscore:
        if args.bertscore_batch_size < 1:
            parser.error("bertscore-batch-size must be positive")
        from metrics.bertscore_eval import BertScoreEvaluator

        frame = BertScoreEvaluator(
            args.bertscore_model,
            device=args.device,
            batch_size=args.bertscore_batch_size,
        ).attach(frame)
    if args.crimson:
        if args.crimson_batch_size < 1:
            parser.error("crimson-batch-size must be positive")
        from metrics.crimson_eval import CrimsonEvaluator

        frame = CrimsonEvaluator(
            args.crimson_model,
            device=args.device,
            batch_size=args.crimson_batch_size,
        ).attach(
            frame,
            checkpoint_path=args.output_dir / "crimson_partial.csv",
        )
    csv_path = args.output_dir / "phase1_test_predictions.csv"
    xlsx_path = args.output_dir / "phase1_test_predictions.xlsx"
    frame.to_csv(csv_path, index=False)
    frame.to_excel(xlsx_path, index=False)

    metrics = compute_metrics(
        frame["reference_report"].tolist(),
        frame["generated_report_phase1"].tolist(),
    )
    if "green_score" in frame.columns:
        green_scores = frame["green_score"].astype(float)
        metrics["green"] = float(green_scores.mean())
        metrics["green_std"] = float(green_scores.std(ddof=0))
    if "bertscore_f1" in frame.columns:
        metrics["bertscore"] = float(frame["bertscore_f1"].astype(float).mean())
        metrics["bertscore_precision"] = float(frame["bertscore_precision"].astype(float).mean())
        metrics["bertscore_recall"] = float(frame["bertscore_recall"].astype(float).mean())
    if "crimson_score" in frame.columns:
        crimson_scores = pd.to_numeric(frame["crimson_score"], errors="coerce").dropna()
        if len(crimson_scores):
            metrics["crimson"] = float(crimson_scores.mean())
            metrics["crimson_std"] = float(crimson_scores.std(ddof=0))
            metrics["crimson_n"] = int(len(crimson_scores))
    metrics_path = args.output_dir / "phase1_test_metrics.json"
    metrics_path.write_text(json.dumps({"n": len(frame), **metrics}, indent=2) + "\n")
    for partial_name in ("green_partial.csv", "crimson_partial.csv"):
        partial = args.output_dir / partial_name
        if partial.exists():
            partial.unlink()

    LOGGER.info("Saved %d predictions to %s and %s", len(frame), csv_path, xlsx_path)
    LOGGER.info(
        "Metrics n=%d BLEU-1=%.4f BLEU-2=%.4f BLEU-3=%.4f BLEU-4=%.4f ROUGE-1=%.4f ROUGE-2=%.4f ROUGE-L=%.4f%s%s%s",
        len(frame),
        metrics["bleu_1"],
        metrics["bleu_2"],
        metrics["bleu_3"],
        metrics["bleu_4"],
        metrics["rouge_1"],
        metrics["rouge_2"],
        metrics["rouge_l"],
        "" if "green" not in metrics else f" GREEN={metrics['green']:.4f}",
        "" if "bertscore" not in metrics else f" BERTScore={metrics['bertscore']:.4f}",
        "" if "crimson" not in metrics else f" CRIMSON={metrics['crimson']:.4f}",
    )


if __name__ == "__main__":
    main()
