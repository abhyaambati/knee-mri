"""Batch evaluator for knee MRI report generation.

Heavy scorers run one family at a time so a 7B GREEN model and RadGraph are
not resident together. ``device="cuda"`` falls back to CPU when CUDA is absent.
"""

from __future__ import annotations

import gc
import logging
from typing import Dict, List, Optional, Sequence

from knee_rrg_eval.evaluators.factuality_eval import FactualityEvaluator
from knee_rrg_eval.evaluators.style_eval import StyleEvaluator
from knee_rrg_eval.metrics.green_eval import GreenReportEvaluator
from knee_rrg_eval.metrics.msk_extractor import MSKExtractor
from knee_rrg_eval.metrics.radgraph_eval import RadGraphEvaluator
from knee_rrg_eval.metrics.text_baselines import TextBaselineScorer

LOGGER = logging.getLogger(__name__)


def resolve_device(device: Optional[str] = None) -> str:
    """Use CUDA when it is requested and a GPU is visible."""
    requested = "cuda" if device is None else device
    if requested.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                return requested
        except ImportError:
            pass
        LOGGER.warning("CUDA is unavailable; evaluation will run on CPU")
        return "cpu"
    return requested


def release_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        return


class KneeMRIReportEvaluator:
    """Score generated knee MRI reports against references and a target style."""

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 4,
        enable_green: bool = True,
        enable_radgraph: bool = True,
        enable_bertscore: bool = True,
        enable_ratescore: bool = True,
        green_model: str = "StanfordAIMI/GREEN-RadLlama2-7b",
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.extractor = MSKExtractor()
        self.factuality = FactualityEvaluator(self.extractor)
        self.style = StyleEvaluator(device=self.device)
        self.text = TextBaselineScorer(
            device=self.device,
            batch_size=batch_size,
            enable_bertscore=enable_bertscore,
            enable_ratescore=enable_ratescore,
        )
        self.radgraph = RadGraphEvaluator(device=self.device, enabled=enable_radgraph)
        self.green = GreenReportEvaluator(
            model_name=green_model,
            device=self.device,
            batch_size=min(batch_size, 2),
            enabled=enable_green,
        )

    def evaluate(self, samples: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        return self.evaluate_batch(samples)

    def evaluate_batch(self, samples: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        if not samples:
            raise ValueError("samples must be nonempty")
        normalized = [_normalize_sample(sample, index) for index, sample in enumerate(samples)]
        references = [sample["reference"] for sample in normalized]
        candidates = [sample["candidate"] for sample in normalized]
        styles = [sample["target_style"] for sample in normalized]

        lexical = self.text.lexical(references, candidates)
        bert_scores = self._safe("BERTScore", lambda: self.text.bertscore(references, candidates))
        release_memory()
        rate_scores = self._safe("RaTEScore", lambda: self.text.ratescore(references, candidates))
        self.text._ratescore = None
        release_memory()
        radgraph_rows = self._safe("RadGraph", lambda: self.radgraph.score(references, candidates))
        self.radgraph._scorer = None
        release_memory()
        green_scores = self._safe("GREEN", lambda: self.green.score(references, candidates))
        release_memory()
        style_rows = self.style.evaluate_batch(candidates, styles)
        self.style._encoder = None
        release_memory()

        radgraph_f1 = _column(radgraph_rows, "radgraph_f1", len(samples))
        factuality_rows = self.factuality.evaluate_batch(references, candidates, radgraph_f1, green_scores)
        msk_corpus = self.extractor.corpus_scores(references, candidates)
        rows = []
        for index, sample in enumerate(normalized):
            fact = factuality_rows[index]
            style = style_rows[index]
            bleu = lexical[index]
            rows.append(
                {
                    "Sample_ID": sample["sample_id"],
                    "Ground_Truth_Report": sample["reference"],
                    "Generated_Report": sample["candidate"],
                    "Target_Style_Label": sample["target_style"],
                    "RadGraph_F1": _number(radgraph_f1[index]),
                    "MSK_Diagnostic_F1": fact["msk_f1"],
                    "GREEN_Score": _number(green_scores[index] if green_scores else None),
                    "BERTScore_BioClinical": _number(bert_scores[index] if bert_scores else None),
                    "RaTEScore": _number(rate_scores[index] if rate_scores else None),
                    "BLEU_4": bleu["bleu_4"],
                    "ROUGE_L": bleu["rouge_l"],
                    "Factuality_Preserved_Flag": fact["factuality_preserved"],
                    "Style_Transfer_Success_Flag": style["style_transfer_success"],
                    "Style_Distance": style["style_distance"],
                    "Style_Similarity": style["style_similarity"],
                    "Status_Flips": _join(fact["status_flips"]),
                    "Laterality_Errors": _join(fact["laterality_errors"]),
                    "Hallucinations": _join(fact["hallucinations"]),
                    "Omissions": _join(fact["omissions"]),
                    "BLEU_1": bleu["bleu_1"],
                    "BLEU_2": bleu["bleu_2"],
                    "BLEU_3": bleu["bleu_3"],
                    "MSK_Precision": msk_corpus["precision"],
                    "MSK_Recall": msk_corpus["recall"],
                    "MSK_Micro_F1": msk_corpus["micro_f1"],
                    "MSK_Macro_F1": msk_corpus["macro_f1"],
                }
            )
        return rows

    def _safe(self, name: str, function):
        try:
            return function()
        except Exception:
            LOGGER.exception("%s failed; those cells will be blank", name)
            return None


def _normalize_sample(sample: Dict[str, object], index: int) -> Dict[str, str]:
    reference = _first(sample, ("Ground_Truth_Report", "ground_truth", "reference", "reference_report"))
    candidate = _first(sample, ("Generated_Report", "generated", "prediction", "generated_report"))
    target = _first(sample, ("Target_Style_Label", "target_style", "style"))
    sample_id = _first(sample, ("Sample_ID", "sample_id", "id")) or f"sample-{index + 1}"
    return {
        "sample_id": sample_id,
        "reference": "" if reference is None else str(reference),
        "candidate": "" if candidate is None else str(candidate),
        "target_style": "" if target is None else str(target),
    }


def _first(sample: Dict[str, object], keys: Sequence[str]) -> Optional[object]:
    for key in keys:
        if key in sample and sample[key] is not None:
            return sample[key]
    return None


def _column(rows: Optional[List[Dict[str, float]]], key: str, count: int) -> List[Optional[float]]:
    if not rows:
        return [None] * count
    return [row.get(key) for row in rows]


def _number(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


def _join(items: object) -> str:
    if not items:
        return ""
    return "; ".join(str(item) for item in items)
