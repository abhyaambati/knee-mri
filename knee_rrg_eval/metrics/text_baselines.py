"""Lexical and embedding baselines for one report pair.

BLEU uses sentence sacreBLEU on the same 0–1 scale as the Phase-1 evaluator.
ROUGE-L is the F1 from ``rouge-score``. BERTScore uses
``emilyalsentzer/Bio_ClinicalBERT``. RaTEScore is the entity-aware score from
the ``RaTEScore`` package.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

LOGGER = logging.getLogger(__name__)

DEFAULT_BERTSCORE_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
DEFAULT_BERTSCORE_LAYER = 9
MAX_SEQUENCE_LENGTH = 512


def _unit_interval(score: float) -> float:
    return min(1.0, max(0.0, float(score) / 100.0))


def _clean(text: Optional[str]) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split())


def score_bleu(reference: str, candidate: str) -> Dict[str, float]:
    """Sentence BLEU-1..4. An empty candidate is 0 on every order."""
    from sacrebleu.metrics import BLEU

    candidate_text = _clean(candidate)
    reference_text = _clean(reference)
    if not candidate_text or not reference_text:
        return {f"bleu_{order}": 0.0 for order in range(1, 5)}
    scores = {}
    for order in range(1, 5):
        metric = BLEU(max_ngram_order=order, effective_order=True)
        scores[f"bleu_{order}"] = _unit_interval(metric.sentence_score(candidate_text, [reference_text]).score)
    return scores


def score_rouge_l(reference: str, candidate: str) -> float:
    """ROUGE-L F1 in [0, 1]. An empty candidate is 0."""
    from rouge_score import rouge_scorer

    candidate_text = _clean(candidate)
    reference_text = _clean(reference)
    if not candidate_text or not reference_text:
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return float(scorer.score(reference_text, candidate_text)["rougeL"].fmeasure)


def _install_bertscore_length_cap() -> None:
    """Bio_ClinicalBERT advertises an unbounded max length that overflows encode."""
    import bert_score.utils as bert_utils

    if getattr(bert_utils.sent_encode, "_knee_rrg_capped", False):
        return
    original = bert_utils.sent_encode

    def sent_encode(tokenizer, sent):
        limit = getattr(tokenizer, "model_max_length", MAX_SEQUENCE_LENGTH)
        if not isinstance(limit, int) or limit > MAX_SEQUENCE_LENGTH:
            tokenizer.model_max_length = MAX_SEQUENCE_LENGTH
        return original(tokenizer, sent)

    sent_encode._knee_rrg_capped = True
    bert_utils.sent_encode = sent_encode


class TextBaselineScorer:
    """Batch BLEU, ROUGE-L, clinical BERTScore, and RaTEScore."""

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 8,
        bertscore_model: str = DEFAULT_BERTSCORE_MODEL,
        enable_bertscore: bool = True,
        enable_ratescore: bool = True,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.device = device
        self.batch_size = batch_size
        self.bertscore_model = bertscore_model
        self.enable_bertscore = enable_bertscore
        self.enable_ratescore = enable_ratescore
        self._ratescore = None

    def lexical(self, references: Sequence[str], candidates: Sequence[str]) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        for reference, candidate in zip(references, candidates):
            bleu = score_bleu(reference, candidate)
            bleu["rouge_l"] = score_rouge_l(reference, candidate)
            rows.append(bleu)
        return rows

    def bertscore(self, references: Sequence[str], candidates: Sequence[str]) -> List[float]:
        if not self.enable_bertscore:
            return [float("nan")] * len(references)
        from bert_score import score as bert_score

        _install_bertscore_length_cap()
        safe_references = [text if _clean(text) else "." for text in references]
        safe_candidates = [text if _clean(text) else "." for text in candidates]
        LOGGER.info("BERTScore %s on %s", self.bertscore_model, self.device)
        _precision, _recall, f1 = bert_score(
            safe_candidates,
            safe_references,
            model_type=self.bertscore_model,
            num_layers=DEFAULT_BERTSCORE_LAYER,
            batch_size=self.batch_size,
            device=self.device,
            idf=False,
            rescale_with_baseline=False,
            verbose=False,
        )
        scores = [min(1.0, max(0.0, float(value))) for value in f1]
        for index, candidate in enumerate(candidates):
            if not _clean(candidate):
                scores[index] = 0.0
        return scores

    def ratescore(self, references: Sequence[str], candidates: Sequence[str]) -> List[float]:
        if not self.enable_ratescore:
            return [float("nan")] * len(references)
        scorer = self._load_ratescore()
        LOGGER.info("RaTEScore on %s", self.device)
        # Score one pair at a time. The packaged NER regroups sentence spans across
        # a batch and can return fewer scores than reports.
        scores: List[float] = []
        for reference, candidate in zip(references, candidates):
            if not _clean(candidate) or not _clean(reference):
                scores.append(0.0)
                continue
            raw = scorer.compute_score([_clean(candidate)], [_clean(reference)])
            scores.append(float(raw[0]) if raw else 0.0)
        return scores

    def _load_ratescore(self):
        if self._ratescore is not None:
            return self._ratescore
        from RaTEScore import RaTEScore

        self._ratescore = RaTEScore(
            batch_size=self.batch_size,
            use_gpu=self.device.startswith("cuda"),
            affinity_matrix="long",
        )
        return self._ratescore
