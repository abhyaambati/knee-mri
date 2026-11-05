"""BERTScore with a clinical encoder.

Default model is ``emilyalsentzer/Bio_ClinicalBERT``. Precision, recall, and F1
are token-level cosine similarities in [0, 1]. The reported BERTScore is F1.
Baseline rescaling is off so the values stay on that scale.
"""

from __future__ import annotations

import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)

# Bio_ClinicalBERT is BERT-base. Its tokenizer config advertises an unbounded
# max length, which overflows the current tokenizers backend.
MAX_SEQUENCE_LENGTH = 512
DEFAULT_BERTSCORE_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
# BERT-base convention used by bert-score: the 9th encoder layer.
DEFAULT_NUM_LAYERS = 9
BERTSCORE_COLUMNS = ("bertscore_precision", "bertscore_recall", "bertscore_f1")


def clip_unit(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def zero_empty_hypotheses(
    precision: list[float],
    recall: list[float],
    f1: list[float],
    hypotheses: list[str],
) -> tuple[list[float], list[float], list[float]]:
    """An empty generated report has no token overlap, so all three scores are 0."""
    for index, hypothesis in enumerate(hypotheses):
        if not str(hypothesis).strip():
            precision[index] = 0.0
            recall[index] = 0.0
            f1[index] = 0.0
    return precision, recall, f1


def _install_length_cap() -> None:
    """Keep bert-score truncation inside the encoder's 512-token window.

    Bio_ClinicalBERT publishes an unbounded ``model_max_length``. bert-score
    passes that value into ``tokenizer.encode``, which overflows the tokenizer
    backend. Cap it at the call that performs the encode.
    """
    import bert_score.utils as bert_utils

    if getattr(bert_utils.sent_encode, "_knee_mri_capped", False):
        return
    original = bert_utils.sent_encode

    def sent_encode(tokenizer, sent):
        limit = getattr(tokenizer, "model_max_length", MAX_SEQUENCE_LENGTH)
        if not isinstance(limit, int) or limit > MAX_SEQUENCE_LENGTH:
            tokenizer.model_max_length = MAX_SEQUENCE_LENGTH
        return original(tokenizer, sent)

    sent_encode._knee_mri_capped = True
    bert_utils.sent_encode = sent_encode


class BertScoreEvaluator:
    """Per-report clinical BERTScore. The encoder loads on first use."""

    def __init__(
        self,
        model_name: str = DEFAULT_BERTSCORE_MODEL,
        device: str = "cuda",
        batch_size: int = 16,
        num_layers: int = DEFAULT_NUM_LAYERS,
    ):
        if batch_size < 1 or num_layers < 1:
            raise ValueError("batch_size and num_layers must be positive")
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.num_layers = num_layers

    def score_reports(self, references: list[str], candidates: list[str]) -> pd.DataFrame:
        if len(references) != len(candidates) or not references:
            raise ValueError("Need nonempty equal-length reference and candidate lists")
        from bert_score import score as bert_score

        _install_length_cap()
        LOGGER.info("BERTScore with %s layer %d on %s", self.model_name, self.num_layers, self.device)
        # bert-score rejects empty strings. Empty hypotheses are zeroed after scoring.
        safe_references = [text if str(text).strip() else "." for text in references]
        safe_candidates = [text if str(text).strip() else "." for text in candidates]
        precision, recall, f1 = bert_score(
            safe_candidates,
            safe_references,
            model_type=self.model_name,
            num_layers=self.num_layers,
            batch_size=self.batch_size,
            device=self.device,
            idf=False,
            rescale_with_baseline=False,
            verbose=True,
        )
        precision_scores = [clip_unit(value) for value in precision]
        recall_scores = [clip_unit(value) for value in recall]
        f1_scores = [clip_unit(value) for value in f1]
        precision_scores, recall_scores, f1_scores = zero_empty_hypotheses(
            precision_scores, recall_scores, f1_scores, candidates
        )
        return pd.DataFrame(
            {
                "bertscore_precision": precision_scores,
                "bertscore_recall": recall_scores,
                "bertscore_f1": f1_scores,
            }
        )

    def attach(
        self,
        frame: pd.DataFrame,
        reference_column: str = "reference_report",
        hypothesis_column: str = "generated_report_phase1",
    ) -> pd.DataFrame:
        missing = {reference_column, hypothesis_column} - set(frame.columns)
        if missing:
            raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
        details = self.score_reports(
            frame[reference_column].astype(str).tolist(),
            frame[hypothesis_column].astype(str).tolist(),
        )
        scored = frame.reset_index(drop=True).copy()
        for column in BERTSCORE_COLUMNS:
            scored[column] = details[column].to_numpy()
        return scored
