"""Clinical preservation from S to S*.

A rewritten report preserves factuality when the knee status vector still
matches: no status flips, no medial/lateral swaps, no invented pathology, and
no dropped abnormal findings. RadGraph F1, MSK diagnostic F1, and GREEN are
reported alongside that decision.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from knee_rrg_eval.metrics.msk_extractor import MSKExtractor


class FactualityEvaluator:
    """Compare a reference report with its rewritten or generated form."""

    def __init__(self, extractor: Optional[MSKExtractor] = None) -> None:
        self.extractor = extractor or MSKExtractor()

    def evaluate(
        self,
        reference: str,
        candidate: str,
        radgraph_f1: Optional[float] = None,
        green_score: Optional[float] = None,
    ) -> Dict[str, object]:
        comparison = self.extractor.compare(reference or "", candidate or "")
        generated = bool(str(candidate or "").strip())
        preserved = bool(
            generated
            and not comparison.flips
            and not comparison.laterality_errors
            and not comparison.hallucinations
            and not comparison.omissions
        )
        return {
            "msk_precision": comparison.precision,
            "msk_recall": comparison.recall,
            "msk_f1": comparison.f1 if generated else 0.0,
            "radgraph_f1": radgraph_f1,
            "green_score": green_score,
            "status_flips": comparison.flips,
            "laterality_errors": comparison.laterality_errors,
            "hallucinations": comparison.hallucinations,
            "omissions": comparison.omissions,
            "factuality_preserved": preserved,
        }

    def evaluate_batch(
        self,
        references: List[str],
        candidates: List[str],
        radgraph_scores: Optional[List[Optional[float]]] = None,
        green_scores: Optional[List[Optional[float]]] = None,
    ) -> List[Dict[str, object]]:
        radgraph_scores = radgraph_scores or [None] * len(references)
        green_scores = green_scores or [None] * len(references)
        return [
            self.evaluate(reference, candidate, radgraph, green)
            for reference, candidate, radgraph, green in zip(
                references, candidates, radgraph_scores, green_scores
            )
        ]
