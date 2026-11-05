"""CRIMSON clinical scoring for radiology reports.

Uses the official ``crimson-score`` implementation and the public
``rajpurkarlab/medgemma-4b-it-crimson`` model. The score is in (-1, 1]:
1 is a perfect report, 0 is no better than a normal template, and negative
values mean weighted errors outweigh correct abnormal findings.

The published model was trained on chest X-ray reports. Knee MRI reports are
scored with that same model and prompt.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_CRIMSON_MODEL = "rajpurkarlab/medgemma-4b-it-crimson"
CRIMSON_COLUMNS = (
    "crimson_score",
    "crimson_false_findings",
    "crimson_missing_findings",
    "crimson_attribute_errors",
    "crimson_analysis",
)


def _row_from_result(result: dict | None) -> dict:
    if not result:
        return {
            "crimson_score": float("nan"),
            "crimson_false_findings": 0,
            "crimson_missing_findings": 0,
            "crimson_attribute_errors": 0,
            "crimson_analysis": "",
        }
    counts = result.get("error_counts", {})
    raw = result.get("raw_evaluation", {})
    return {
        "crimson_score": float(result["crimson_score"]),
        "crimson_false_findings": int(counts.get("false_findings", 0)),
        "crimson_missing_findings": int(counts.get("missing_findings", 0)),
        "crimson_attribute_errors": int(counts.get("attribute_errors", 0)),
        "crimson_analysis": json.dumps(raw, ensure_ascii=False),
    }


class CrimsonEvaluator:
    """Batch CRIMSON inference. The MedGemma model loads on first use."""

    def __init__(
        self,
        model_name: str = DEFAULT_CRIMSON_MODEL,
        device: str = "cuda",
        batch_size: int = 2,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._scorer = None

    def load(self):
        if self._scorer is not None:
            return self._scorer
        from CRIMSON.generate_score import CRIMSONScore
        from CRIMSON.utils import resolve_model_for_vllm

        # The Hub index points at shards that were not uploaded. Resolve a
        # local snapshot that keeps the consolidated weight file.
        local_model = resolve_model_for_vllm(self.model_name)
        LOGGER.info("Loading CRIMSON model %s on %s", self.model_name, self.device)
        self._scorer = CRIMSONScore(api="hf", model_name=local_model, device=self.device)
        return self._scorer

    def score_reports(
        self,
        references: list[str],
        candidates: list[str],
        checkpoint_path: str | Path | None = None,
    ) -> pd.DataFrame:
        from tqdm import tqdm

        from CRIMSON.utils import clean_report_text

        if len(references) != len(candidates) or not references:
            raise ValueError("Need nonempty equal-length reference and candidate lists")
        scorer = self.load()
        references = [clean_report_text(str(text)) for text in references]
        candidates = [clean_report_text(str(text)) for text in candidates]
        rows: list[dict] = []
        progress = tqdm(
            range(0, len(references), self.batch_size),
            desc="crimson",
            total=(len(references) + self.batch_size - 1) // self.batch_size,
            dynamic_ncols=True,
        )
        for start in progress:
            stop = start + self.batch_size
            try:
                results = scorer.evaluate_batch(
                    references[start:stop],
                    candidates[start:stop],
                    include_guidelines=False,
                    batch_size=self.batch_size,
                )
            except Exception as exc:
                LOGGER.warning("CRIMSON batch %d failed (%s); scoring one at a time", start, exc)
                results = []
                for reference, candidate in zip(references[start:stop], candidates[start:stop]):
                    try:
                        results.append(
                            scorer.evaluate(
                                reference,
                                candidate,
                                include_guidelines=False,
                            )
                        )
                    except Exception as item_exc:
                        LOGGER.warning("CRIMSON sample %d failed: %s", start + len(results), item_exc)
                        results.append(None)
            rows.extend(_row_from_result(result) for result in results)
            scored = [row["crimson_score"] for row in rows if row["crimson_score"] == row["crimson_score"]]
            mean = sum(scored) / len(scored) if scored else float("nan")
            progress.set_postfix(n=len(rows), crimson=f"{mean:.3f}")
            if checkpoint_path is not None:
                pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
        return pd.DataFrame(rows, columns=list(CRIMSON_COLUMNS))

    def attach(
        self,
        frame: pd.DataFrame,
        reference_column: str = "reference_report",
        hypothesis_column: str = "generated_report_phase1",
        checkpoint_path: str | Path | None = None,
    ) -> pd.DataFrame:
        missing = {reference_column, hypothesis_column} - set(frame.columns)
        if missing:
            raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
        details = self.score_reports(
            frame[reference_column].astype(str).tolist(),
            frame[hypothesis_column].astype(str).tolist(),
            checkpoint_path=checkpoint_path,
        )
        scored = frame.reset_index(drop=True).copy()
        for column in CRIMSON_COLUMNS:
            scored[column] = details[column].to_numpy()
        return scored
