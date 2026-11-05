"""Local GREEN scoring for knee MRI reports.

This calls the repository GREEN runner, which follows the Stanford-AIMI prompt,
cleanup, and formula for ``StanfordAIMI/GREEN-RadLlama2-7b``:

    GREEN = matched / (matched + significant errors)    if matched > 0 else 0

The packaged ``green-score`` class initializes distributed processing and exits
when it is imported outside that launcher, so inference stays in-process.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import List, Optional, Sequence

LOGGER = logging.getLogger(__name__)

DEFAULT_GREEN_MODEL = "StanfordAIMI/GREEN-RadLlama2-7b"


def _load_repo_green():
    path = Path(__file__).resolve().parents[2] / "metrics" / "green_eval.py"
    if not path.is_file():
        raise FileNotFoundError(f"Repository GREEN implementation is missing: {path}")
    spec = importlib.util.spec_from_file_location("knee_mri_repo_green_eval", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GreenReportEvaluator:
    """Batch GREEN scores in [0, 1]. The language model loads on first use."""

    def __init__(
        self,
        model_name: str = DEFAULT_GREEN_MODEL,
        device: str = "cuda",
        batch_size: int = 2,
        enabled: bool = True,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.enabled = enabled

    def score(self, references: Sequence[str], candidates: Sequence[str]) -> List[Optional[float]]:
        if len(references) != len(candidates):
            raise ValueError("references and candidates must have the same length")
        if not self.enabled:
            return [None] * len(references)
        module = _load_repo_green()
        evaluator = module.GreenEvaluator(self.model_name, device=self.device, batch_size=self.batch_size)
        LOGGER.info("GREEN %s on %s", self.model_name, self.device)
        frame = evaluator.score_reports(list(references), [text if str(text).strip() else "" for text in candidates])
        scores: List[Optional[float]] = []
        for value, candidate in zip(frame["green_score"].tolist(), candidates):
            if not str(candidate).strip():
                scores.append(0.0)
            else:
                scores.append(float(value))
        del evaluator
        return scores
