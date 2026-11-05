"""RadGraph entity and relation overlap.

The published reward has three levels. ``RadGraph_F1`` is RG_ER, the partial
entity-relation F1 that RRG papers usually report. RadGraph was trained on
chest radiographs, so knee wording outside that entity set lowers the score
even when the MSK statuses match.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

LOGGER = logging.getLogger(__name__)


def _clean(text: Optional[str]) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split())


def _install_encode_plus_compat() -> None:
    """RadGraph's bundled AllenNLP still calls ``encode_plus``.

    Current transformers keeps that behavior on ``_encode_plus`` and no longer
    exposes the old name. Map the public arguments AllenNLP passes.
    """
    from transformers import PreTrainedTokenizerBase

    if getattr(PreTrainedTokenizerBase, "_knee_encode_plus", False):
        return

    def encode_plus(self, text=None, text_pair=None, *args, **kwargs):
        if args:
            text = args[0] if text is None else text
            if len(args) > 1 and text_pair is None:
                text_pair = args[1]
        if "truncation" in kwargs:
            truncation = kwargs.pop("truncation")
            if truncation is True:
                kwargs["truncation_strategy"] = "longest_first"
            elif truncation is False or truncation is None:
                kwargs["truncation_strategy"] = "do_not_truncate"
            else:
                kwargs["truncation_strategy"] = truncation
        if "padding" in kwargs:
            padding = kwargs.pop("padding")
            if padding is True:
                kwargs["padding_strategy"] = "longest"
            elif padding is False or padding is None:
                kwargs["padding_strategy"] = "do_not_pad"
            else:
                kwargs["padding_strategy"] = padding
        return self._encode_plus(text, text_pair, **kwargs)

    PreTrainedTokenizerBase.encode_plus = encode_plus
    PreTrainedTokenizerBase._knee_encode_plus = True

    if not hasattr(PreTrainedTokenizerBase, "build_inputs_with_special_tokens"):

        def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
            cls = [self.cls_token_id] if self.cls_token_id is not None else []
            sep = [self.sep_token_id] if self.sep_token_id is not None else []
            if token_ids_1 is None:
                return cls + list(token_ids_0) + sep
            return cls + list(token_ids_0) + sep + list(token_ids_1) + sep

        PreTrainedTokenizerBase.build_inputs_with_special_tokens = build_inputs_with_special_tokens


class RadGraphEvaluator:
    """Lazy RadGraph-XL F1 scorer."""

    def __init__(self, device: str = "cuda", enabled: bool = True) -> None:
        self.device = device
        self.enabled = enabled
        self._scorer = None

    def score(self, references: Sequence[str], candidates: Sequence[str]) -> List[Dict[str, float]]:
        if len(references) != len(candidates):
            raise ValueError("references and candidates must have the same length")
        if not self.enabled:
            return [
                {
                    "radgraph_simple": None,
                    "radgraph_partial": None,
                    "radgraph_complete": None,
                    "radgraph_f1": None,
                }
                for _ in references
            ]
        scorer = self._load()
        prepared_refs = [text if _clean(text) else "no acute finding" for text in references]
        prepared_hyps = [text if _clean(text) else "no acute finding" for text in candidates]
        LOGGER.info("RadGraph F1 on %s", self.device)
        _mean_reward, reward_list, _hyp_ann, _ref_ann = scorer(hyps=prepared_hyps, refs=prepared_refs)
        simple, partial, complete = _as_levels(reward_list, len(references))
        rows: List[Dict[str, float]] = []
        for index, candidate in enumerate(candidates):
            if not _clean(candidate):
                rows.append(_blank())
                continue
            rows.append(
                {
                    "radgraph_simple": simple[index],
                    "radgraph_partial": partial[index],
                    "radgraph_complete": complete[index],
                    "radgraph_f1": partial[index],
                }
            )
        return rows

    def _load(self):
        if self._scorer is not None:
            return self._scorer
        _install_encode_plus_compat()
        from radgraph import F1RadGraph

        scorer = F1RadGraph(reward_level="all", model_type="radgraph-xl")
        if hasattr(scorer, "to"):
            scorer = scorer.to(self.device)
        self._scorer = scorer
        return self._scorer


def _blank() -> Dict[str, float]:
    return {
        "radgraph_simple": 0.0,
        "radgraph_partial": 0.0,
        "radgraph_complete": 0.0,
        "radgraph_f1": 0.0,
    }


def _as_levels(reward_list, count: int):
    """Normalize RadGraph's per-example reward into three parallel lists."""
    if (
        isinstance(reward_list, (list, tuple))
        and len(reward_list) == 3
        and not isinstance(reward_list[0], (float, int))
    ):
        levels = [list(map(float, level)) for level in reward_list]
        return levels[0], levels[1], levels[2]
    simple: List[float] = []
    partial: List[float] = []
    complete: List[float] = []
    for item in list(reward_list):
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            simple.append(float(item[0]))
            partial.append(float(item[1]))
            complete.append(float(item[2]))
        else:
            value = float(item)
            simple.append(value)
            partial.append(value)
            complete.append(value)
    if len(partial) != count:
        raise RuntimeError(f"RadGraph returned {len(partial)} scores for {count} reports")
    return simple, partial, complete
