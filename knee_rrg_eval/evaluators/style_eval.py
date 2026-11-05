"""Style fidelity from L to L*.

Target labels may be generic templates (bullet / narrative) or institutional
site names (jefferson, dubrava, samsun, ...). A transfer succeeds when the
rewrite matches that site's template cues and sits closer to the site centroid
than to competing styles in Bio_ClinicalBERT space.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from knee_rrg_eval.style_transfer import (
    DEFAULT_MAPPED,
    load_site_exemplars,
    site_template_match,
)

LOGGER = logging.getLogger(__name__)

BULLET_STYLE = "bullet_findings_impression"
NARRATIVE_STYLE = "narrative_block"

_FINDINGS = re.compile(
    r"(?im)^\s*(findings|results|resultados|befund|bevindingen|bulgular|hallazgos)\s*:",
)
_IMPRESSION = re.compile(
    r"(?im)^\s*(impression|impresi[oó]n|beurteilung|conclusie|sonu[cç]|conclusi[oó]n)\s*:",
)
_BULLET = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+\S")

GENERIC_EXEMPLARS: Dict[str, List[str]] = {
    BULLET_STYLE: [
        "FINDINGS:\n- Medial meniscus tear.\n- ACL intact.\n\nIMPRESSION:\n- Medial meniscus tear.",
        "FINDINGS:\n- No ligament tear.\n- Small joint effusion.\n\nIMPRESSION:\n- Small effusion.",
        "FINDINGS:\n- Patellar cartilage defect.\n- Lateral meniscus intact.\n\nIMPRESSION:\n- Patellar chondromalacia.",
    ],
    NARRATIVE_STYLE: [
        "The anterior cruciate ligament is intact. There is a tear of the posterior horn of the medial meniscus. A small joint effusion is present.",
        "The cruciate and collateral ligaments are intact. The menisci are intact. Patellar cartilage is normal and there is no joint effusion.",
        "Mild bone marrow edema is present in the medial femoral condyle. The ligaments and menisci are otherwise intact.",
    ],
}


def section_flags(text: str) -> Dict[str, object]:
    report = "" if text is None else str(text)
    return {
        "has_findings": _FINDINGS.search(report) is not None,
        "has_impression": _IMPRESSION.search(report) is not None,
        "bullet_count": len(_BULLET.findall(report)),
    }


def template_matches(text: str, target_style: str) -> bool:
    """Header and formatting check for a target style or site label."""
    report = "" if text is None else str(text)
    if not report.strip():
        return False
    if target_style == BULLET_STYLE:
        findings = _FINDINGS.search(report) is not None
        impression = _IMPRESSION.search(report) is not None
        bullets = len(_BULLET.findall(report))
        return findings and impression and bullets >= 1
    if target_style == NARRATIVE_STYLE:
        return len(_BULLET.findall(report)) == 0 and len(report.split()) >= 8
    return site_template_match(report, target_style)


class StyleEvaluator:
    """Score style-transfer fidelity against named target templates or sites."""

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        mapped_csv: Path | str = DEFAULT_MAPPED,
        site_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.device = device
        self.model_name = model_name
        self.mapped_csv = Path(mapped_csv)
        self.site_names = list(site_names) if site_names is not None else None
        self._encoder = None
        self._centroids: Optional[Dict[str, List[float]]] = None
        self._exemplars: Optional[Dict[str, List[str]]] = None

    def evaluate(self, candidate: str, target_style: str) -> Dict[str, object]:
        return self.evaluate_batch([candidate], [target_style])[0]

    def evaluate_batch(self, candidates: Sequence[str], target_styles: Sequence[str]) -> List[Dict[str, object]]:
        vectors = self._embed([text if str(text).strip() else " " for text in candidates])
        centroids = self._style_centroids(target_styles)
        rows: List[Dict[str, object]] = []
        for candidate, target, vector in zip(candidates, target_styles, vectors):
            distances = {
                style: _cosine_distance(vector, centroid) for style, centroid in centroids.items()
            }
            nearest = min(distances, key=distances.get) if distances else ""
            template_ok = template_matches(candidate, target)
            distance = distances.get(target)
            # Site transfers succeed when the template cues match and the rewrite
            # is closer to the target site than to a competing site, when available.
            if target in (BULLET_STYLE, NARRATIVE_STYLE):
                success = bool(template_ok and nearest == target and str(candidate).strip())
            else:
                competitors = [style for style in distances if style != target]
                better_than_others = True
                if competitors and distance is not None:
                    better_than_others = distance <= min(distances[style] for style in competitors)
                success = bool(template_ok and better_than_others and str(candidate).strip())
            rows.append(
                {
                    "target_style": target,
                    "template_match": template_ok,
                    "nearest_style": nearest,
                    "style_distance": distance,
                    "style_similarity": None if distance is None else max(0.0, 1.0 - distance),
                    "style_transfer_success": success,
                }
            )
        return rows

    def _load_exemplars(self) -> Dict[str, List[str]]:
        if self._exemplars is not None:
            return self._exemplars
        exemplars = dict(GENERIC_EXEMPLARS)
        try:
            site_styles = load_site_exemplars(self.mapped_csv, per_site=3)
            for site_name, style in site_styles.items():
                if self.site_names is not None and site_name not in self.site_names:
                    continue
                if style.exemplars:
                    exemplars[site_name] = style.exemplars
        except Exception:
            LOGGER.exception("Could not load site exemplars from %s", self.mapped_csv)
        self._exemplars = exemplars
        return exemplars

    def _style_centroids(self, needed: Sequence[str]) -> Dict[str, List[float]]:
        exemplars = self._load_exemplars()
        if self._centroids is None:
            self._centroids = {}
        missing = [style for style in set(needed) | set(exemplars) if style not in self._centroids]
        for style in missing:
            texts = exemplars.get(style)
            if not texts:
                continue
            vectors = self._embed(texts)
            width = len(vectors[0])
            mean = [sum(vector[index] for vector in vectors) / len(vectors) for index in range(width)]
            self._centroids[style] = mean
        return {style: self._centroids[style] for style in self._centroids if style in needed or style in exemplars}

    def _embed(self, texts: Sequence[str]) -> List[List[float]]:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self._encoder is None:
            LOGGER.info("Loading style encoder %s on %s", self.model_name, self.device)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if getattr(tokenizer, "model_max_length", 512) > 512 or not isinstance(
                getattr(tokenizer, "model_max_length", 512), int
            ):
                tokenizer.model_max_length = 512
            model = AutoModel.from_pretrained(self.model_name)
            model.to(self.device)
            model.eval()
            self._encoder = (tokenizer, model)
        tokenizer, model = self._encoder
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled.detach().float().cpu().tolist()


def _cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    similarity = max(-1.0, min(1.0, dot / (left_norm * right_norm)))
    return 1.0 - similarity
