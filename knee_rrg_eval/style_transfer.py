"""Site-conditioned rewrite of knee MRI ground-truth reports (L → L*).

Each institutional site has a language and a report template. A causal LM
rewrites the original report into that template while keeping every clinical
finding, laterality, and severity unchanged.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_MAPPED = Path(__file__).resolve().parents[1] / "EDA" / "mapped.csv"

SITE_PROFILES: Dict[str, Dict[str, str]] = {
    "jefferson": {
        "language": "English",
        "template": (
            "US academic template with Exam Type / Indication / TECHNIQUE / "
            "FINDINGS under OSSEOUS STRUCTURES, menisci, ligaments, cartilage, "
            "and soft tissues, then IMPRESSION."
        ),
    },
    "mater_dei": {
        "language": "English",
        "template": (
            "Concise English narrative with Findings: and Conclusion: headers. "
            "Lists ligaments as ACL, PCL, MCL, LCL and covers both menisci."
        ),
    },
    "santiago": {
        "language": "Spanish",
        "template": (
            "Spanish sectioned report with Antecedentes Clínicos / Hallazgos / "
            "Conclusión. Uses menisco medial/lateral and ligamentos cruzados."
        ),
    },
    "rossi": {
        "language": "Spanish",
        "template": (
            "Compact Spanish template: Técnica: RMN de la rodilla. Resultados: "
            "... Impresión: ... Short telegraphic sentences."
        ),
    },
    "samsun": {
        "language": "Turkish",
        "template": (
            "Turkish report with Tetkik protokolü / Bulgular / Sonuç. Uses "
            "medial/lateral menisküs and ön/arka çapraz bağ."
        ),
    },
    "hacettepe": {
        "language": "Turkish",
        "template": (
            "Turkish academic narrative covering menisci, cruciate and collateral "
            "ligaments, cartilage, and effusion, ending with a short Sonuç."
        ),
    },
    "dubrava": {
        "language": "Croatian",
        "template": (
            "Croatian narrative MRI description covering menisci (medijalni/"
            "lateralni), križni ligamenti, hrskavica, and izljev."
        ),
    },
    "sarajevo": {
        "language": "Bosnian/Croatian",
        "template": (
            "South-Slavic narrative covering menisci, cruciate ligaments, "
            "cartilage, and joint fluid."
        ),
    },
    "heraklion": {
        "language": "Greek",
        "template": (
            "Greek MRI report with ΤΕΧΝΙΚΗ and ΕΥΡΗΜΑΤΑ sections covering "
            "μηνίσκοι, χιαστοί, and αρθρικός χόνδρος."
        ),
    },
    "sofia": {
        "language": "Bulgarian",
        "template": (
            "Bulgarian report headed МР находка covering менискуси, кръстни "
            "връзки, хрущяли, and ставен излив."
        ),
    },
    "wurzburg": {
        "language": "German",
        "template": (
            "German narrative covering Menisken, Kreuzbänder, Kollateralbänder, "
            "Knorpel, and Gelenkerguss without English headers."
        ),
    },
    "az_delta_belgium": {
        "language": "Dutch",
        "template": (
            "Dutch report with Klinische inlichtingen / Bevindingen / CONCLUSIE. "
            "Uses voorste/achterste kruisband and mediale/laterale meniscus."
        ),
    },
    "mcgill": {
        "language": "English or French",
        "template": (
            "Canadian academic style with FINDINGS / IMPRESSION or French "
            "RÉSULTATS / CONCLUSION compartment headings."
        ),
    },
    "unity_health_toronto": {
        "language": "English",
        "template": (
            "Canadian English narrative with Findings and Impression covering "
            "menisci, ligaments, cartilage, and effusion."
        ),
    },
    "khon_kaen": {
        "language": "English",
        "template": (
            "English structured findings with Technique / Findings / Impression "
            "and explicit laterality."
        ),
    },
    "china_medical_university": {
        "language": "English",
        "template": (
            "English structured MRI report with Technique, Findings, and "
            "Impression sections."
        ),
    },
}

_HEADER_CUES: Dict[str, re.Pattern] = {
    "jefferson": re.compile(r"(?i)\b(exam type|technique|findings|impression|osseous)\b"),
    "mater_dei": re.compile(r"(?i)\b(findings|conclusion)\b"),
    "santiago": re.compile(r"(?i)\b(hallazgos|conclusi[oó]n|antecedentes)\b"),
    "rossi": re.compile(r"(?i)\b(t[eé]cnica|resultados|impresi[oó]n)\b"),
    "samsun": re.compile(r"(?i)\b(bulgular|sonu[cç]|tetkik)\b"),
    "hacettepe": re.compile(r"(?i)\b(bulgular|sonu[cç]|menisk)\b"),
    "dubrava": re.compile(r"(?i)\b(menisk|kri[zž]n|hrskavic|izljev)\b"),
    "sarajevo": re.compile(r"(?i)\b(menisk|kri[zž]n|hrskavic)\b"),
    "heraklion": re.compile(r"(ΤΕΧΝΙΚ|ΕΥΡΗΜ|μηνίσκ|χιασ)", re.IGNORECASE),
    "sofia": re.compile(r"(МР|мениск|връз|хрущ|излив)", re.IGNORECASE),
    "wurzburg": re.compile(r"(?i)\b(menisk|kreuzband|kollateral|gelenkerguss|knorpel)\b"),
    "az_delta_belgium": re.compile(r"(?i)\b(bevindingen|conclusie|kruisband|meniscus)\b"),
    "mcgill": re.compile(r"(?i)\b(findings|impression|r[eé]sultats|conclusion)\b"),
    "unity_health_toronto": re.compile(r"(?i)\b(findings|impression)\b"),
    "khon_kaen": re.compile(r"(?i)\b(technique|findings|impression)\b"),
    "china_medical_university": re.compile(r"(?i)\b(technique|findings|impression)\b"),
}


@dataclass
class SiteStyle:
    site_name: str
    language: str
    template: str
    exemplars: List[str]


def load_site_exemplars(
    mapped_csv: Path | str = DEFAULT_MAPPED,
    per_site: int = 3,
    max_chars: int = 1200,
) -> Dict[str, SiteStyle]:
    """Build few-shot exemplars for every site that appears in mapped.csv."""
    frame = pd.read_csv(mapped_csv, dtype=str, keep_default_na=False)
    if "site_name" not in frame.columns or "Report" not in frame.columns:
        raise ValueError(f"{mapped_csv} must contain site_name and Report")
    styles: Dict[str, SiteStyle] = {}
    for site_name, group in frame.groupby("site_name"):
        if not site_name or site_name == "#N/A":
            continue
        reports = [
            " ".join(str(text).split())[:max_chars]
            for text in group["Report"].tolist()
            if str(text).strip()
        ]
        profile = SITE_PROFILES.get(
            site_name,
            {
                "language": "the original report language",
                "template": "the institutional section order and phrasing of this site",
            },
        )
        styles[site_name] = SiteStyle(
            site_name=site_name,
            language=profile["language"],
            template=profile["template"],
            exemplars=reports[:per_site],
        )
    return styles


def site_template_match(text: str, site_name: str) -> bool:
    """True when the rewrite shows the site's characteristic headers or terms."""
    report = "" if text is None else str(text)
    if not report.strip():
        return False
    cue = _HEADER_CUES.get(site_name)
    if cue is None:
        return len(report.split()) >= 8
    return cue.search(report) is not None and len(report.split()) >= 8


class SiteStyleTransferrer:
    """Rewrite reports into a mapped institutional style with an instruct LM."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        mapped_csv: Path | str = DEFAULT_MAPPED,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.styles = load_site_exemplars(mapped_csv)
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        LOGGER.info("Loading style-transfer model %s on %s", self.model_name, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        dtype = torch.bfloat16 if str(self.device).startswith("cuda") else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=self.device if str(self.device).startswith("cuda") else None,
        )
        if not str(self.device).startswith("cuda"):
            self._model = self._model.to(self.device)
        self._model.eval()

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            return

    def rewrite(self, report: str, site_name: str) -> str:
        return self.rewrite_batch([report], [site_name])[0]

    def rewrite_batch(self, reports: Sequence[str], site_names: Sequence[str]) -> List[str]:
        if len(reports) != len(site_names):
            raise ValueError("reports and site_names must have the same length")
        self.load()
        assert self._tokenizer is not None and self._model is not None
        import torch

        prompts = [
            self._build_messages(report, site_name)
            for report, site_name in zip(reports, site_names)
        ]
        rendered = [
            self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in prompts
        ]
        encoded = self._tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        encoded = {key: value.to(self._model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                top_p=0.9 if self.temperature > 0 else None,
                eos_token_id=self._tokenizer.eos_token_id,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        decoded = self._tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return [
            _postprocess(text, original)
            for text, original in zip(decoded, reports)
        ]

    def _build_messages(self, report: str, site_name: str) -> List[Dict[str, str]]:
        style = self.styles.get(site_name)
        if style is None:
            style = SiteStyle(
                site_name=site_name,
                language="the original report language",
                template="the institutional phrasing of this site",
                exemplars=[],
            )
        examples = ""
        for index, exemplar in enumerate(style.exemplars, start=1):
            examples += f"\nExample {index} ({site_name} style):\n{exemplar}\n"
        system = (
            "You are a musculoskeletal radiologist. Rewrite knee MRI reports into a "
            "target institutional style. Preserve every abnormal and normal finding, "
            "laterality (medial/lateral, left/right), measurements, and severity. "
            "Do not invent findings. Do not omit findings. Output only the rewritten report."
        )
        user = (
            f"Target institution: {site_name}\n"
            f"Target language: {style.language}\n"
            f"Target template: {style.template}\n"
            f"{examples}\n"
            f"Original report:\n{report.strip()}\n\n"
            f"Rewritten report in the {site_name} style:"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]


def _postprocess(text: str, original: str) -> str:
    cleaned = str(text).strip()
    if not cleaned:
        return str(original).strip()
    cleaned = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", cleaned).strip()
    # Drop accidental prompt echoes.
    for marker in ("Rewritten report", "Original report:", "Target institution:"):
        if marker in cleaned:
            cleaned = cleaned.split(marker)[0].strip()
    return cleaned or str(original).strip()
