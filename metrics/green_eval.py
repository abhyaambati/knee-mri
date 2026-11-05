"""GREEN clinical-error scoring for radiology reports.

Local Hugging Face inference for StanfordAIMI/GREEN-RadLlama2-7b. The prompt,
response cleanup, and score match the Stanford-AIMI GREEN scorer:

    GREEN = matched / (matched + significant errors)  if matched > 0 else 0

Scores are in [0, 1]. Insignificant errors do not enter the score.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_GREEN_MODEL = "StanfordAIMI/GREEN-RadLlama2-7b"
SIGNIFICANT = "Clinically Significant Errors"
INSIGNIFICANT = "Clinically Insignificant Errors"
MATCHED = "Matched Findings"
CATEGORIES = (SIGNIFICANT, INSIGNIFICANT, MATCHED)
SUBCATEGORIES = (
    "(a) False report of a finding in the candidate",
    "(b) Missing a finding present in the reference",
    "(c) Misidentification of a finding's anatomic location/position",
    "(d) Misassessment of the severity of a finding",
    "(e) Mentioning a comparison that isn't in the reference",
    "(f) Omitting a comparison detailing a change from a prior study",
)
ERROR_COLUMNS = tuple(f"green_error_{letter}" for letter in "abcdef")
GREEN_COLUMNS = (
    "green_score",
    "green_matched_findings",
    "green_significant_errors",
    *ERROR_COLUMNS,
    "green_analysis",
)

# Official green_score.utils chat template. GREEN was trained with these roles.
CHAT_TEMPLATE = (
    "{% for message in messages %}\n"
    "{% if message['from'] == 'human' %}\n"
    "{{ '<|user|>\\n' + message['value'] + eos_token }}\n"
    "{% elif message['from'] == 'system' %}\n"
    "{{ '<|system|>\\n' + message['value'] + eos_token }}\n"
    "{% elif message['from'] == 'gpt' %}\n"
    "{{ '<|assistant|>\\n' + message['value'] + eos_token }}\n"
    "{% endif %}\n"
    "{% if loop.last and add_generation_prompt %}\n"
    "{{ '<|assistant|>' }}\n"
    "{% endif %}\n"
    "{% endfor %}"
)


def make_prompt(reference: str, candidate: str, max_words: int = 300) -> str:
    """Official GREEN comparison prompt. Each report is capped at 300 words."""
    reference = " ".join(str(reference).split()[:max_words])
    candidate = " ".join(str(candidate).split()[:max_words])
    return (
        "Objective: Evaluate the accuracy of a candidate radiology report in comparison to a "
        "reference radiology report composed by expert radiologists.\n\n"
        "    Process Overview: You will be presented with:\n\n"
        "    1. The criteria for making a judgment.\n"
        "    2. The reference radiology report.\n"
        "    3. The candidate radiology report.\n"
        "    4. The desired format for your assessment.\n\n"
        "    1. Criteria for Judgment:\n\n"
        "    For each candidate report, determine:\n\n"
        "    The count of clinically significant errors.\n"
        "    The count of clinically insignificant errors.\n\n"
        "    Errors can fall into one of these categories:\n\n"
        "    a) False report of a finding in the candidate.\n"
        "    b) Missing a finding present in the reference.\n"
        "    c) Misidentification of a finding's anatomic location/position.\n"
        "    d) Misassessment of the severity of a finding.\n"
        "    e) Mentioning a comparison that isn't in the reference.\n"
        "    f) Omitting a comparison detailing a change from a prior study.\n"
        "    Note: Concentrate on the clinical findings rather than the report's writing style. "
        "Evaluate only the findings that appear in both reports.\n\n"
        f"    2. Reference Report:\n    {reference}\n\n"
        f"    3. Candidate Report:\n    {candidate}\n\n"
        "    4. Reporting Your Assessment:\n\n"
        "    Follow this specific format for your output, even if no errors are found:\n"
        "    ```\n"
        "    [Explanation]:\n"
        "    <Explanation>\n\n"
        "    [Clinically Significant Errors]:\n"
        "    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n"
        "    ....\n"
        "    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n\n"
        "    [Clinically Insignificant Errors]:\n"
        "    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n"
        "    ....\n"
        "    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n\n"
        "    [Matched Findings]:\n"
        "    <The number of matched findings>. <Finding 1>; <Finding 2>; ...; <Finding n>\n"
        "    ```\n"
    )


def clean_response(response: str) -> str:
    """Drop the echoed prompt and special tokens, matching green_score.utils."""
    if "[Explanation]:" in response:
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1]
        template_echo = (
            "[Explanation]:\n    <Explanation>\n" in response
            or "[Explanation]:\n<Explanation>" in response
        )
        response = response.split("[Explanation]:")[1 if template_echo else -1]
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1]
    return response.replace("</s>", "").replace("<unk>", "").strip()


def parse_error_counts(text: str, category: str) -> tuple[int, list[int]]:
    """Return (total, per-subcategory counts) for one GREEN section."""
    if category not in CATEGORIES:
        raise ValueError(f"Unknown GREEN category: {category}")
    sub_counts = [0] * len(SUBCATEGORIES)
    match = re.search(
        rf"\[{re.escape(category)}\]:\s*(.*?)(?:\n\s*\n|\n\s*\[|\Z)",
        text,
        flags=re.DOTALL,
    )
    if match is None or match.group(1).startswith("No"):
        return 0, sub_counts
    body = match.group(1)
    if category == MATCHED:
        counts = re.findall(r"^\b\d+\b(?=\.)", body) or re.findall(r"\b\d+\b(?=\.)", body)
        return (int(counts[0]), sub_counts) if counts else (0, sub_counts)

    lines = sorted(re.findall(r"\([a-f]\) .*", body))
    prefixes = [item.split(" ", 1)[0] + " " for item in SUBCATEGORIES]
    if not lines:
        lines = sorted(re.findall(r"\([1-6]\) .*", body))
        prefixes = [f"({index}) " for index in range(1, len(SUBCATEGORIES) + 1)]
    for position, prefix in enumerate(prefixes):
        for line in lines:
            if line.startswith(prefix):
                counts = re.findall(r"(?<=: )\b\d+\b(?=\.)", line)
                if counts:
                    sub_counts[position] = int(counts[0])
    return sum(sub_counts), sub_counts


def score_analysis(analysis: str) -> dict[str, float | int | str]:
    """Parse one GREEN completion into a 0-1 score and error counts."""
    cleaned = clean_response(analysis or "")
    _, significant = parse_error_counts(cleaned, SIGNIFICANT)
    matched, _ = parse_error_counts(cleaned, MATCHED)
    significant_total = sum(significant)
    score = 0.0 if matched == 0 else matched / (matched + significant_total)
    row: dict[str, float | int | str] = {
        "green_score": float(score),
        "green_matched_findings": int(matched),
        "green_significant_errors": int(significant_total),
        "green_analysis": cleaned,
    }
    for column, count in zip(ERROR_COLUMNS, significant):
        row[column] = int(count)
    return row


class GreenEvaluator:
    """Batch GREEN inference on one GPU. The language model loads on first use."""

    def __init__(
        self,
        model_name: str = DEFAULT_GREEN_MODEL,
        device: str = "cuda",
        batch_size: int = 4,
        max_new_tokens: int = 768,
    ):
        if batch_size < 1 or max_new_tokens < 1:
            raise ValueError("batch_size and max_new_tokens must be positive")
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for GREEN; pass device='cpu'")
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        LOGGER.info("Loading GREEN model %s on %s", self.model_name, self.device)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=False,
            trust_remote_code=True,
        )
        tokenizer.chat_template = CHAT_TEMPLATE
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
            device_map=self.device if self.device.startswith("cuda") else None,
        )
        model.config.use_cache = True
        if not self.device.startswith("cuda"):
            model = model.to(self.device)
        model.eval()
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        self._tokenizer = tokenizer
        self._model = model

    def score_batch(self, references: list[str], candidates: list[str]) -> list[str]:
        """Generate GREEN analyses. One string per report pair."""
        import torch

        if len(references) != len(candidates):
            raise ValueError("Reference and candidate batches must have the same length")
        if not references:
            return []
        self.load()
        assert self._model is not None and self._tokenizer is not None
        prompts = [
            [{"from": "human", "value": make_prompt(reference, candidate)}, {"from": "gpt", "value": ""}]
            for reference, candidate in zip(references, candidates)
        ]
        rendered = [
            self._tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            for prompt in prompts
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
                do_sample=False,
                eos_token_id=self._tokenizer.eos_token_id,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        return [clean_response(text) for text in self._tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]

    def score_reports(
        self,
        references: list[str],
        candidates: list[str],
        checkpoint_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """Score every pair. Optionally rewrite a partial CSV after each batch."""
        from tqdm import tqdm

        if len(references) != len(candidates) or not references:
            raise ValueError("Need nonempty equal-length reference and candidate lists")
        rows: list[dict] = []
        progress = tqdm(
            range(0, len(references), self.batch_size),
            desc="green",
            total=(len(references) + self.batch_size - 1) // self.batch_size,
            dynamic_ncols=True,
        )
        for start in progress:
            stop = start + self.batch_size
            analyses = self.score_batch(references[start:stop], candidates[start:stop])
            rows.extend(score_analysis(analysis) for analysis in analyses)
            scored = sum(row["green_score"] for row in rows) / len(rows)
            progress.set_postfix(n=len(rows), green=f"{scored:.3f}")
            if checkpoint_path is not None:
                pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
        return pd.DataFrame(rows, columns=list(GREEN_COLUMNS))

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
        for column in GREEN_COLUMNS:
            scored[column] = details[column].to_numpy()
        return scored
