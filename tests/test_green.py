import pytest

from metrics.green_eval import clean_response, make_prompt, score_analysis


ANALYSIS = """
[Explanation]:
The candidate repeats the ACL finding and adds an effusion.

[Clinically Significant Errors]:
(a) False report of a finding in the candidate: 1. joint effusion
(b) Missing a finding present in the reference: 0.
(c) Misidentification of a finding's anatomic location/position: 0.
(d) Misassessment of the severity of a finding: 0.
(e) Mentioning a comparison that isn't in the reference: 0.
(f) Omitting a comparison detailing a change from a prior study: 0.

[Clinically Insignificant Errors]:
(a) False report of a finding in the candidate: 0.

[Matched Findings]:
2. ACL intact; medial meniscus intact
"""


def test_prompt_keeps_reports_and_caps_words():
    prompt = make_prompt("alpha " * 400, "beta candidate")
    assert "2. Reference Report:" in prompt
    assert "beta candidate" in prompt
    assert prompt.split("2. Reference Report:")[1].split("3. Candidate Report:")[0].count("alpha") == 300


def test_score_uses_significant_errors_only():
    row = score_analysis(ANALYSIS)
    assert row["green_matched_findings"] == 2
    assert row["green_error_a"] == 1
    assert row["green_significant_errors"] == 1
    assert row["green_score"] == pytest.approx(2 / 3)


def test_no_matched_findings_scores_zero():
    text = ANALYSIS.replace("2. ACL intact; medial meniscus intact", "0. ")
    row = score_analysis(text)
    assert row["green_matched_findings"] == 0
    assert row["green_score"] == 0.0


def test_clean_response_drops_prompt_echo():
    raw = "<|user|>\nprompt</s><|assistant|>\n[Explanation]:\nThe reports agree.\n\n[Matched Findings]:\n1. effusion"
    cleaned = clean_response(raw)
    assert cleaned.startswith("The reports agree.")
    assert "<|assistant|>" not in cleaned
    assert score_analysis(raw)["green_score"] == pytest.approx(1.0)
