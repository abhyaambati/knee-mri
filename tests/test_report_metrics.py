import pandas as pd
import pytest

from evaluate_phase1_test import attach_report_metrics, compute_metrics, score_report


def test_identical_report_is_perfect():
    text = "The ACL is intact. No joint effusion."
    row = score_report(text, text)
    assert row["bleu_4"] == pytest.approx(1.0)
    assert all(0.0 <= value <= 1.0 for value in row.values())
    assert row["rouge_1"] == pytest.approx(1.0)
    assert row["rouge_2"] == pytest.approx(1.0)
    assert row["rouge_l"] == pytest.approx(1.0)


def test_empty_hypothesis_is_zero():
    row = score_report("Medial meniscus tear.", "")
    assert row["bleu_1"] == 0.0
    assert row["rouge_l"] == 0.0


def test_attach_scores_every_report():
    frame = pd.DataFrame(
        {
            "reference_report": ["ACL intact.", "Joint effusion."],
            "generated_report_phase1": ["ACL intact.", "No effusion."],
        }
    )
    scored = attach_report_metrics(frame)
    assert len(scored) == 2
    assert scored.loc[0, "bleu_1"] == pytest.approx(1.0)
    assert scored[["bleu_1", "bleu_4", "rouge_1", "rouge_l"]].map(lambda value: 0.0 <= value <= 1.0).all().all()
    assert scored.loc[1, "rouge_l"] < scored.loc[0, "rouge_l"]
    summary = compute_metrics(
        scored["reference_report"].tolist(),
        scored["generated_report_phase1"].tolist(),
    )
    assert set(summary) >= {"bleu_1", "bleu_4", "rouge_1", "rouge_2", "rouge_l"}
    assert summary["rouge_l"] == float(scored["rouge_l"].mean())
