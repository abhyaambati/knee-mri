from CRIMSON.generate_score import CRIMSONScore


def _score(evaluation):
    return CRIMSONScore._calculate_crimson(None, evaluation)


def test_perfect_urgent_match_scores_one():
    result = _score({
        "errors": {"false_findings": [], "missing_findings": [], "attribute_errors": []},
        "matched_findings": [{"ref_id": "R1", "pred_id": "P1"}],
        "reference_findings": [{"id": "R1", "clinical_significance": "urgent"}],
        "predicted_findings": [{"id": "P1", "clinical_significance": "urgent"}],
    })
    assert result["crimson_score"] == 1.0


def test_false_urgent_finding_is_negative():
    result = _score({
        "errors": {"false_findings": ["P1"], "missing_findings": [], "attribute_errors": []},
        "matched_findings": [],
        "reference_findings": [],
        "predicted_findings": [{"id": "P1", "clinical_significance": "urgent"}],
    })
    assert -1.0 < result["crimson_score"] < 0.0


def test_significant_attribute_error_reduces_credit():
    perfect = _score({
        "errors": {"false_findings": [], "missing_findings": [], "attribute_errors": []},
        "matched_findings": [{"ref_id": "R1", "pred_id": "P1"}],
        "reference_findings": [{"id": "R1", "clinical_significance": "actionable_not_urgent"}],
        "predicted_findings": [{"id": "P1", "clinical_significance": "actionable_not_urgent"}],
    })
    partial = _score({
        "errors": {
            "false_findings": [],
            "missing_findings": [],
            "attribute_errors": [{
                "ref_id": "R1",
                "pred_id": "P1",
                "severity": "significant",
                "error_types": ["location"],
            }],
        },
        "matched_findings": [{"ref_id": "R1", "pred_id": "P1"}],
        "reference_findings": [{"id": "R1", "clinical_significance": "actionable_not_urgent"}],
        "predicted_findings": [{"id": "P1", "clinical_significance": "actionable_not_urgent"}],
    })
    assert partial["crimson_score"] < perfect["crimson_score"]
    assert 0.0 < partial["crimson_score"] < 1.0
