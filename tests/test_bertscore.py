from metrics.bertscore_eval import clip_unit, zero_empty_hypotheses


def test_clip_unit_bounds():
    assert clip_unit(-0.2) == 0.0
    assert clip_unit(1.2) == 1.0
    assert clip_unit(0.4) == 0.4


def test_empty_hypothesis_is_zero():
    precision, recall, f1 = zero_empty_hypotheses([0.9, 0.8], [0.7, 0.6], [0.8, 0.7], ["ACL intact.", "  "])
    assert precision == [0.9, 0.0]
    assert recall == [0.7, 0.0]
    assert f1 == [0.8, 0.0]
