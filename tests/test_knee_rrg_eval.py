from pathlib import Path

from knee_rrg_eval.evaluators.factuality_eval import FactualityEvaluator
from knee_rrg_eval.evaluators.style_eval import template_matches
from knee_rrg_eval.metrics.msk_extractor import MSKExtractor
from knee_rrg_eval.run_experiment import load_test_with_sites
from knee_rrg_eval.style_transfer import site_template_match
from knee_rrg_eval.utils.excel_exporter import RESULT_COLUMNS, export_results_to_excel


def test_extracts_meniscus_tear_acl_intact_and_effusion():
    status = MSKExtractor().extract(
        "The anterior cruciate ligament is intact. There is a tear of the posterior horn "
        "of the medial meniscus. A small joint effusion is present."
    )
    assert status["ACL"] == "Intact"
    assert status["Medial Meniscus"] == "Tear"
    assert status["Joint Effusion"] == "Present"
    assert status["Lateral Meniscus"] == "Not Mentioned"


def test_multilingual_mentions_are_detected():
    extractor = MSKExtractor()
    turkish = extractor.extract(
        "Diz eklemi içi sıvı miktarı normal. Çapraz ve yan bağlar normal. "
        "Medyal ve lateral menisküs normal."
    )
    croatian = extractor.extract(
        "Medijalni menisk bez znakova rupture. Prednji križni ligament: manja parcijalna ruptura. Nema izljeva."
    )
    spanish = extractor.extract("Contusión ósea en el cóndilo femoral medial. Derrame articular.")
    assert turkish["Medial Meniscus"] == "Intact"
    assert turkish["ACL"] == "Intact"
    assert croatian["Medial Meniscus"] == "Intact"
    assert croatian["ACL"] == "Partial Tear"
    assert croatian["Joint Effusion"] == "Absent"
    assert spanish["Joint Effusion"] == "Present"
    assert spanish["Bone Marrow Edema"] == "Present"


def test_laterality_swap_is_not_factual():
    reference = (
        "The anterior cruciate ligament is intact. There is a tear of the medial meniscus. "
        "A small joint effusion is present."
    )
    candidate = "There is a tear of the lateral meniscus. The anterior cruciate ligament is intact."
    result = FactualityEvaluator().evaluate(reference, candidate)
    assert result["laterality_errors"]
    assert result["factuality_preserved"] is False


def test_identical_report_is_preserved_and_empty_is_not():
    reference = "The anterior cruciate ligament is intact. There is a tear of the medial meniscus."
    preserved = FactualityEvaluator().evaluate(reference, reference)
    empty = FactualityEvaluator().evaluate(reference, "  ")
    assert preserved["factuality_preserved"] is True
    assert preserved["msk_f1"] == 1.0
    assert empty["factuality_preserved"] is False
    assert empty["msk_f1"] == 0.0


def test_style_templates_distinguish_bullets_sites_and_narrative():
    bullet = "FINDINGS:\n- Medial meniscus tear.\n\nIMPRESSION:\n- Medial meniscus tear."
    narrative = "The medial meniscus is torn and the anterior cruciate ligament is intact."
    jefferson = "Exam Type: MRI KNEE\nTECHNIQUE: MRI\nFINDINGS:\nOSSEOUS STRUCTURES: Normal.\nIMPRESSION: Normal."
    assert template_matches(bullet, "bullet_findings_impression")
    assert not template_matches(narrative, "bullet_findings_impression")
    assert template_matches(narrative, "narrative_block")
    assert site_template_match(jefferson, "jefferson")


def test_test_csv_maps_to_sites():
    frame = load_test_with_sites()
    assert len(frame) == 434
    assert frame["site_name"].isna().sum() == 0
    assert "dubrava" in set(frame["site_name"])
    assert "jefferson" in set(frame["site_name"])


def test_excel_export_writes_required_columns(tmp_path: Path):
    row = {column: False if column.endswith("_Flag") else "x" for column in RESULT_COLUMNS}
    row["RadGraph_F1"] = 0.5
    path = export_results_to_excel([row], tmp_path / "results.xlsx")
    from openpyxl import load_workbook

    workbook = load_workbook(path)
    headers = [cell.value for cell in workbook["results"][1]]
    assert headers[: len(RESULT_COLUMNS)] == RESULT_COLUMNS
    assert workbook["results"]["L2"].value is False
