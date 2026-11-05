"""Write the audit sheet with pandas and openpyxl."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Union

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

RESULT_COLUMNS = [
    "Sample_ID",
    "Ground_Truth_Report",
    "Generated_Report",
    "Target_Style_Label",
    "RadGraph_F1",
    "MSK_Diagnostic_F1",
    "GREEN_Score",
    "BERTScore_BioClinical",
    "RaTEScore",
    "BLEU_4",
    "ROUGE_L",
    "Factuality_Preserved_Flag",
    "Style_Transfer_Success_Flag",
]

AUDIT_COLUMNS = [
    "Style_Distance",
    "Style_Similarity",
    "Status_Flips",
    "Laterality_Errors",
    "Hallucinations",
    "Omissions",
]

FLOAT_COLUMNS = {
    "RadGraph_F1",
    "MSK_Diagnostic_F1",
    "GREEN_Score",
    "BERTScore_BioClinical",
    "RaTEScore",
    "BLEU_4",
    "ROUGE_L",
    "Style_Distance",
    "Style_Similarity",
}
BOOL_COLUMNS = {"Factuality_Preserved_Flag", "Style_Transfer_Success_Flag"}


def export_results_to_excel(
    rows: Sequence[Dict[str, object]],
    path: Union[str, Path] = "results.xlsx",
    sheet_name: str = "results",
) -> Path:
    """Save one row per report. Missing metrics stay blank. Flags stay booleans."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = RESULT_COLUMNS + [column for column in AUDIT_COLUMNS if _any_present(rows, column)]
    frame = pd.DataFrame(list(rows))
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[columns]
    for column in FLOAT_COLUMNS.intersection(frame.columns):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in BOOL_COLUMNS.intersection(frame.columns):
        frame[column] = frame[column].map(_as_bool)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name=sheet_name)
        worksheet = writer.sheets[sheet_name]
        _format_sheet(worksheet, frame)
    return path


def _any_present(rows: Sequence[Dict[str, object]], column: str) -> bool:
    return any(column in row for row in rows)


def _as_bool(value: object) -> object:
    if value is None or (isinstance(value, float) and value != value):
        return pd.NA
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
        return pd.NA
    return bool(value)


def _format_sheet(worksheet, frame: pd.DataFrame) -> None:
    header_fill = PatternFill("solid", fgColor="1F4E79")
    header_font = Font(bold=True, color="FFFFFF")
    for cell in worksheet[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    worksheet.auto_filter.ref = worksheet.dimensions
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    for index, column in enumerate(frame.columns, start=1):
        width = 18
        if column in {"Ground_Truth_Report", "Generated_Report"}:
            width = 48
        elif column.endswith("_Flag") or column.endswith("_Errors") or column in {"Status_Flips", "Hallucinations", "Omissions"}:
            width = 28
        worksheet.column_dimensions[get_column_letter(index)].width = width
        if column in FLOAT_COLUMNS:
            for cell in worksheet.iter_cols(min_col=index, max_col=index, min_row=2, max_row=worksheet.max_row):
                for item in cell:
                    item.number_format = "0.000"
    worksheet.row_dimensions[1].height = 30
    for row in worksheet.iter_rows(min_row=2, max_row=worksheet.max_row):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
