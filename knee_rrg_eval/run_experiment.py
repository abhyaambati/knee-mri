"""Site style-transfer audit on the Phase-1 test split.

Pipeline:
1. Load ``Split/test.csv`` and map each study to ``site_name`` via ``EDA/mapped.csv``.
2. Rewrite every ground-truth report into that site's institutional style (L → L*).
3. Score clinical invariance of the rewrite against the original (S → S*).
4. Write ``results.xlsx`` and print the factuality / style trade-off.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knee_rrg_eval.knee_mri_evaluator import KneeMRIReportEvaluator, release_memory
from knee_rrg_eval.style_transfer import SiteStyleTransferrer
from knee_rrg_eval.utils.excel_exporter import RESULT_COLUMNS, export_results_to_excel

LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST = ROOT / "Split" / "test.csv"
DEFAULT_MAPPED = ROOT / "EDA" / "mapped.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results.xlsx"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "style_transfer_checkpoint.csv"


def load_test_with_sites(
    test_csv: Path = DEFAULT_TEST,
    mapped_csv: Path = DEFAULT_MAPPED,
) -> pd.DataFrame:
    """Join ground-truth reports to institutional site labels."""
    test = pd.read_csv(test_csv, dtype=str, keep_default_na=False)
    mapped = pd.read_csv(mapped_csv, dtype=str, keep_default_na=False)
    if "StudyInstanceUID" not in test.columns:
        raise ValueError(f"{test_csv} must contain StudyInstanceUID")
    report_col = _pick_column(test, ("Report", "reference_report", "ground_truth", "reference"))
    if "site_name" not in mapped.columns:
        raise ValueError(f"{mapped_csv} must contain site_name")
    sites = mapped[["StudyInstanceUID", "site_name"]].drop_duplicates("StudyInstanceUID")
    frame = test.merge(sites, on="StudyInstanceUID", how="left")
    missing = frame["site_name"].isna() | (frame["site_name"].astype(str).str.strip() == "") | (
        frame["site_name"] == "#N/A"
    )
    if missing.any():
        raise ValueError(f"{int(missing.sum())} test studies have no usable site_name in {mapped_csv}")
    frame = frame.rename(columns={report_col: "Ground_Truth_Report"})
    frame["Sample_ID"] = frame["StudyInstanceUID"]
    frame["Target_Style_Label"] = frame["site_name"]
    return frame[["Sample_ID", "StudyInstanceUID", "Ground_Truth_Report", "Target_Style_Label", "site_name"]]


def _pick_column(frame: pd.DataFrame, names: Sequence[str]) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise ValueError(f"Need one of {list(names)}; found {list(frame.columns)}")


def rewrite_ground_truths(
    frame: pd.DataFrame,
    transferrer: SiteStyleTransferrer,
    batch_size: int = 4,
    checkpoint_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Render each ground truth into its mapped site style."""
    scored = frame.copy()
    start = 0
    rewritten = [""] * len(scored)
    if checkpoint_path is not None and checkpoint_path.is_file():
        partial = pd.read_csv(checkpoint_path, dtype=str, keep_default_na=False)
        if "Sample_ID" in partial.columns and "Generated_Report" in partial.columns:
            by_id = dict(zip(partial["Sample_ID"], partial["Generated_Report"]))
            for index, sample_id in enumerate(scored["Sample_ID"].tolist()):
                if sample_id in by_id and str(by_id[sample_id]).strip():
                    rewritten[index] = by_id[sample_id]
            start = next((i for i, text in enumerate(rewritten) if not str(text).strip()), len(rewritten))
            LOGGER.info("Resuming style transfer from row %d / %d", start, len(scored))

    progress = tqdm(range(start, len(scored), batch_size), desc="style-transfer", dynamic_ncols=True)
    for begin in progress:
        end = min(begin + batch_size, len(scored))
        reports = scored["Ground_Truth_Report"].iloc[begin:end].tolist()
        sites = scored["Target_Style_Label"].iloc[begin:end].tolist()
        try:
            outputs = transferrer.rewrite_batch(reports, sites)
        except Exception:
            LOGGER.exception("Batch %d failed; rewriting one report at a time", begin)
            outputs = []
            for report, site in zip(reports, sites):
                try:
                    outputs.append(transferrer.rewrite(report, site))
                except Exception:
                    LOGGER.exception("Failed rewrite for site=%s", site)
                    outputs.append(report)
        for offset, text in enumerate(outputs):
            rewritten[begin + offset] = text
        done = sum(1 for text in rewritten if str(text).strip())
        progress.set_postfix(n=done)
        if checkpoint_path is not None:
            checkpoint = scored[["Sample_ID", "Target_Style_Label"]].copy()
            checkpoint["Generated_Report"] = rewritten
            checkpoint.to_csv(checkpoint_path, index=False)
    scored["Generated_Report"] = rewritten
    return scored


def build_eval_samples(frame: pd.DataFrame) -> List[Dict[str, str]]:
    return [
        {
            "Sample_ID": row.Sample_ID,
            "Ground_Truth_Report": row.Ground_Truth_Report,
            "Generated_Report": row.Generated_Report,
            "Target_Style_Label": row.Target_Style_Label,
        }
        for row in frame.itertuples(index=False)
    ]


def pareto_front(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    scored = []
    for row in rows:
        factuality = _mean(
            [
                row.get("MSK_Diagnostic_F1"),
                row.get("GREEN_Score"),
                row.get("RadGraph_F1"),
                row.get("BERTScore_BioClinical"),
            ]
        )
        style = row.get("Style_Similarity")
        style_value = float(style) if isinstance(style, (int, float)) else 0.0
        scored.append((row, factuality, style_value))
    front = []
    for row, factuality, style in scored:
        dominated = any(
            other_f >= factuality and other_s >= style and (other_f > factuality or other_s > style)
            for _other, other_f, other_s in scored
        )
        if not dominated:
            front.append(
                {
                    "Sample_ID": row["Sample_ID"],
                    "site": row.get("Target_Style_Label"),
                    "factuality": factuality,
                    "style_similarity": style,
                }
            )
    return front


def print_summary(rows: Sequence[Dict[str, object]]) -> None:
    print("\nSite style-transfer audit: original GT (S, L) → site rewrite (S*, L*)")
    print("-" * 72)
    metric_names = [
        "BLEU_4",
        "ROUGE_L",
        "BERTScore_BioClinical",
        "RaTEScore",
        "RadGraph_F1",
        "MSK_Diagnostic_F1",
        "GREEN_Score",
        "Style_Similarity",
    ]
    for name in metric_names:
        values = [float(row[name]) for row in rows if isinstance(row.get(name), (int, float))]
        if not values:
            print(f"{name:28}  n=0")
            continue
        print(f"{name:28}  n={len(values):<4} mean={sum(values) / len(values):.3f}")
    preserved = sum(1 for row in rows if row["Factuality_Preserved_Flag"])
    styled = sum(1 for row in rows if row["Style_Transfer_Success_Flag"])
    both = sum(
        1 for row in rows if row["Factuality_Preserved_Flag"] and row["Style_Transfer_Success_Flag"]
    )
    print(f"\nFactuality preserved (S→S*): {preserved}/{len(rows)}")
    print(f"Style transfer succeeded (L→L*): {styled}/{len(rows)}")
    print(f"Both preserved: {both}/{len(rows)}")
    if rows:
        print(
            "MSK corpus micro F1 "
            f"{rows[0]['MSK_Micro_F1']:.3f}  macro F1 {rows[0]['MSK_Macro_F1']:.3f}"
        )
    by_site: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_site.setdefault(str(row["Target_Style_Label"]), []).append(row)
    print("\nPer-site factuality / style rates:")
    for site in sorted(by_site):
        group = by_site[site]
        fact = sum(1 for row in group if row["Factuality_Preserved_Flag"]) / len(group)
        style = sum(1 for row in group if row["Style_Transfer_Success_Flag"]) / len(group)
        msk = _mean([row.get("MSK_Diagnostic_F1") for row in group])
        print(f"  {site:28} n={len(group):<3} fact={fact:.2f} style={style:.2f} msk_f1={msk:.3f}")
    print("\nPareto front (higher factuality composite, higher style similarity):")
    for item in pareto_front(rows)[:15]:
        print(
            f"  {str(item['site']):24} {item['Sample_ID'][:28]} "
            f"factuality={item['factuality']:.3f} style={item['style_similarity']:.3f}"
        )


def _mean(values: Sequence[object]) -> float:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    if not numeric:
        return 0.0
    return sum(numeric) / len(numeric)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--mapped-csv", type=Path, default=DEFAULT_MAPPED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--rewrite-batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for smoke tests")
    parser.add_argument("--skip-rewrite", action="store_true", help="Reuse checkpoint rewrites only")
    parser.add_argument("--skip-green", action="store_true")
    parser.add_argument("--skip-radgraph", action="store_true")
    parser.add_argument("--skip-ratescore", action="store_true")
    return parser.parse_args()


def main() -> Path:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.batch_size < 1 or args.rewrite_batch_size < 1:
        raise SystemExit("batch sizes must be positive")

    frame = load_test_with_sites(args.test_csv, args.mapped_csv)
    if args.limit is not None:
        frame = frame.head(args.limit).copy()
    LOGGER.info("Loaded %d test reports across %d sites", len(frame), frame["site_name"].nunique())
    print(frame["site_name"].value_counts().to_string())

    if args.skip_rewrite:
        if not args.checkpoint.is_file():
            raise SystemExit(f"--skip-rewrite needs checkpoint at {args.checkpoint}")
        partial = pd.read_csv(args.checkpoint, dtype=str, keep_default_na=False)
        frame = frame.merge(partial[["Sample_ID", "Generated_Report"]], on="Sample_ID", how="left")
        if frame["Generated_Report"].isna().any() or (frame["Generated_Report"].astype(str).str.strip() == "").any():
            raise SystemExit("Checkpoint is missing rewrites for some Sample_IDs")
    else:
        transferrer = SiteStyleTransferrer(
            model_name=args.model,
            device=args.device,
            mapped_csv=args.mapped_csv,
        )
        frame = rewrite_ground_truths(
            frame,
            transferrer,
            batch_size=args.rewrite_batch_size,
            checkpoint_path=args.checkpoint,
        )
        transferrer.unload()
        release_memory()

    samples = build_eval_samples(frame)
    evaluator = KneeMRIReportEvaluator(
        device=args.device,
        batch_size=args.batch_size,
        enable_green=not args.skip_green,
        enable_radgraph=not args.skip_radgraph,
        enable_ratescore=not args.skip_ratescore,
    )
    # Restrict style centroids to sites present in this run.
    evaluator.style.site_names = sorted(frame["Target_Style_Label"].unique())
    rows = evaluator.evaluate_batch(samples)
    export_results_to_excel(rows, args.output)
    print_summary(rows)
    summary = {
        "n": len(rows),
        "factuality_preserved": sum(1 for row in rows if row["Factuality_Preserved_Flag"]),
        "style_transfer_success": sum(1 for row in rows if row["Style_Transfer_Success_Flag"]),
        "both": sum(
            1 for row in rows if row["Factuality_Preserved_Flag"] and row["Style_Transfer_Success_Flag"]
        ),
        "means": {
            name: _mean([row.get(name) for row in rows])
            for name in (
                "BLEU_4",
                "ROUGE_L",
                "BERTScore_BioClinical",
                "RaTEScore",
                "RadGraph_F1",
                "MSK_Diagnostic_F1",
                "GREEN_Score",
                "Style_Similarity",
            )
        },
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote {args.output}")
    print(f"Wrote {summary_path}")
    missing = [column for column in RESULT_COLUMNS if column not in rows[0]]
    if missing:
        raise RuntimeError(f"Result rows are missing columns: {missing}")
    return args.output


if __name__ == "__main__":
    main()
