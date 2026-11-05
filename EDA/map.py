from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, detect

EDA_DIR = Path(__file__).resolve().parent

# Set seed for deterministic language detection results
DetectorFactory.seed = 0


def detect_language(text):
    """Detects the language of a text string using langdetect.

    Returns 'unknown' if detection fails or text is invalid.
    """
    if pd.isna(text) or not str(text).strip():
        return "unknown"
    try:
        return detect(str(text))
    except Exception:
        # Handles cases where text is too short, numbers-only, or unrecognizable
        return "unknown"


def main():
    # 1. Read the input CSV file
    input_file = EDA_DIR / "mapped.csv"
    print(f"Loading '{input_file}'...")
    df = pd.read_csv(input_file)

    # 2. Apply language detection to the 'Report' column
    print("Detecting languages...")
    df["lan"] = df["Report"].apply(detect_language)

    # 3. Print breakdown of detected languages
    print("\nLanguage Counts:")
    print(df["lan"].value_counts())

    # 4. Save as CSV
    csv_output = EDA_DIR / "map_with_language.csv"
    df.to_csv(csv_output, index=False)
    print(f"\nSaved CSV to '{csv_output}'")

    # 5. Save as XLSX (Excel)
    xlsx_output = EDA_DIR / "map_with_language.xlsx"
    df.to_excel(xlsx_output, index=False, engine="openpyxl")
    print(f"Saved Excel file to '{xlsx_output}'")


if __name__ == "__main__":
    main()