from pathlib import Path

import pandas as pd

# Load your processed data file
df = pd.read_csv(Path(__file__).resolve().parent / "map_with_language.csv")

# ---------------------------------------------------------
# 1. Counts by site_name and lan (including blanks)
# ---------------------------------------------------------
site_lan_counts = (
    df.groupby(["site_name", "lan"], dropna=False)
    .size()
    .reset_index(name="count")
)

# Replace NaN values with '(Blank)' for readability
site_lan_counts["site_name"] = site_lan_counts["site_name"].fillna("(Blank)")
site_lan_counts["lan"] = site_lan_counts["lan"].fillna("(Blank)")

print("=" * 50)
print("COUNTS BY SITE NAME AND LANGUAGE")
print("=" * 50)
print(site_lan_counts.to_string(index=False))
print("\n" + "=" * 50)

# ---------------------------------------------------------
# 2. Overall Stats by Language (Total counts & percentages)
# ---------------------------------------------------------
lang_stats = (
    df["lan"]
    .value_counts(dropna=False)
    .reset_index()
)
lang_stats.columns = ["lan", "total_count"]
lang_stats["lan"] = lang_stats["lan"].fillna("(Blank)")

# Calculate percentage share across the dataset
total_rows = len(df)
lang_stats["percentage"] = (lang_stats["total_count"] / total_rows * 100).round(2)

print("OVERALL STATS BY LANGUAGE")
print("=" * 50)
print(lang_stats.to_string(index=False))
print(f"\nTotal Records: {total_rows}")
print("=" * 50)