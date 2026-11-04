"""Stratified 80/10/10 split of EDA/map_with_language.csv by (site_name, lan)."""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "EDA" / "map_with_language.csv"
OUT_DIR = Path(__file__).resolve().parent
KEY = "_strat_key"


def stratified_split(df: pd.DataFrame, key: str, test_frac: float, seed: int):
    """Per-stratum shuffle and cut; returns (keep, holdout)."""
    rng = np.random.default_rng(seed)
    keep_idx, hold_idx = [], []
    for _, grp in df.groupby(key, sort=True):
        idx = rng.permutation(grp.index.to_numpy())
        n_hold = int(round(len(idx) * test_frac))
        hold_idx.extend(idx[:n_hold])
        keep_idx.extend(idx[n_hold:])
    return df.loc[keep_idx], df.loc[hold_idx]


def main():
    df = pd.read_csv(SRC)

    site = df["site_name"].astype("string").str.strip()
    blank_mask = site.isna() | (site == "")
    blanks = df[blank_mask]
    valid = df[~blank_mask].copy()

    valid[KEY] = valid["site_name"].astype(str) + "_" + valid["lan"].astype(str)
    counts = valid[KEY].map(valid[KEY].value_counts())
    singletons = valid[counts == 1]
    regular = valid[counts > 1]

    train_part, temp = stratified_split(regular, KEY, test_frac=0.2, seed=SEED)
    val, test = stratified_split(temp, KEY, test_frac=0.5, seed=SEED + 1)

    train = pd.concat([blanks, singletons, train_part])
    splits = {"train": train, "val": val, "test": test}

    total = len(df)
    for name, split in splits.items():
        split = split.drop(columns=[KEY, "site_name", "lan"], errors="ignore").sample(
            frac=1, random_state=SEED
        )
        split.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"{name:5s}: {len(split):5d} ({len(split) / total:.1%})")

    print(f"blanks -> train: {len(blanks)}, singletons -> train: {len(singletons)}")
    assert sum(len(s) for s in splits.values()) == total


if __name__ == "__main__":
    main()
