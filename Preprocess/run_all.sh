#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

KAGGLE=${KAGGLE:-/workspace/kagglehub/competitions/rsna-knee-abnormality-detection}
OUT_ROOT=${OUT_ROOT:-/workspace/knee-mri-rrg/processed_h5}
WORKERS=${WORKERS:-4}

for split in train val test; do
    python Preprocess/preprocess.py \
        --train_csv "Split/${split}.csv" \
        --series_csv "$KAGGLE/train_series.csv" \
        --dicom_dir "$KAGGLE/train_series" \
        --output_dir "$OUT_ROOT/${split}" \
        --workers "$WORKERS"
done
