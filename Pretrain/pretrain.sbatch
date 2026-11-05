#!/usr/bin/env bash
#SBATCH --job-name=knee-stage3
#SBATCH --partition=overflow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere|ada-6000|ada-l40s|hopper
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --no-requeue
#SBATCH --output=Pretrain/stage3-%j.out
#SBATCH --error=Pretrain/stage3-%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-/root/knee-mri}}
PYTHON=${PYTHON:-/venv/main/bin/python}
H5_ROOT=${H5_ROOT:-/workspace/knee-mri-rrg/processed_h5}
OUTPUT_DIR=${OUTPUT_DIR:-/workspace/knee-mri-rrg/checkpoints/stage3-${SLURM_JOB_ID:?Submit with sbatch}}

cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

printf 'Job %s on %s; Python: %s\n' "$SLURM_JOB_ID" "$(hostname)" "$PYTHON"
printf 'HDF5 root: %s\nCheckpoint directory: %s\n' "$H5_ROOT" "$OUTPUT_DIR"
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable on allocated node"; assert torch.cuda.is_bf16_supported(), "Allocated GPU lacks BF16 support"; print("GPU:", torch.cuda.get_device_name(0), "PyTorch:", torch.__version__)'

exec srun --ntasks=1 "$PYTHON" -u pretrain.py \
    --h5-root "$H5_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda \
    --precision bf16 \
    --batch-size 16 \
    --accumulation-steps 4 \
    --num-workers 2 \
    "$@"