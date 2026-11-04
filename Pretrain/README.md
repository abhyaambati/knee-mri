# Stage 3: Contrastive Pretraining

Run from the repository root with `torch`, `transformers`, `tokenizers`,
`sentencepiece`, `pandas`, `numpy`, and `h5py` installed. A CUDA GPU with BF16
support is required for the default configuration. Initial startup downloads
ViT-B/16, its image processor, and XLM-R-base plus its tokenizer; local Hugging
Face model directories can be supplied instead.

```bash
python pretrain.py \
  --h5-root /opt/scratchspace/aambat3/knee-mri-rrg/processed_h5 \
  --output-dir checkpoints/stage3
```

The root contains `train/` and `val/`, matching `Preprocess/run_all.sh`.
For other layouts, pass `--train-h5-dir` and `--val-h5-dir` explicitly.
Training and validation CSVs default to `Split/train.csv` and `Split/val.csv`.
Study IDs must be unique within each split and disjoint between splits.

## Slurm Batch Training

Submit from the repository root, without an interactive GPU session:

```bash
sbatch Pretrain/pretrain.sbatch
squeue -u "$USER"
```

The batch script requests one GPU (not an exclusive node), 8 CPUs, 64 GiB RAM,
and 48 hours on `overflow`. Its `cuda13` constraint matches the installed
CUDA 13 PyTorch build and the cluster's BF16-capable Ada/Hopper nodes. Actual
start time still depends on scheduler priority and available GPU/CPU/RAM.
Override resource directives with `sbatch` options when needed.

Logs appear as `Pretrain/stage3-JOBID.out` and `.err`. Checkpoints go to
`/opt/scratchspace/$USER/knee-mri-rrg/checkpoints/stage3-JOBID/` by default.
`PROJECT_DIR`, `PYTHON`, `H5_ROOT`, and `OUTPUT_DIR` can be set in the submitting
shell to override paths. Additional trainer arguments can follow the script:

```bash
sbatch Pretrain/pretrain.sbatch --epochs 20 --slice-batch-size 16
```

The job checks CUDA/BF16 availability before starting. Automatic requeue is
disabled because the trainer does not implement checkpoint resume: a restart
would train from epoch one. Preemption or the wall-time limit can still end a
run; only completed best-validation checkpoints are saved. Initial model
downloads require network access or an existing shared Hugging Face cache.

## Architecture

The shared ViT produces a 768-dimensional CLS vector for each slice. Learned
depth positions and 8-head SDPA attention aggregate 24 slices per view. Learned
plane identities and cross-view attention fuse sagittal/coronal/axial vectors.
The vision encoder normalizes the incoming [0, 1] pixels using the pretrained
image processor's statistics. XLM-R-base uses attention-mask-weighted mean
pooling over at most 384 tokens. Separate two-layer MLPs project both vectors
to L2-normalized 256-dimensional embeddings.

## Training And Evaluation

Defaults are batch size 16, four gradient accumulation steps, BF16, gradient
checkpointing, AdamW with learning rate `1e-4` and weight decay `0.01`, and 10%
linear warmup followed by cosine decay. The learned inverse temperature starts
at `1 / 0.07` and is capped at 100. Gradients are clipped to norm 1.

Each complete microbatch has 15 negatives per anchor. Accumulation gives a
nominal optimizer update size of 64, not a 64-pair negative pool. Invalid HDF5
fallbacks and blank reports are excluded; fewer than two usable pairs skip a
training microbatch. Remaining short batches have fewer negatives, and partial
accumulation windows are averaged by their actual microbatch count. The learning
rate schedule uses the nominal number of updates; skipped batches can reduce
the number of executed schedule steps.

Each epoch reports full-validation-pool symmetric InfoNCE loss and image-to-text
and text-to-image Top-1/Top-5 recall as fractions. Similarities are computed in
chunks on CPU, avoiding storage of the full square matrix. Top-5 uses all
candidates when fewer than five exist. Validation loss is not directly
comparable to training loss because its negative pool is larger.

MRI batches are large: batch size 16 alone uses about 662 MiB of float32 image
storage. Worker prefetch and model activations require additional memory.
Reduce `--slice-batch-size` for activation pressure; reduce `--num-workers` for
host/shared-memory pressure. Reducing `--batch-size` also reduces negatives.
This trainer is single-device, not distributed. CPU debugging is supported via
`--device cpu --precision fp32`; full-size training is intended for a GPU.

## Stage 4 Handoff

The lowest validation loss saves:

- `best_visual_encoder.pt`: standalone vision state dictionary, including ViT,
  depth/view fusion, positional parameters, and pixel normalization buffers.
- `visual_encoder_config.json`: architecture settings for offline reconstruction.
- `best_pretraining.pt`: complete contrastive model, optimizer, scheduler,
  metrics, backbone configs, epoch, and training arguments.
- `tokenizer/`: tokenizer saved at training startup.

The standalone visual encoder returns a 768-dimensional study vector by
default; the contrastive 256-dimensional projection is in the full checkpoint.
Reconstruct the Stage 4 encoder without downloading pretrained weights:

```python
import json
from pathlib import Path

import torch
from transformers import ViTConfig, ViTModel

from models import MultiViewVisionEncoder

directory = Path("checkpoints/stage3")
config = json.loads((directory / "visual_encoder_config.json").read_text())
backbone = ViTModel(ViTConfig.from_dict(config.pop("vision_config")), add_pooling_layer=False)
config.pop("view_order")
encoder = MultiViewVisionEncoder(backbone, gradient_checkpointing=False, **config)
encoder.load_state_dict(torch.load(directory / "best_visual_encoder.pt", map_location="cpu", weights_only=True))
encoder.eval()
```

Offline verification, including a complete tiny-model training run:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/test_dataset.py tests/test_pretrain.py -q
```