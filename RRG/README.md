# Stage 4: Radiology Report Generation

Condition a LoRA-tuned causal LM on Stage-3 multi-view MRI embeddings via a
trainable visual prefix projector.

## Layout

| Artifact | Path |
|---|---|
| Stage-3 vision | `/workspace/knee-mri-rrg/checkpoints/stage3/best_visual_encoder.pt` |
| Stage-3 full (optional init) | `.../best_pretraining.pt` |
| HDF5 splits | `/workspace/knee-mri-rrg/processed_h5/{train,val,test}` |
| Stage-4 outputs | `/workspace/knee-mri-rrg/checkpoints/stage4` |

Code: `rrg_models.py`, `train_rrg.py`.

## Architecture

1. `MultiViewVisionEncoder` → study vector `[B, 768]`
2. `VisualPrefixProjector` → prefix tokens `[B, P, H_lm]` (default `P=32`)
3. Prefix tokens prepended to decoder token embeddings
4. Causal LM loss on report tokens only (prefix positions labeled `-100`)
5. LoRA on decoder attention/MLP projections

## Phase 1 — frozen vision

```bash
source /venv/main/bin/activate
cd /root/knee-mri
python -u train_rrg.py \
  --stage3-dir /workspace/knee-mri-rrg/checkpoints/stage3 \
  --vision-checkpoint best_visual_encoder.pt \
  --h5-root /workspace/knee-mri-rrg/processed_h5 \
  --output-dir /workspace/knee-mri-rrg/checkpoints/stage4 \
  --decoder Qwen/Qwen2.5-1.5B-Instruct \
  --batch-size 1 \
  --accumulation-steps 8 \
  --epochs 5 \
  --lr 1e-4 \
  --precision bf16 \
  --device cuda
```

## Phase 2 — joint fine-tune

Resume Phase-1 `best/` (projector + LoRA), swap vision to Stage-3
`best_pretraining.pt`, and unfreeze the encoder at a lower LR:

```bash
python -u train_rrg.py \
  --init-from /workspace/knee-mri-rrg/checkpoints/stage4/best \
  --unfreeze-vision \
  --vision-checkpoint best_pretraining.pt \
  --vision-lr 1e-6 \
  --lr 1e-4 \
  --batch-size 1 \
  --accumulation-steps 8 \
  --epochs 5 \
  --precision bf16 \
  --output-dir /workspace/knee-mri-rrg/checkpoints/stage4-phase2
```

`--unfreeze-vision` enables vision grads and uses `--vision-lr` for the encoder
while LoRA/projector keep `--lr`.

## Checkpoints

`checkpoints/stage4/best/` contains:

- `lora/` — PEFT adapters
- `rrg_projector.pt` — prefix projector (+ metadata)
- `tokenizer/`
- `visual_encoder.pt` — only when vision was unfrozen

## Phase 1 test generation

```bash
python -u evaluate_phase1_test.py \
  --checkpoint-dir /workspace/knee-mri-rrg/checkpoints/stage4/best \
  --output-dir /workspace/knee-mri-rrg/results
```

Writes `phase1_test_predictions.csv` / `.xlsx` and `phase1_test_metrics.json`
under `--output-dir`. Every report row has sentence BLEU-1..4 and
ROUGE-1/2/L F1. The JSON BLEU scores are corpus sacreBLEU. The JSON ROUGE
scores are the mean of the per-report F1 values. All scores are in [0, 1].

Add per-report GREEN (0–1) with the local `StanfordAIMI/GREEN-RadLlama2-7b` model.
The dataset summary is `green` and `green_std` in the metrics JSON. Each row
also stores matched-finding and significant-error counts plus the model analysis.

```bash
python -u evaluate_phase1_test.py --score-only --green \
  --predictions /workspace/knee-mri-rrg/results/phase1_test_predictions.csv \
  --output-dir /workspace/knee-mri-rrg/results
```

Add per-report BERTScore (precision, recall, and F1, all in [0, 1]) with
`emilyalsentzer/Bio_ClinicalBERT`. The JSON field `bertscore` is the mean F1.

```bash
python -u evaluate_phase1_test.py --score-only --bertscore \
  --predictions /workspace/knee-mri-rrg/results/phase1_test_predictions.csv \
  --output-dir /workspace/knee-mri-rrg/results
```

Add per-report CRIMSON with `rajpurkarlab/medgemma-4b-it-crimson`. Unlike the
other metrics, CRIMSON is in **(-1, 1]**: 1 is perfect, 0 matches a normal
template, and negative scores mean weighted errors exceed correct findings.
The published model was trained on chest X-ray reports.

```bash
python -u evaluate_phase1_test.py --score-only --crimson \
  --predictions /workspace/knee-mri-rrg/results/phase1_test_predictions.csv \
  --output-dir /workspace/knee-mri-rrg/results
```

Rescore BLEU and ROUGE without loading a model:

```bash
python -u evaluate_phase1_test.py --score-only \
  --predictions /workspace/knee-mri-rrg/results/phase1_test_predictions.csv \
  --output-dir /workspace/knee-mri-rrg/results
```

## Tests

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/test_rrg.py -q
```
