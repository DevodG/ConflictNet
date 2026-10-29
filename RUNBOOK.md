# ConflictNet Runbook: from zero to reported numbers

## 0. Environment

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ruff check . && python -m pytest tests/ -q   # 156 passed expected
```

## 1. Data

```bash
python scripts/download_datasets.py --datasets cremad,meld,mustard
# IEMOCAP needs a USC license — place under data/iemocap manually.
```

## 2. Prosody statistics (train-only — never recompute on eval data)

```bash
python scripts/compute_prosody_stats.py \
  --cremad_root data/cremad --meld_root data/meld \
  --mustard_root data/mustard \
  --output_file prosody_stats.json
# → prosody_stats.train.zscores.json + prosody_stats.val.zscores.json
```

## 3a. Binary acted-emotion protocol (the 95+ headline scope)

```bash
python scripts/train.py --config configs/default.yaml \
  --cremad_root data/cremad --iemocap_root data/iemocap \
  --prosody_stats prosody_stats.json --amp \
  --output_dir checkpoints/binary_run1

python scripts/eval_cremad.py \
  --cremad_root data/cremad \
  --checkpoint checkpoints/binary_run1/best_model.safetensors \
  --prosody_stats prosody_stats.json --output_dir results/binary_run1
# Reports BINARY F1 ONLY — label it as acted-emotion conflict, never deception.
```

## 3b. Full conflict training (subtype heads)

```bash
python scripts/train.py --config configs/default.yaml \
  --iemocap_root data/iemocap --mustard_root data/mustard \
  --cremad_root data/cremad --meld_root data/meld \
  --prosody_stats prosody_stats.json --amp \
  --output_dir checkpoints/full_run1

python scripts/evaluate.py \
  --checkpoint checkpoints/full_run1/best_model.safetensors \
  --mustard_root data/mustard --iemocap_root data/iemocap \
  --prosody_stats prosody_stats.json \
  --fairness --ood_probe --dump_npz results/full_run1/model.npz \
  --output_dir results/full_run1
```

## 4. Statistics (required for every paper number)

```bash
python scripts/stats.py --npz results/full_run1/model.npz --metric macro_f1 \
  --baseline results/baseline.npz --output results/full_run1/stats.json
# Every table cell = point + bootstrap 95% CI; comparisons need paired p-values.
```

## 5. Ablations

```bash
python scripts/ablate.py --config_dir configs --iemocap_root data/iemocap \
  --mustard_root data/mustard --output_dir runs/abl1
python scripts/ablation_table.py --runs full,no_speaker_norm \
  --checkpoints_dir checkpoints --output ablation_table.tex
```

## Honesty gates (enforced in code — do not bypass)

- Eval refuses legacy combined `*.zscores.json` for reported metrics.
- Subtype loss trains only on `has_real_type_labels`; deception has no
  positives until CASE annotation lands.
- `retry_attempts` in checkpoint meta must be disclosed with any number.
- Thresholds freeze on val, applied once to test.
