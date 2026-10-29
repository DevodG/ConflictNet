# ConflictNet — what is what (Oct 29 2026)

Audio+text conflict detection (sarcasm / suppression / deception + severity).
Target: Interspeech 2027 long paper, deadline Feb 9 2027.

## State: code green, data partial, no full training run yet

- Gates: `ruff check .` clean, `pytest tests/ -q` 160/160 green.
- Pushed to `origin main` through `4bc8f9f`; tree clean.

## Key files

| Path | What it is |
|---|---|
| `RUNBOOK.md` | Zero-to-numbers pipeline (§0 env → §5 ablations). Start here. |
| `paper/literature_tracker.xlsx` | 118 papers (USE/SKIP + why + params + findings), 7 SOTA anchors, 22-row Optimal-path plan. |
| `paper/01–06_*.md` | Paper draft (intro → references). Numbers still TBD placeholders. |
| `scripts/train.py` | Training entry. Refuses fallback encoders unless `--allow_fallback_encoders`. |
| `scripts/eval_cremad.py` | Binary CREMA-D eval (the scoped 95+ protocol). Reports binary F1 only. |
| `scripts/stats.py` | Bootstrap 95% CIs + paired tests. Every paper number must come from here. |
| `scripts/compute_prosody_stats.py` | Train-only prosody stats → split z-scores (no leakage). |
| `data/datasets.py` | All dataset loaders + `load_prosody_lookups`. Eval refuses legacy combined z-scores. |
| `models/conflictnet.py` | Full model (WavLM + DeBERTa-LoRA + speaker-norm + temporal + hinge). |
| `prosody_stats.json` | Real CREMA-D stats (73 speakers, 5,885 train utts). Split files are gitignored build outputs. |

## Data on this machine

| Dataset | State |
|---|---|
| CREMA-D | Ready: 7,442 clips in `data/cremad/` (Kaggle). |
| MELD | Tarball only (`data/meld/MELD.Raw.tar.gz`); audio download failed. |
| MUStARD++ | Text/JSON only; audio needs manual download. |
| IEMOCAP | Absent — needs USC license. |

## What is proven vs pending

- Proven: full pipeline to the training gate (dry run passes shapes/loss); prosody stats; eval + stats scripts; 160 tests.
- Pending: first real training run. Blocked on this Mac by full disk (107MB free; need ~6GB for DeBERTa weights + checkpoints) — see next step.

## Next step (needs ~8GB free or a CUDA box)

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/train.py --config configs/default.yaml \
  --cremad_root data/cremad --prosody_stats prosody_stats.json \
  --output_dir checkpoints/binary_run1 --batch_size 8 --pretrain_epochs 0 --amp
python scripts/eval_cremad.py --cremad_root data/cremad \
  --checkpoint checkpoints/binary_run1/best_model.safetensors \
  --prosody_stats prosody_stats.json --output_dir results/binary_run1
```
