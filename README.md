# ConflictNet v2

**Speaker-normalised cross-modal emotion conflict detection** with temporal context, interpretability, and multi-label subtype classification.

> Original implementation (~2,100 lines of novel modelling code) following the HuBERT-CLAP projection-head *pattern* (Nguyen et al., 2024) — no HuBERT-CLAP weights or code are reused.

---

## Architecture

```
Audio (wav) ──► Emotion2Vec / WavLM / wav2vec2 ──► ProjectionHead ──┐
                                                                      ├──► FusionGate ──► TransformerTemporalContext ──► ConflictClassifier
Text (str) ───► DeBERTa-v3 + LoRA ──────────────► ProjectionHead ──┤                                                     ├── logits_type (sarcasm/suppression/deception)
                                                                      │                                                     ├── severity [0,1]
ECAPA-TDNN ──► SpeakerNormalizer (z-score) ─────────────────────────┘                                                     └── conflict_flag (bool)
                                                       │
              MFA TextGrids ──► WordLevelDivergence ───┘ (optional)
```

**Losses (jointly optimised via Kendall 2018 uncertainty weighting):**
1. Context-Gated InfoNCE contrastive loss (audio ↔ text alignment)
2. Multi-label BCE for conflict subtypes
3. Severity MSE regression
4. Self-supervised swap detection (pre-training phase)

---

## Quick Start

```bash
conda create -n conflictnet python=3.10 -y
conda activate conflictnet
pip install -r requirements.txt
conda install -c conda-forge montreal-forced-aligner -y
mfa model download acoustic english_mfa
mfa model download dictionary english_us_arpa
```

### Reproduce baseline (IEMOCAP)
```bash
python scripts/train.py \
    --iemocap_root /data/iemocap \
    --audio_encoder emotion2vec \
    --epochs 30 \
    --pretrain_epochs 5 \
    --output_dir checkpoints/emotion2vec_run1
```

### Evaluate
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/emotion2vec_run1/best_model.safetensors \
    --iemocap_root /data/iemocap \
    --fairness \
    --output_dir results/
```

### Run tests (no GPU needed)
```bash
pytest tests/ -v
```

---

## Project Structure

```
conflictnet/
├── configs/
│   ├── default.yaml            # YAML config (train.py --config)
│   ├── fast_training.yaml      # Reduced-cost preset
│   └── ablate_*.yaml           # 7 ablation configs
├── data/
│   ├── datasets.py             # 7 dataset loaders + collate_fn + z-score lookups
│   ├── audio_io.py             # Backend-free waveform IO (soundfile)
│   ├── cached_dataset.py       # Pre-computed embedding dataset
│   ├── augmentation.py         # MUSAN noise / SpecAugment
│   └── synthetic.py            # StarGANv2-VC conflict pair generation
├── models/
│   ├── encoders/
│   │   ├── audio.py            # Emotion2Vec, WavLM, wav2vec2
│   │   └── text.py             # DeBERTa-v3-large + LoRA
│   ├── speaker_norm/
│   │   └── speaker_norm.py     # ECAPA-TDNN + prosody z-score + cold-start
│   ├── temporal/
│   │   └── temporal.py         # Causal Transformer over dialogue turns ⭐
│   ├── alignment/
│   │   ├── alignment.py        # ProjectionHead + ContextGatedContrastiveLoss ⭐
│   │   └── word_divergence.py  # MFA word-level divergence features ⭐
│   ├── classifier/
│   │   └── classifier.py       # Multi-label subtype + severity head ⭐
│   ├── conflictnet.py          # Full model assembly + MultiTaskLoss + SwapObjective
│   ├── cached_conflictnet.py   # Cached-embedding fast path
│   ├── checkpoint_utils.py     # safetensors IO + full-config restore
│   ├── experiment_config.py    # Structured run config
│   └── device_utils.py         # CUDA/MPS/CPU resolution
├── training/
│   ├── trainer.py              # Training loop + scheduler + metrics.json + WandB
│   └── curriculum.py           # CurriculumSampler ⭐
├── evaluation/
│   ├── metrics.py              # WAcc, macro-F1, per-type AP/AUC, severity MAE
│   ├── fairness.py             # FairLearn demographic parity + equalized odds
│   ├── attribution.py          # Captum integrated gradients (text + audio)
│   ├── calibration.py          # Threshold tuning + ECE + reliability diagrams
│   ├── llm_baseline.py         # GPT-4o text-only ceiling
│   ├── human_eval.py           # Annotator agreement (kappa)
│   ├── latency.py              # Latency benchmarking
│   └── ood_probe.py            # Held-out-speaker OOD evaluation
├── scripts/
│   ├── train.py / evaluate.py / benchmark.py / ensemble.py
│   ├── stats.py                # Bootstrap CIs + significance tests
│   ├── eval_cremad.py          # CREMA-D binary-only eval
│   ├── ablate.py               # Ablation runner
│   ├── compute_prosody_stats.py / compute_difficulties.py
│   ├── precompute_embeddings.py / precompute_incremental.py
│   ├── run_mfa_alignment.py / download_datasets.py / export_onnx.py
│   └── generate_synthetic.py / preprocess_all.py
├── serve/                      # FastAPI inference (REST + /ws streaming)
├── paper/                      # Paper drafts + verified references (06)
├── tests/
│   ├── test_components.py      # Unit tests (all components, no GPU required)
│   ├── test_evaluation.py      # Eval + fairness + calibration tests
│   ├── test_serve.py           # Serving tests
│   └── test_integrity.py       # Regression tests for audit fixes
├── lightning_train.py          # Lightning AI remote launcher (env-configured)
└── requirements.txt
```

⭐ = novel contribution

---

## Novel Components (~2,100 lines)

| Component | File | Lines | Novelty |
|-----------|------|-------|---------|
| Speaker normalization + cold-start | `models/speaker_norm/speaker_norm.py` | ~470 | ⭐ Novel |
| Transformer temporal context | `models/temporal/temporal.py` | ~160 | ⭐ Novel |
| Context-gated contrastive loss | `models/alignment/alignment.py` | ~250 | ⭐ Novel |
| Multi-label classifier + severity | `models/classifier/classifier.py` | ~220 | ⭐ Novel |
| Word-level divergence (MFA) | `models/alignment/word_divergence.py` | ~300 | ⭐ Novel |
| Self-supervised swap objective | `models/conflictnet.py` | ~50 | ⭐ Novel |
| Curriculum sampler | `training/curriculum.py` | ~75 | Semi-novel |
| Multi-task uncertainty loss | `models/conflictnet.py` | ~30 | Impl. |
| Full model assembly + forward | `models/conflictnet.py` | ~550 | Engineering |

---

## Datasets

| Dataset | Use | Source |
|---------|-----|--------|
| IEMOCAP | Primary train/eval | USC (request) |
| CREMA-D | Augmentation | HuggingFace |
| MUStARD++ | Sarcasm labels | GitHub |
| MELD | Dialogue context | HuggingFace |
| GoEmotions | Text pre-training | HuggingFace |
| VoxCeleb1/2 | Speaker embeddings | robots.ox.ac.uk |
| MUSAN | Noise augmentation | openslr.org |

---

## 12-Week Execution Plan

| Week | Milestone |
|------|-----------|
| 1 | Implement encoder stack (DeBERTa-v3 + audio backbones, HuBERT-CLAP-style projection pattern) |
| 2 | Swap wav2vec2→Emotion2Vec → compare all 3 audio encoders |
| 3 | Add ECAPA-TDNN speaker norm → z-score pipeline |
| 4 | Build Transformer temporal context → integrate |
| 5 | Multi-label classifier + severity head |
| 6 | MFA word alignment → per-word divergence features |
| 7 | Self-supervised swap pre-training → synthetic data via StarGANv2-VC |
| 8 | Multi-task loss balancing → curriculum learning → full training |
| 9 | Captum integrated gradients → attribution maps |
| 10 | Evaluation — fairness, latency, LLM baseline, human eval |
| 11 | Ablation studies → error analysis |
| 12 | Paper writing + architecture diagram |

---

## Dependencies

See `requirements.txt` (single source of truth):

```
torch>=2.1  torchaudio>=2.1  transformers>=4.36 (+sentencepiece)  peft>=0.7
speechbrain  captum  fairlearn  librosa  soundfile  praat-parselmouth
scikit-learn  scipy  pyyaml  pandas  matplotlib  wandb  audiomentations
safetensors  pytest  ruff  fastapi/uvicorn/pydantic  onnx (optional)
funasr  # optional, Emotion2Vec backend
# Via conda: montreal-forced-aligner
# Via git clone: github.com/yl4579/StarGANv2-VC
```
