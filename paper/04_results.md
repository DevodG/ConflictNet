# Results

> **Status:** Placeholder — to be populated after training runs with the corrected codebase.
> Reporting protocol (binding): every cell carries a bootstrap 95% CI from
> `scripts/stats.py`; subtype macro-F1 only where real labels exist
> (`empty_classes` must be empty for a full 3-type claim); CREMA-D column is
> BINARY F1 only (no real subtype labels); thresholds frozen on val, applied
> once to test; `retry_attempts` disclosed per run.

---

## Main Results

Conflict detection performance on held-out sets.

| Model | IEMOCAP sess-5 (binary F1) | MUStARD++ val (sarcasm F1) | CREMA-D val (binary F1) | Severity MAE |
|-------|----------------------------|----------------------------|-------------------------|--------------|
| WavLM+DeBERTa late-fusion (baseline) | — | — | — | — |
| ConflictNet (ours) | — | — | — | — |
| _vs. published SOTA_ | — | — | — | — |

> **Dataset definitions:**
> - IEMOCAP: sessions 1–4 train, session 5 held-out (speaker-disjoint). Binary F1 on the anger/frustration suppression-proxy.
> - MUStARD++: speaker-stratified 80/20 split (seed 42). Sarcasm F1 (the only real subtype labels).
> - CREMA-D: speaker-stratified 80/20 split (actors not shared across splits). Binary F1 via `scripts/eval_cremad.py`.
> - Severity MAE: annotated splits only (currently none — column stays empty until CASE annotation).

---

## Ablation Study

Effect of each component on weighted F1 (IEMOCAP held-out test).

| Variant | F1 | \(\Delta\) |
|---------|-----|------------|
| Full ConflictNet | — | — |
| – Speaker normalisation | — | — |
| – Temporal context | — | — |
| – Word-level divergence | — | — |
| – Cross-modal attention | — | — |
| – Context-gated contrastive | — | — |
| – Speaker-adaptive threshold | — | — |
| – Baseline subtract | — | — |

---

## Fairness Audit

Demographic parity and equalised-odds differences across gender groups (IEMOCAP).

| Metric | Male | Female | Max gap |
|--------|------|--------|---------|
| F1 score | — | — | — |
| Selection rate | — | — | — |
| DP diff | — | — | — |
| EO diff | — | — | — |

---

## Latency

| Batch size | Avg (ms) | Std (ms) |
|------------|----------|----------|
| 1 | — | — |
| 8 | — | — |
| 32 | — | — |

Hardware: —

---

## LLM Baseline Comparison (MUStARD++)

| Model | F1 | Notes |
|-------|-----|-------|
| GPT-4o (text-only) | — | Zero-shot, prompt from §3 |
| ConflictNet (audio+text) | — | — |

---

## Sample Efficiency

| % training data | F1 |
|----------------|-----|
| 10% | — |
| 25% | — |
| 50% | — |
| 100% | — |
