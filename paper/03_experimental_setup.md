# Experimental Setup

## 4.1 Datasets

We evaluate on four public datasets spanning diverse domains and conflict subtypes (plus a pending custom CASE annotation split, §4.1.1):

**IEMOCAP** (Busso et al., 2008) contains 10,039 dyadic utterances from 10 actors across 5 sessions. Each utterance is annotated with categorical emotions. We map anger and frustration to emotional suppression (binary conflict = 1), all other emotions to no conflict (0). We train on sessions 1–4 and evaluate on the held-out session 5 (speaker-disjoint). Leave-one-session-out 5-fold CV is planned as an extension; all current numbers are single-holdout and must be read as such.

**MUStARD++** (Ray et al., 2022; built on MUStARD, Castro et al., 2019) provides 1,202 sarcasm-labeled utterances from The Big Bang Theory and Friends. Sarcastic utterances are mapped to [sarcasm=1, suppression=0, deception=0].

**MELD** (Poria et al., 2019) contains 13,000+ utterances from Friends with multi-label emotion annotations. We map anger, disgust, and fear to emotional suppression. MELD provides dialogue context (multi-speaker, with speaker identities).

**CMU-MOSEI** (Zadeh et al., 2018) contains 23,453 YouTube opinion video clips annotated for emotion and sentiment. We map anger, disgust, and fear to suppression.

### 4.1.1 CASE (pending custom annotation)

**CASE** is a planned custom annotation split covering all three conflict subtypes (sarcasm, suppression, deception) with severity. It is NOT a public release: no data exists yet, and no result in this paper depends on it. Until annotation completes (we follow the PodSarc-style LLM + human-verification recipe, Li et al., 2026), deception has zero positive labels repo-wide and severity is unannotated — subtype macro-F1 is therefore reported only where real labels exist (MUStARD++ sarcasm), and severity MAE is reported only when annotated.

All audio is resampled to 16 kHz mono, truncated/padded to 10 seconds. Text is tokenized with DeBERTa-v3's tokenizer (max 512 tokens). Table 1 summarizes the datasets.

| Dataset | Size | Subtypes | Domain | Role |
|---------|------|----------|--------|------|
| IEMOCAP | 10,039 | Suppression-proxy | Acted dyadic | Primary (sess-5 holdout) |
| MUStARD++ | 1,202 | Sarcasm | TV sitcoms | Sarcasm evaluation |
| MELD | 13,000+ | Suppression-proxy | TV sitcom | Dialogue evaluation |
| CMU-MOSEI | 23,453 | Suppression-proxy | YouTube | Multi-speaker eval |
| CASE (pending) | — | All 3 + severity | Forensic/clinical | Future benchmark |

## 4.2 Training Configuration

We train with AdamW (learning rate 2e-5, weight decay 0.01, β=[0.9, 0.999]) with linear warmup over 500 steps followed by cosine annealing to zero. The training batch size is 16 by default (32 in `configs/default.yaml`; see `scripts/train.py --config`). Training proceeds in two phases: pre-training for 5 epochs (swap detection + contrastive only; supervised subtype/severity losses are disabled in this phase) followed by supervised fine-tuning. Curriculum sampling is opt-in (`--curriculum` + `scripts/compute_difficulties.py`): difficulty is the model-uncertainty proxy 1 − max sigmoid probability, and the sampler ramps from the easy third (warmup epochs) to all examples. Early stopping (patience=10) and checkpoint selection use binary weighted-F1 by default (`--val_metric val/macro_f1` switches to subtype macro-F1); both are logged every epoch so the two can be compared. Mixed precision is opt-in (`--amp` fp16 / `--bf16`).

**Data augmentation.** During training, we apply speed perturbation (factors 0.9, 1.0, 1.1, p=0.5), MUSAN noise injection (p=0.3), and SpecAugment-style time masking (p=0.3). Augmentations are composed with global probability 0.5.

## 4.3 Baselines

We compare against four categories of baselines:

**Unimodal audio.** A WavLM base-plus classifier (frozen encoder + trained projection + classification head) operating on audio alone, and an Emotion2Vec+ variant of the same architecture.

**Unimodal text.** DeBERTa-v3-large with LoRA (same configuration as our text encoder) with a classification head, operating on text alone.

**Multimodal (no speaker norm).** Our full architecture with speaker normalization disabled (speaker_feat = zeros), measuring the contribution of speaker-invariant features.

**LLM ceiling.** GPT-4o (zero-shot, temperature=0, structured JSON prompt) classifying conflict from transcript alone, providing a text-only upper bound on what can be achieved without audio.

## 4.4 Ablation Studies

We conduct seven ablation experiments by disabling individual architectural components:

1. **No cross-modal attention**: disables the audio↔text cross-attention layers; modalities interact only through the fusion gate.
2. **No temporal context**: removes the causal Transformer; each utterance processed independently.
3. **No speaker normalization**: removes ECAPA-TDNN and prosody features; speaker_feat = zeros.
4. **No word-level divergence**: disables MFA-based word-level divergence features.
5. **No baseline subtract**: switches from EMA neutral-centroid to standard z-score normalization.
6. **No adaptive threshold**: uses fixed 0.5 threshold instead of speaker-adaptive offset.
7. **No pre-training**: skips the self-supervised swap phase; trains from scratch on supervised data only.

Each ablation preserves all other components, ensuring isolated measurement of each contribution.

## 4.5 Evaluation Metrics

Our primary metric is **macro-averaged F1** across the conflict types that have real labels in the evaluated split (fixed 0.5 threshold); `evaluation/metrics.py` records `empty_classes` and warns whenever a subtype has zero positives, since the macro average over a missing class is capped below 1.0 and is not comparable to a full 3-type macro-F1. We additionally report:

- **Per-type F1, average precision (AP), and AUC-ROC** for individual subtypes (AP/AUC only where positives exist).
- **Binary F1, accuracy, and AUC** for conflict-vs-no-conflict detection.
- **Weighted accuracy (WAcc)**, where sample weights are inversely proportional to class frequency.
- **Severity mean absolute error (MAE)** for the regression head (annotated splits only).

All metrics are computed per dataset. Every number is reported with a bootstrap 95% CI (`scripts/stats.py`, 10k resamples) and model-vs-baseline comparisons use paired bootstrap + McNemar tests. Thresholds are frozen on validation and applied once to test; any `--target_f1` retry attempts are disclosed via `retry_attempts` in the checkpoint metadata.

## 4.6 Binary acted-emotion protocol (the 95+ headline scope)

Our 95+ F1 target applies ONLY to binary acted-emotion conflict (e.g. CREMA-D anger/disgust/fear-vs-rest, IEMOCAP anger/frustration-vs-rest) with speaker-disjoint splits, evaluated with `scripts/eval_cremad.py` (binary-only by construction). These are emotion proxies, honestly labeled as such — not genuine deception/conflict detection, where published multimodal SOTA sits at ~70–74% F1 (§6 references).

## 4.7 Implementation Details

Models are implemented in PyTorch 2.x. Reference training uses a single NVIDIA A100 (80 GB) GPU; wall-clock time will be recorded from the first completed run (no full run has completed yet — the previous "~6h" estimate was aspirational and is withdrawn). The codebase and trained models will be released upon publication.
