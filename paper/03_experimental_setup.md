# Experimental Setup

## 4.1 Datasets

We evaluate on five datasets spanning diverse domains and conflict subtypes:

**IEMOCAP** (Busso et al., 2008) contains 10,039 dyadic utterances from 10 actors across 5 sessions. Each utterance is annotated with categorical emotions. We map anger and frustration to emotional suppression (binary conflict = 1), all other emotions to no conflict (0). Following established protocol, we use leave-one-session-out 5-fold cross-validation, reporting the macro-average across folds. Pre-training uses sessions 1-4; fine-tuning and evaluation follow the standard session-5 held-out protocol.

**MUStARD++** (Castro et al., 2022) provides 1,200 sarcasm-labeled utterances from The Big Bang Theory and Friends. Sarcastic utterances are mapped to [sarcasm=1, suppression=0, deception=0].

**MELD** (Poria et al., 2019) contains 13,000+ utterances from Friends with multi-label emotion annotations. We map anger, disgust, and fear to emotional suppression. MELD provides dialogue context (multi-speaker, with speaker identities).

**CMU-MOSEI** (Zadeh et al., 2018) contains 23,453 YouTube opinion video clips annotated for emotion and sentiment. We map anger, disgust, and fear to suppression.

**CASE 2026** provides a custom benchmark with annotated conflict types (sarcasm, suppression, deception) for forensic and clinical scenarios.

All audio is resampled to 16 kHz mono, truncated/padded to 10 seconds. Text is tokenized with DeBERTa-v3's tokenizer (max 512 tokens). Table 1 summarizes the datasets.

| Dataset | Size | Subtypes | Domain | Role |
|---------|------|----------|--------|------|
| IEMOCAP | 10,039 | Suppression | Acted dyadic | Primary (5-fold) |
| MUStARD++ | 1,200 | Sarcasm | TV sitcoms | Sarcasm evaluation |
| MELD | 13,000+ | Suppression | TV sitcom | Dialogue evaluation |
| CMU-MOSEI | 23,453 | Suppression | YouTube | Multi-speaker eval |
| CASE 2026 | ~4,000 | All 3 | Forensic/clinical | Benchmark |

## 4.2 Training Configuration

We train with AdamW (learning rate 2e-5, weight decay 0.01, β=[0.9, 0.999]) with linear warmup over 500 steps followed by cosine annealing to zero. The effective batch size is 32 (batch_size=32, gradient_accumulation=1). Training proceeds in two phases: pre-training for 5 epochs (swap detection only, no labels required) followed by supervised fine-tuning for 25 epochs with curriculum sampling. Curriculum difficulty is defined as the frame-level cosine distance between audio and text embeddings from a randomly-initialized baseline model; the sampler linearly ramps from easy-only (epoch 0) to all-examples (epoch 25). Early stopping (patience=10 on val macro-F1) is applied during fine-tuning. We use automatic mixed precision (fp16) when available.

**Data augmentation.** During training, we apply speed perturbation (factors 0.9, 1.0, 1.1, p=0.5), MUSAN noise injection (p=0.3), and SpecAugment-style time masking (p=0.3). Augmentations are composed with global probability 0.5.

## 4.3 Baselines

We compare against four categories of baselines:

**Unimodal audio.** A WavLM-large classifier (frozen encoder + trained projection + classification head) operating on audio alone, and an Emotion2Vec+ variant of the same architecture.

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

Our primary metric is **macro-averaged F1** across the three conflict types (sarcasm, suppression, deception). We additionally report:

- **Per-type F1, average precision (AP), and AUC-ROC** for individual subtypes.
- **Binary F1, accuracy, and AUC** for conflict-vs-no-conflict detection.
- **Weighted accuracy (WAcc)**, where sample weights are inversely proportional to class frequency.
- **Severity mean absolute error (MAE)** for the regression head.

All metrics are computed per dataset and aggregated as mean ± std across datasets. For IEMOCAP, we additionally report 5-fold cross-validated metrics.

## 4.6 Implementation Details

Models are implemented in PyTorch 2.x and trained on a single NVIDIA A100 (80 GB) GPU. Training takes approximately 6 hours (5 epochs pre-training + 25 epochs fine-tuning). The codebase and trained models will be released upon publication.
