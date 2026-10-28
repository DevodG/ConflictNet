# ConflictNet References (verified 2026-10-28)

Every entry below was checked against the publisher / arXiv record. Entries
marked CORRECTION differ from what the paper drafts currently cite — the
drafts must be updated before submission.

## Datasets

- Busso et al., 2008. *IEMOCAP: Interactive emotional dyadic motion capture
  database.* Language Resources and Evaluation. The primary train/eval corpus
  (10 actors, 5 sessions, ~10k utterances).
- Castro et al., 2019. *Towards multimodal sarcasm detection (An _Obviously_
  perfect paper).* ACL 2019 (P19-1455). The original MUStARD corpus (TV-show
  clips with dialogue context). CORRECTION: drafts cite "Castro et al., 2022"
  for MUStARD++ — Castro is 2019 and is the base MUStARD paper.
- Ray et al., 2022. *MUStARD++* — extension of MUStARD to ~1.2k utterances
  with emotion, valence, arousal and sarcasm-type labels. CORRECTION: this is
  the correct MUStARD++ citation, not Castro 2022.
- Poria et al., 2019. *MELD: A multimodal multi-party dataset for emotion
  recognition in conversations.* ACL 2019. ~13k Friends utterances.
- Zadeh et al., 2018. *CMU-MOSEI.* ~23k YouTube opinion clips.
- Li, Zhang, Gao, Nayak & Coler, 2025/2026. *Leveraging LLMs for sarcastic
  speech annotation in sarcasm detection.* Interspeech 2025 (arXiv:2506.00955,
  rev. Apr 2026). Introduces **PodSarc**, a large-scale sarcastic-speech
  dataset built with a GPT-4o + LLaMA-3 annotation pipeline + human
  verification; speech-only detector reaches 73.63% F1. NEW — directly
  relevant as a data source and as an annotation recipe for our missing
  deception/suppression labels.
- CASE 2026: NO VERIFIABLE RECORD. The "CASE 2026 forensic/clinical" dataset
  cited in drafts (placeholder github.com/anonymous/case2026) does not exist
  publicly. Do not cite it until a real release appears.

## Speech & text encoders (our backbones — all still current)

- Ma et al., 2024. *emotion2vec: Self-supervised pre-training for speech
  emotion representation.* ACL Findings 2024 (arXiv:2312.15185); emotion2vec+
  large released May 2024. CORRECTION: drafts cite "Peng et al., 2024".
  No newer emotion foundation model supersedes it as of Oct 2026
  (Emotion2Vec-S is a small-data variant, Feb 2025).
- Chen et al., 2022. *WavLM: Large-scale self-supervised pre-training for
  full stack speech processing.* IEEE JSTSP. Alternative audio backbone.
- Baevski et al., 2020. *wav2vec 2.0.* NeurIPS 2020. Alternative backbone.
- He et al., 2021/2023. *DeBERTaV3: Improving DeBERTa using ELECTRA-style
  pre-training.* ICLR 2023. Our text backbone (frozen + LoRA r=16).
- Hu et al., 2022. *LoRA: Low-rank adaptation of large language models.*
  ICLR 2022 (NeurIPS 2021 ws). PEFT method for the text encoder.
- Desplanques et al., 2020. *ECAPA-TDNN.* Interspeech 2020. Speaker
  embeddings for the normalisation pipeline.

## Methods we build on

- Nguyen et al., 2024. *HuBERT-CLAP: Contrastive learning-based multimodal
  emotion recognition using self-alignment.* MMAsia 2024 (ACM
  10.1145/3696409.3700183). CORRECTION: drafts cite "Liang et al., 2024".
  Note: we reuse its projection-head *pattern*, not its weights — the "70%
  reused / fork" wording in drafts is false and must go.
- Van den Oord et al., 2018. *Representation learning with contrastive
  predictive coding.* arXiv:1807.03748. InfoNCE (our contrastive loss base).
- Kendall, Gal & Cipolla, 2018. *Multi-task learning using uncertainty to
  weigh losses.* CVPR 2018. Our multi-task weighting (learned log σ²).
- McAuliffe et al., 2017. *Montreal Forced Aligner.* Interspeech 2017.
  Word-level alignment for the divergence features.
- Khan et al., 2025. *MemoCMT: Multimodal emotion recognition using
  cross-modal transformer-based feature fusion.* Scientific Reports 15.
  CORRECTION: drafts cite "Li et al., 2025". Closest published
  audio+text cross-modal fusion baseline — implement or cite as baseline.
- Gao et al., 2025 / Latif et al., 2025 (speaker-brittleness claims in
  intro §1): NOT VERIFIED. Either confirm exact papers or drop the sentence
  and argue from our own ablations.

## SOTA anchors (what "good" looks like — scoping our claims)

- Multimodal sarcasm (MUStARD/MUStARD++): ~70–72% F1 (Ray et al.; best
  reported fusion numbers). PodSarc speech-only: 73.63% F1 (Li et al. 2026).
- IEMOCAP 4-class emotion: multimodal ~74% WAcc (concept-bottleneck,
  arXiv:2609.39453, Sep 2026); audio+text fusion typically 69–74%.
- Implication: a 95+ macro-F1 claim is defensible ONLY on narrow binary
  acted-emotion proxies (e.g. CREMA-D anger-vs-neutral, speaker-disjoint),
  never on genuine 3-type conflict labels. See paper/04_results.md protocol.

## Survey (read first for related work)

- *A Survey of Multimodal Sarcasm Detection.* arXiv:2410.18882 (Oct 2024).
  Covers MUStARD-family results and fusion taxonomies — use it to structure
  §2 Related Work.

## Freshness verdict (Oct 2026 vs ~Aug 2026)

- Encoders NOT outdated: emotion2vec+, WavLM, DeBERTa-v3 remain standard.
- New since Aug 2026 that affects us: (1) PodSarc dataset + LLM annotation
  recipe (Apr 2026 rev) — adopt for label scarcity; (2) IEMOCAP
  concept-bottleneck 74.37% (Sep 2026) — cite as contemporary anchor.
- No new result legitimises 95+ on real conflict labels. The 95+ target
  stays scoped to binary acted-emotion proxies with speaker-disjoint splits.
