# Conclusion

> **Status:** Placeholder — populate after results are available.

---

## Summary

We presented **ConflictNet**, a speaker-normalised cross-modal model for emotional conflict detection in dialogue. Key contributions:

1. **Speaker normalisation** via ECAPA-TDNN + prosody z-score disentangles speaker identity from conflict expression, reducing the speaker confound identified in prior work.
2. **Context-gated contrastive learning** learns per-sample temperature modulation from dialogue context, improving alignment of ambiguous utterances.
3. **Temporal context** from preceding dialogue turns improves detection accuracy by modeling conversational dynamics (ongoing).
4. **Multi-label + severity output** provides fine-grained conflict characterisation beyond binary detection.

---

## Limitations

- Severity labels are not available in public corpora; the severity head trains only where annotated (currently nowhere — predictions are untrained) and severity MAE is reported only on annotated splits.
- No public corpus annotates deception: the deception head has zero positive labels and its metrics are degenerate until CASE annotation completes.
- Emotion-to-conflict mappings (anger/frustration → suppression-proxy) are a research definition — real conflict types are more nuanced, and proxy results are labeled as such throughout.
- Word-level divergence features require Montreal Forced Aligner, adding preprocessing overhead; the model runs without them.
- Acted corpora (IEMOCAP, CREMA-D) differ distributionally from spontaneous conflict; real-world evaluation on in-the-wild data is needed.

---

## Ethics, Misuse, and Licensing

Conflict — especially deception — detection is dual-use. Forensic or workplace deployment without consent risks serious harm (false accusations, surveillance creep), and acted-corpus performance does not transfer to high-stakes spontaneous speech. We therefore: (1) scope all deception claims to future annotated evaluation, never to current proxy numbers; (2) require the FairLearn audit (`evaluation/fairness.py`: demographic parity + equalized odds) on real labels before any deployment claim; (3) publish per-group error breakdowns alongside headline F1; (4) document annotator protocols and consent for any human evaluation (`evaluation/human_eval.py`).

Dataset licenses: IEMOCAP requires a USC restricted-use license (request access; do not redistribute); MUStARD/MUStARD++ audio derives from copyrighted TV shows (research use); MELD/CMU-MOSEI/CREMA-D follow their respective releases. Voice data is biometric — store and share it accordingly.

---

## Future Work

- **CASE annotation:** Fine-grained subtype + severity labels via a PodSarc-style LLM + human-verification pipeline (Li et al., 2026).
- **Cross-lingual conflict detection:** Adapt audio encoder to multilingual speech representations (e.g., Whisper, MMS).
- **End-to-end streaming:** Integrate temporal context into a streaming pipeline for real-time dialogue monitoring (see `serve/api.py` `/ws`).
- **Causal intervention:** Use the speaker normalisation module for counterfactual analysis (e.g., "would this utterance be conflictual if spoken by a different speaker?").
