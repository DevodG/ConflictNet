"""Dataset that loads pre-computed encoder embeddings.

Pairs with ``scripts/precompute_embeddings.py`` which runs the frozen
encoders once and saves ``embeddings.pt``.  This dataset loads that
cache and serves the embeddings directly, bypassing WavLM + DeBERTa
during training (~60x speedup).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CachedEmbeddingDataset(Dataset):
    """Dataset over pre-computed encoder embeddings.

    Expects a ``embeddings.pt`` file produced by
    ``scripts/precompute_embeddings.py`` with keys:
      - audio_embed:   (N, embed_dim)
      - text_embed:    (N, embed_dim)
      - speaker_feat:  (N, embed_dim)
      - conflict_type_labels: (N, 3)
      - severity:      (N, 1)
      - conflict_binary: (N,)
      - utterance_ids: List[str] of length N
      - speaker_ids:   List[str] of length N (for speaker-stratified split)

    Args:
        cache_path: Path to ``embeddings.pt`` file.
        split: One of ``"train"``, ``"val"``, or ``"all"``.
        train_ratio: Fraction of samples for training split (default 0.8).
        seed: Random seed for deterministic split.
    """

    def __init__(
        self,
        cache_path: str,
        split: str = "train",
        train_ratio: float = 0.8,
        seed: int = 42,
    ):
        self.cache_path = Path(cache_path)
        if not self.cache_path.exists():
            raise FileNotFoundError(f"Cache not found: {self.cache_path}")

        data = torch.load(self.cache_path, weights_only=True)
        self.audio_embed = data["audio_embed"]
        self.text_embed = data["text_embed"]
        self.speaker_feat = data["speaker_feat"]
        self.labels = data["conflict_type_labels"]
        self.severity = data["severity"]
        self.binary = data["conflict_binary"]
        self.has_real_type_labels = data.get("has_real_type_labels", None)

        speaker_ids = data.get("speaker_ids", None)
        if speaker_ids is None:
            logger.warning("[CachedDataset] No speaker_ids in cache, falling back to random split")
            speaker_ids = [f"spk_{i}" for i in range(self.audio_embed.size(0))]

        N = self.audio_embed.size(0)

        # Speaker-stratified split: group by speaker, assign whole speakers to train/val
        speaker_groups: Dict[str, List[int]] = {}
        for idx, spk_id in enumerate(speaker_ids):
            speaker_groups.setdefault(spk_id, []).append(idx)

        rng = torch.Generator().manual_seed(seed)
        speaker_list = list(speaker_groups.keys())
        perm = torch.randperm(len(speaker_list), generator=rng).tolist()
        speaker_list = [speaker_list[i] for i in perm]

        train_items: List[int] = []
        val_items: List[int] = []
        target_train = int(N * train_ratio)
        for spk_id in speaker_list:
            indices = speaker_groups[spk_id]
            if not train_items or len(train_items) + len(indices) <= target_train:
                train_items.extend(indices)
            else:
                val_items.extend(indices)

        if split == "train":
            self.indices = train_items
        elif split == "val":
            self.indices = val_items
        else:
            self.indices = list(range(N))

        logger.info(f"[CachedDataset] {split}: {len(self.indices)} samples from {self.cache_path.name} (speaker-stratified)")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        i = self.indices[idx]
        item = {
            "audio_embed": self.audio_embed[i],
            "text_embed": self.text_embed[i],
            "speaker_feat": self.speaker_feat[i],
            "conflict_type_labels": self.labels[i],
            "severity": self.severity[i],
            "conflict_binary": self.binary[i],
        }
        # Older caches predate label provenance. Treat them as proxy-only
        # rather than silently training subtype heads on fabricated labels.
        item["has_real_type_labels"] = (
            self.has_real_type_labels[i]
            if self.has_real_type_labels is not None
            else torch.tensor(False)
        )
        return item


def cached_collate_fn(batch):
    """Simple collation for cached embeddings (all fixed-size, no padding needed)."""
    result = {}
    keys = batch[0].keys()
    for k in keys:
        result[k] = torch.stack([b[k] for b in batch])
    return result
