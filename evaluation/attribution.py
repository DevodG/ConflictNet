"""Captum-based attribution for ConflictNet.

Produces:
  - Token-level text saliency (integrated gradients over DeBERTa embeddings)
  - Frame-level audio saliency (integrated gradients over audio input)

Usage:
    attr = ConflictNetAttribution(model)
    text_saliency = attr.text_attribution(input_ids, attention_mask, audio)
    audio_saliency = attr.audio_attribution(audio, input_ids, attention_mask)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

# Fix for macOS OpenMP multiple initialization error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class _TextWrapper(nn.Module):
    """Wraps ConflictNet to accept token embeddings as input (for IG)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        # Replace normal embedding lookup with pre-computed embeddings
        # Access DeBERTa's embedding layer
        deberta = self.model.text_encoder.encoder
        # Get position/token type embeddings normally
        position_ids = torch.arange(input_embeds.size(1), device=input_embeds.device).unsqueeze(0)

        # Call deberta forward with inputs_embeds instead of input_ids
        text_raw = deberta(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
        ).last_hidden_state[:, 0, :]  # [CLS]

        text_embed = self.model.text_proj(text_raw)
        audio_raw = self.model.audio_encoder(audio)
        audio_embed = self.model.audio_proj(audio_raw)
        speaker_feat = torch.zeros_like(audio_embed)
        fused = self.model.fuse(audio_embed, text_embed, speaker_feat)
        logits, _, _, _ = self.model.classifier(fused)
        return logits.sum(dim=-1)  # scalar per batch item


class _AudioWrapper(nn.Module):
    """Wraps ConflictNet to accept audio as input (for IG)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        audio: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        audio_raw = self.model.audio_encoder(audio)
        audio_embed = self.model.audio_proj(audio_raw)
        text_raw = self.model.text_encoder(input_ids, attention_mask)
        text_embed = self.model.text_proj(text_raw)
        speaker_feat = torch.zeros_like(audio_embed)
        fused = self.model.fuse(audio_embed, text_embed, speaker_feat)
        logits, _, _, _ = self.model.classifier(fused)
        return logits.sum(dim=-1)


class ConflictNetAttribution:
    """Integrated Gradients attribution for ConflictNet.

    Args:
        model: Trained ConflictNet model.
        n_steps: Number of IG interpolation steps (higher = more accurate).
    """

    def __init__(self, model: nn.Module, n_steps: int = 50):
        self.model = model
        self.n_steps = n_steps
        self._text_wrapper = _TextWrapper(model)
        self._audio_wrapper = _AudioWrapper(model)

        try:
            from captum.attr import IntegratedGradients  # type: ignore
            self._ig = IntegratedGradients
            logger.info("[Attribution] Captum loaded successfully")
        except ImportError:
            logger.warning("[Attribution] captum not installed — attribution disabled")
            self._ig = None

    def text_attribution(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio: torch.Tensor,
        target: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """Compute token-level text saliency via integrated gradients.

        Returns:
            (B, seq_len) attribution scores, or None if captum unavailable.
        """
        if self._ig is None:
            return None

        # Get input embeddings from DeBERTa embedding layer
        with torch.no_grad():
            deberta = self.model.text_encoder.encoder
            input_embeds = deberta.get_input_embeddings()(input_ids)  # (B, L, H)

        ig = self._ig(self._text_wrapper)

        baseline = torch.zeros_like(input_embeds)
        attributions, _ = ig.attribute(
            inputs=input_embeds,
            baselines=baseline,
            additional_forward_args=(attention_mask, audio),
            n_steps=self.n_steps,
            return_convergence_delta=True,
        )

        # Aggregate over hidden dim → (B, seq_len)
        token_attrs = attributions.abs().sum(dim=-1)
        # Mask padding
        token_attrs = token_attrs * attention_mask.float()
        return token_attrs

    def audio_attribution(
        self,
        audio: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Compute frame-level audio saliency via integrated gradients.

        Returns:
            (B, T_audio) attribution scores, or None if captum unavailable.
        """
        if self._ig is None:
            return None

        ig = self._ig(self._audio_wrapper)
        baseline = torch.zeros_like(audio)
        attributions, _ = ig.attribute(
            inputs=audio,
            baselines=baseline,
            additional_forward_args=(input_ids, attention_mask),
            n_steps=self.n_steps,
            return_convergence_delta=True,
        )
        return attributions.abs()  # (B, T_audio)

    def top_conflicting_tokens(
        self,
        token_attrs: torch.Tensor,
        input_ids: torch.Tensor,
        tokenizer,
        top_k: int = 5,
    ) -> list:
        """Return the top-k most salient tokens for each sample in batch."""
        results = []
        for i in range(input_ids.size(0)):
            attrs = token_attrs[i]  # (seq_len,)
            top_k_idx = attrs.topk(min(top_k, attrs.size(0))).indices
            tokens = tokenizer.convert_ids_to_tokens(input_ids[i, top_k_idx].tolist())
            scores = attrs[top_k_idx].tolist()
            results.append(list(zip(tokens, scores)))
        return results
