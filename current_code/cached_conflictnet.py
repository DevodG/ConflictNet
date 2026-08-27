"""Cached ConflictNet — trains on pre-computed encoder embeddings.

Reuses all trainable components from the full ``ConflictNet`` after
the frozen encoder ``encode()`` step: fusion gate, cross-modal attention,
temporal context, classifier, contrastive loss, swap objective, and
multi-task loss balancing.

This allows training at ~60× speedup since WavLM (316M) and DeBERTa
(304M) are bypassed. Only ~3.6M trainable params remain.

Usage::

    from models.cached_conflictnet import CachedConflictNet

    model = CachedConflictNet(embed_dim=256, n_conflict_types=3)
    out = model(
        audio_embed=audio_embed,   # (B, 256) pre-computed
        text_embed=text_embed,     # (B, 256) pre-computed
        speaker_feat=speaker_feat, # (B, 256) pre-computed
        ...
    )
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .alignment import ProjectionHead, ContextGatedContrastiveLoss, CrossModalAttention
from .alignment.word_divergence import WordLevelDivergence
from .classifier import ConflictClassifier
from .conflictnet import ConflictNetOutput, MultiTaskLoss, SwapPretrainingObjective, focal_loss_with_logits
from .temporal import TransformerTemporalContext

logger = logging.getLogger(__name__)


class CachedConflictNet(nn.Module):
    """ConflictNet variant that trains on pre-computed encoder embeddings.

    Identical to ``ConflictNet`` from the fusion-gate onward, but accepts
    ``audio_embed``, ``text_embed``, and ``speaker_feat`` directly instead
    of raw audio/text input.  No encoders are loaded — only trainable
    downstream modules.

    Args:
        embed_dim: Shared embedding dimensionality (256).
        n_conflict_types: Number of conflict subtypes (3).
        temporal_n_layers: Layers in the temporal Transformer.
        temporal_n_heads: Attention heads in the temporal Transformer.
        temporal_max_turns: Max dialogue turns in context window.
        use_speaker_norm: If True, expects ``speaker_feat`` to be passed.
        use_temporal: Enable temporal context Transformer.
        use_cross_attn_injection: Enable audio↔text cross-attention.
        use_speaker_adaptive_threshold: Enable per-sample threshold offset.
        use_word_divergence: Enable word-level divergence features.
        use_swap_pretraining: Enable self-supervised swap objective.
        focal_loss_gamma: Focal loss gamma (0 = standard BCE).
        label_smoothing: Label smoothing for multi-label BCE.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        n_conflict_types: int = 3,
        temporal_n_layers: int = 2,
        temporal_n_heads: int = 4,
        temporal_max_turns: int = 16,
        use_speaker_norm: bool = True,
        use_temporal: bool = True,
        use_cross_attn_injection: bool = True,
        use_speaker_adaptive_threshold: bool = True,
        use_word_divergence: bool = False,
        use_swap_pretraining: bool = True,
        focal_loss_gamma: float = 0.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_speaker_norm = use_speaker_norm
        self.use_temporal = use_temporal
        self.use_cross_attn_injection = use_cross_attn_injection
        self.use_speaker_adaptive_threshold = use_speaker_adaptive_threshold
        self.use_word_divergence = use_word_divergence
        self.use_swap_pretraining = use_swap_pretraining
        self.focal_loss_gamma = focal_loss_gamma
        self.label_smoothing = label_smoothing

        # Gating network: fuse (audio_proj + text_proj + speaker_feat) → fused_embed
        fuse_in = embed_dim * 3 if use_speaker_norm else embed_dim * 2
        self.fusion_gate = nn.Sequential(
            nn.Linear(fuse_in, embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Cross-modal attention
        self.cross_modal_attn = CrossModalAttention(
            embed_dim=embed_dim, n_heads=temporal_n_heads,
        ) if use_cross_attn_injection else None

        # Temporal context
        self.temporal = TransformerTemporalContext(
            embed_dim=embed_dim, n_layers=temporal_n_layers,
            n_heads=temporal_n_heads, max_turns=temporal_max_turns,
        ) if use_temporal else None

        # Word-level divergence (optional)
        self.word_divergence = WordLevelDivergence(embed_dim=embed_dim) if use_word_divergence else None
        word_div_dim = WordLevelDivergence.DIVERGENCE_FEAT_DIM if use_word_divergence else 0

        # Classifier
        self.classifier = ConflictClassifier(
            embed_dim=embed_dim, n_types=n_conflict_types,
            word_div_dim=word_div_dim,
            speaker_adaptive_threshold=use_speaker_adaptive_threshold,
        )

        # Contrastive loss
        self.contrastive_loss_fn = ContextGatedContrastiveLoss(embed_dim=embed_dim)

        # Self-supervised swap objective
        self.swap_objective = SwapPretrainingObjective(embed_dim=embed_dim) if use_swap_pretraining else None

        # Multi-task loss balancing
        n_tasks = 4 if use_swap_pretraining else 3
        self.multi_task_loss = MultiTaskLoss(n_tasks=n_tasks)

        logger.info(
            f"[CachedConflictNet] embed_dim={embed_dim} | "
            f"temporal={use_temporal} | cross_attn={use_cross_attn_injection} | "
            f"speaker_adaptive_threshold={use_speaker_adaptive_threshold}"
        )

    def fuse(
        self,
        audio_embed: torch.Tensor,
        text_embed: torch.Tensor,
        speaker_feat: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_speaker_norm:
            combined = torch.cat([audio_embed, text_embed, speaker_feat], dim=-1)
        else:
            combined = torch.cat([audio_embed, text_embed], dim=-1)
        return self.fusion_gate(combined)

    def forward(
        self,
        audio_embed: torch.Tensor,
        text_embed: torch.Tensor,
        speaker_feat: torch.Tensor,
        context_embeds: Optional[torch.Tensor] = None,
        context_padding: Optional[torch.Tensor] = None,
        speaker_roles: Optional[torch.Tensor] = None,
        word_div_feats: Optional[torch.Tensor] = None,
        conflict_type_labels: Optional[torch.Tensor] = None,
        severity_labels: Optional[torch.Tensor] = None,
        conflict_binary_labels: Optional[torch.Tensor] = None,
        pretraining: bool = False,
    ) -> ConflictNetOutput:
        B = audio_embed.size(0)

        # 1. Cross-modal attention
        if self.cross_modal_attn is not None:
            audio_embed, text_embed = self.cross_modal_attn(
                audio_embed, text_embed,
                context_seq=context_embeds,
                context_padding=context_padding,
            )

        # 2. Fuse
        fused_embed = self.fuse(audio_embed, text_embed, speaker_feat)

        # 3. Temporal context
        if self.temporal is not None:
            current_turn = fused_embed.unsqueeze(1)
            if context_embeds is not None:
                turn_seq = torch.cat([context_embeds.to(device=fused_embed.device), current_turn], dim=1)
                if context_padding is not None:
                    curr_pad = torch.zeros(B, 1, dtype=torch.bool, device=fused_embed.device)
                    pad_mask = torch.cat([context_padding.to(device=fused_embed.device), curr_pad], dim=1)
                else:
                    pad_mask = None
            else:
                turn_seq = current_turn
                pad_mask = None
            per_turn_ctx, context_pooled = self.temporal(turn_seq, padding_mask=pad_mask, speaker_roles=speaker_roles)
            current_ctx = per_turn_ctx[:, -1, :]
        else:
            per_turn_ctx = fused_embed.unsqueeze(1)
            context_pooled = fused_embed
            current_ctx = fused_embed

        # 4. Classify
        logits_type, probs_type, severity, conflict_flag = self.classifier(
            fused_embed=current_ctx,
            word_div=word_div_feats,
            speaker_feat=speaker_feat,
        )

        # 5. Losses
        loss = None
        loss_breakdown = None
        if conflict_type_labels is not None or pretraining:
            losses = []

            cl = self.contrastive_loss_fn(
                audio_embed, text_embed,
                context_pooled=context_pooled,
                conflict_labels=conflict_binary_labels,
            )
            losses.append(cl)

            if conflict_type_labels is not None:
                targets = conflict_type_labels.float()
                if self.label_smoothing > 0.0:
                    targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
                if self.focal_loss_gamma > 0.0:
                    type_loss = focal_loss_with_logits(logits_type, targets, gamma=self.focal_loss_gamma)
                else:
                    type_loss = nn.functional.binary_cross_entropy_with_logits(logits_type, targets)
                losses.append(type_loss)
            else:
                losses.append(torch.tensor(0.0, device=audio_embed.device))

            if severity is not None and severity_labels is not None:
                sev_target = severity_labels.float().view(-1)
                sev_pred = severity.view(-1)
                sev_loss = nn.functional.mse_loss(sev_pred, sev_target)
                losses.append(sev_loss)
            else:
                losses.append(torch.tensor(0.0, device=audio_embed.device))

            if self.swap_objective is not None:
                swap_loss = self.swap_objective(audio_embed, text_embed)
                losses.append(swap_loss)

            loss, sigma_weights = self.multi_task_loss(losses)
            loss_breakdown = {
                "contrastive": losses[0].detach().item(),
                "type_loss": losses[1].detach().item(),
                "severity_mse": losses[2].detach().item(),
                **sigma_weights,
            }
            if self.swap_objective is not None:
                loss_breakdown["swap"] = losses[3].detach().item()

        return ConflictNetOutput(
            logits_type=logits_type,
            probs_type=probs_type,
            severity=severity,
            conflict_flag=conflict_flag,
            audio_embed=audio_embed,
            text_embed=text_embed,
            speaker_feat=speaker_feat,
            fused_embed=fused_embed,
            context_pooled=context_pooled,
            per_turn_context=per_turn_ctx,
            word_div_feats=word_div_feats,
            loss=loss,
            loss_breakdown=loss_breakdown,
        )

    def count_parameters(self) -> Dict[str, Dict[str, int]]:
        result = {}
        for name, module in self.named_children():
            n_total = sum(p.numel() for p in module.parameters())
            n_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
            result[name] = {"total": n_total, "trainable": n_train}
        return result
