"""ConflictNet evaluation metrics.

Primary metrics (matching the paper evaluation):
  - WAcc: Weighted accuracy (sklearn)
  - Macro-F1: Macro-averaged F1 across conflict types
  - Per-type AP: Average precision per subtype (sarcasm / suppression / deception)
  - Severity MAE: Mean absolute error for severity regression
  - AUC-ROC: Per-type and binary

All metrics returned as a flat dict for easy WandB/logging consumption.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any

import numpy as np
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


logger = logging.getLogger(__name__)

DEFAULT_TYPE_NAMES = ["sarcasm", "suppression", "deception"]


def compute_all_metrics(
    probs_type: np.ndarray,         # (N, n_types) — sigmoid probabilities
    labels_type: np.ndarray,        # (N, n_types) — multi-hot ground truth
    severity_pred: Optional[np.ndarray] = None,  # (N,)
    severity_true: Optional[np.ndarray] = None,  # (N,)
    type_threshold: float = 0.5,
    type_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute all evaluation metrics.

    Args:
        probs_type: Model sigmoid outputs (N, n_types).
        labels_type: Ground truth multi-hot labels (N, n_types).
        severity_pred: Predicted severity scores.
        severity_true: Ground truth severity.
        type_threshold: Sigmoid threshold for binary classification.
        type_names: Names of the conflict subtypes.

    Returns:
        Dict of metric_name → value. ``empty_classes`` lists subtypes with
        zero positive labels — their per-type F1/AP/AUC are degenerate and
        the macro average over them is NOT comparable to a full 3-type
        macro-F1 (a missing class caps the ceiling below 1.0).
    """
    if type_names is None:
        type_names = DEFAULT_TYPE_NAMES
    metrics: Dict[str, Any] = {}
    n_types = probs_type.shape[1]

    # Binary predictions via threshold
    preds_type = (probs_type >= type_threshold).astype(int)

    # Macro F1, precision, recall across types
    metrics["macro_f1"] = f1_score(labels_type, preds_type, average="macro", zero_division=0)
    metrics["macro_precision"] = precision_score(labels_type, preds_type, average="macro", zero_division=0)
    metrics["macro_recall"] = recall_score(labels_type, preds_type, average="macro", zero_division=0)

    # Per-type metrics
    empty_classes = []
    for i, name in enumerate(type_names[:n_types]):
        p = probs_type[:, i]
        y = labels_type[:, i]
        pred = preds_type[:, i]

        if y.sum() == 0:
            empty_classes.append(name)
        metrics[f"f1_{name}"] = f1_score(y, pred, zero_division=0)
        metrics[f"precision_{name}"] = precision_score(y, pred, zero_division=0)
        metrics[f"recall_{name}"] = recall_score(y, pred, zero_division=0)
        if y.sum() > 0:
            metrics[f"ap_{name}"] = average_precision_score(y, p)
            try:
                metrics[f"auc_{name}"] = roc_auc_score(y, p)
            except ValueError:
                metrics[f"auc_{name}"] = float("nan")
    metrics["empty_classes"] = empty_classes
    if empty_classes:
        logger.warning(
            "[Metrics] Empty classes (zero positives): %s — per-type F1/AP/AUC "
            "for these are degenerate and macro-F1 is capped at %.3f, NOT "
            "comparable to a full %d-type macro-F1",
            empty_classes, (n_types - len(empty_classes)) / n_types, n_types,
        )

    # Binary conflict flag (any type)
    conflict_pred = preds_type.any(axis=1).astype(int)
    conflict_true = labels_type.any(axis=1).astype(int)
    metrics["binary_f1"] = f1_score(conflict_true, conflict_pred, zero_division=0)
    metrics["binary_precision"] = precision_score(conflict_true, conflict_pred, zero_division=0)
    metrics["binary_recall"] = recall_score(conflict_true, conflict_pred, zero_division=0)
    metrics["binary_acc"] = accuracy_score(conflict_true, conflict_pred)
    try:
        metrics["binary_auc"] = roc_auc_score(conflict_true, probs_type.max(axis=1))
    except ValueError:
        metrics["binary_auc"] = float("nan")

    # ECE (Expected Calibration Error) — per-class, macro-averaged
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece_per_class = []
    for i in range(n_types):
        confs = probs_type[:, i]
        # Calibration measures whether a confidence value matches the
        # observed outcome, not whether it agrees with our own threshold.
        accs = labels_type[:, i]
        ece_c = 0.0
        for b in range(n_bins):
            in_bin = (confs > bin_boundaries[b]) & (confs <= bin_boundaries[b + 1])
            if in_bin.any():
                ece_c += abs(accs[in_bin].mean() - confs[in_bin].mean()) * in_bin.sum()
        ece_per_class.append(ece_c / max(len(confs), 1))
    metrics["ece_macro"] = float(np.mean(ece_per_class))

    # Weighted accuracy (sklearn accuracy_score uses equal weights by default)
    # WAcc = accuracy weighted by inverse class frequency
    from sklearn.utils.class_weight import compute_sample_weight  # type: ignore
    sample_weights = compute_sample_weight("balanced", conflict_true)
    metrics["wacc"] = float(np.average(conflict_pred == conflict_true, weights=sample_weights))

    # Severity metrics
    if severity_pred is not None and severity_true is not None:
        sev_t = np.asarray(severity_true).ravel()
        sev_p = np.asarray(severity_pred).ravel()
        mask = ~np.isnan(sev_t)
        if mask.sum() > 0:
            metrics["severity_mae"] = mean_absolute_error(sev_t[mask], sev_p[mask])

    return metrics


def print_metrics(metrics: Dict[str, Any], prefix: str = ""):
    """Pretty-print metrics table."""
    title = prefix or "ConflictNet Evaluation Results"
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)
    for k, v in sorted(metrics.items()):
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            print(f"  {k:25s}: {float(v):.4f}")
        else:
            print(f"  {k:25s}: {v}")
    print("=" * 50 + "\n")
