"""Fairness audit using FairLearn.

Evaluates demographic parity and equalized odds across:
  - Gender (M / F)
  - Any other provided sensitive attribute

Usage:
    from evaluation.fairness import fairness_audit
    report = fairness_audit(preds, labels, sensitive_features=genders)
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def fairness_audit(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    sensitive_features: List[Any],
    metric_fn=None,
) -> Dict[str, Any]:
    """Run FairLearn fairness audit.

    Args:
        y_pred: Binary predictions (N,).
        y_true: Ground truth labels (N,).
        sensitive_features: Group membership per sample (e.g. gender list).
        metric_fn: Metric function (default: f1_score).

    Returns:
        Dict with:
          - by_group: per-group metric values
          - disparity: max - min across groups
          - demographic_parity_difference
          - equalized_odds_difference
    """
    try:
        from fairlearn.metrics import (  # type: ignore
            MetricFrame,
            demographic_parity_difference,
            equalized_odds_difference,
        )
        from sklearn.metrics import f1_score  # type: ignore

        if metric_fn is None:
            def metric_fn(y, p):
                return f1_score(y, p, zero_division=0)

        mf = MetricFrame(
            metrics={"f1": metric_fn},
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        )

        dpd = demographic_parity_difference(
            y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features
        )
        eod = equalized_odds_difference(
            y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features
        )

        def _as_float(value: Any, key: str = "f1") -> float:
            # fairlearn returns Series or DataFrame depending on version and
            # metric count — normalise to a scalar either way.
            if hasattr(value, "get") and not hasattr(value, "iloc"):
                value = value.get(key, value)
            if hasattr(value, "iloc"):
                value = value.iloc[0] if len(value) > 0 else float("nan")
            return float(value)

        by_group_raw = mf.by_group.to_dict()
        by_group = by_group_raw.get("f1", by_group_raw)
        if not isinstance(by_group, dict):
            by_group = {"overall": float(by_group)}
        overall = _as_float(mf.overall)
        disparity = _as_float(mf.difference())

        return {
            "overall_f1": overall,
            "by_group": by_group,
            "disparity": disparity,
            "demographic_parity_difference": float(dpd),
            "equalized_odds_difference": float(eod),
        }

    except ImportError:
        print("[WARN] fairlearn not installed. Run: pip install fairlearn")
        from sklearn.metrics import f1_score  # type: ignore

        groups = sorted(set(sensitive_features))
        by_group = {}
        for g in groups:
            mask = np.array(sensitive_features) == g
            if mask.sum() > 0:
                by_group[g] = f1_score(y_true[mask], y_pred[mask], zero_division=0)

        values = list(by_group.values())
        return {
            "overall_f1": f1_score(y_true, y_pred, zero_division=0),
            "by_group": by_group,
            "disparity": max(values) - min(values) if len(values) >= 2 else 0.0,
            "demographic_parity_difference": None,
            "equalized_odds_difference": None,
        }
