#!/usr/bin/env python3
"""Statistical rigor for ConflictNet evaluations.

Computes bootstrap 95% confidence intervals for macro-F1 / binary-F1 and
paired significance tests (paired bootstrap + McNemar) for model-vs-baseline
comparisons. Every number reported in the paper must come with a CI from
this script — single-run point estimates are not publishable.

Inputs are ``.npz`` files with ``probs`` (N, C) sigmoid outputs and
``labels`` (N, C) multi-hot ground truth, as written by
``scripts/evaluate.py``-style runs (see ``--dump_npz``).

Usage:
    python scripts/stats.py --npz results/model_a.npz --metric macro_f1
    python scripts/stats.py --npz results/model_a.npz --baseline results/base.npz
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return np.asarray(data["probs"], dtype=float), np.asarray(data["labels"], dtype=float)


def score_at_threshold(probs: np.ndarray, labels: np.ndarray, metric: str, threshold: float = 0.5) -> float:
    preds = (probs >= threshold).astype(int)
    if metric == "macro_f1":
        return float(f1_score(labels, preds, average="macro", zero_division=0))
    if metric == "binary_f1":
        return float(f1_score(labels.any(axis=1), preds.any(axis=1), zero_division=0))
    raise ValueError(f"Unknown metric: {metric} (choose macro_f1 or binary_f1)")


def bootstrap_ci(
    probs: np.ndarray,
    labels: np.ndarray,
    metric: str = "macro_f1",
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    threshold: float = 0.5,
    seed: int = 42,
) -> dict:
    """Percentile bootstrap CI for a metric. Returns point/ci_lo/ci_hi/se."""
    rng = np.random.default_rng(seed)
    n = probs.shape[0]
    point = score_at_threshold(probs, labels, metric, threshold)
    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        boot[i] = score_at_threshold(probs[idx], labels[idx], metric, threshold)
    alpha = 1 - ci
    return {
        "metric": metric,
        "point": point,
        "ci_lo": float(np.percentile(boot, 100 * alpha / 2)),
        "ci_hi": float(np.percentile(boot, 100 * (1 - alpha / 2))),
        "se": float(boot.std(ddof=1)) if n_bootstrap > 1 else 0.0,
        "n_bootstrap": n_bootstrap,
        "n_samples": n,
        "seed": seed,
    }


def paired_bootstrap_pvalue(
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    labels: np.ndarray,
    metric: str = "macro_f1",
    n_bootstrap: int = 10000,
    threshold: float = 0.5,
    seed: int = 42,
) -> dict:
    """Two-sided paired bootstrap p-value for (A − B) metric difference."""
    rng = np.random.default_rng(seed)
    n = probs_a.shape[0]
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        diffs[i] = score_at_threshold(probs_a[idx], labels[idx], metric, threshold) - score_at_threshold(
            probs_b[idx], labels[idx], metric, threshold
        )
    observed = float(np.mean(diffs))
    # Two-sided: mass at least as extreme as |observed| in either tail.
    p = float(np.mean(np.abs(diffs - diffs.mean()) >= abs(observed)) + 1e-12)
    return {
        "test": f"paired_bootstrap_{metric}",
        "delta": observed,
        "p_value": min(p, 1.0),
        "n_bootstrap": n_bootstrap,
        "significant_05": bool(p < 0.05),
    }


def mcnemar_test(y_true_bin: np.ndarray, pred_a_bin: np.ndarray, pred_b_bin: np.ndarray) -> dict:
    """McNemar's test on binary correctness of A vs B (same samples)."""
    correct_a = (pred_a_bin == y_true_bin)
    correct_b = (pred_b_bin == y_true_bin)
    b = int((correct_a & ~correct_b).sum())  # A right, B wrong
    c = int((~correct_a & correct_b).sum())  # A wrong, B right
    n_discord = b + c
    if n_discord == 0:
        return {"test": "mcnemar", "statistic": 0.0, "p_value": 1.0, "b": b, "c": c, "significant_05": False}
    if n_discord < 25:
        res = scipy_stats.binomtest(min(b, c), n_discord, 0.5, alternative="two-sided")
        stat, p = 0.0, float(res.pvalue)
    else:
        stat = float((abs(b - c) - 1) ** 2 / n_discord)  # Edwards continuity correction
        p = float(scipy_stats.chi2.sf(stat, 1))
    return {"test": "mcnemar", "statistic": stat, "p_value": p, "b": b, "c": c, "significant_05": bool(p < 0.05)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Bootstrap CIs + significance tests for ConflictNet")
    p.add_argument("--npz", type=str, required=True, help=".npz with probs (N,C) + labels (N,C)")
    p.add_argument("--baseline", type=str, default=None, help="Baseline .npz (same samples/order)")
    p.add_argument("--metric", type=str, default="macro_f1", choices=["macro_f1", "binary_f1"])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--n_bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=None, help="Write JSON report here")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    probs, labels = load_npz(args.npz)
    report: dict = {"model": bootstrap_ci(probs, labels, args.metric, args.n_bootstrap, seed=args.seed,
                                           threshold=args.threshold)}
    logger.info(f"[Stats] {args.metric} = {report['model']['point']:.4f} "
                f"[{report['model']['ci_lo']:.4f}, {report['model']['ci_hi']:.4f}] (n={probs.shape[0]})")
    if args.baseline:
        probs_b, labels_b = load_npz(args.baseline)
        if labels_b.shape != labels.shape or probs_b.shape != probs.shape:
            raise ValueError("Baseline .npz must have the same shapes/samples as --npz")
        report["paired_bootstrap"] = paired_bootstrap_pvalue(
            probs, probs_b, labels, args.metric, args.n_bootstrap, args.threshold, args.seed)
        y_true = labels.any(axis=1).astype(int)
        report["mcnemar"] = mcnemar_test(y_true, (probs.max(axis=1) >= args.threshold).astype(int),
                                         (probs_b.max(axis=1) >= args.threshold).astype(int))
        logger.info(f"[Stats] paired Δ={report['paired_bootstrap']['delta']:+.4f} "
                    f"p={report['paired_bootstrap']['p_value']:.4g} | "
                    f"McNemar p={report['mcnemar']['p_value']:.4g}")
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"[Stats] Report → {out}")
    return report


if __name__ == "__main__":
    main()
