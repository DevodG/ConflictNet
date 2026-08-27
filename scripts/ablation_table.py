#!/usr/bin/env python3
"""Generate LaTeX ablation table from multiple training runs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Generate ablation LaTeX table")
    p.add_argument("--runs", type=str, required=True,
                   help="Comma-separated run names (e.g., run1_full,run2_no_sep,run3_no_temporal,run5_baseline)")
    p.add_argument("--checkpoints_dir", type=str, default="checkpoints",
                   help="Base directory containing run checkpoints")
    p.add_argument("--output", type=str, default="ablation_table.tex",
                   help="Output LaTeX file")
    return p.parse_args()


def load_run_metrics(run_name, checkpoints_dir):
    """Load metrics from a run's best_model_meta.json."""
    meta_path = Path(checkpoints_dir) / run_name / "best_model_meta.json"
    if not meta_path.exists():
        logger.warning(f"Meta file not found: {meta_path}")
        return None
    
    with open(meta_path) as f:
        meta = json.load(f)
    
    return {
        "name": run_name,
        "best_val_f1": meta.get("best_val_f1", 0.0),
        "best_val_type_f1_micro": meta.get("best_val_type_f1_micro", 0.0),
        "best_val_type_f1_macro": meta.get("best_val_type_f1_macro", 0.0),
        "best_val_severity_mse": meta.get("best_val_severity_mse", 0.0),
        "epoch": meta.get("epoch", 0),
        "trainable_params": meta.get("trainable_params", 0),
    }


def generate_latex_table(runs_data, output_path):
    """Generate LaTeX ablation table."""
    
    # Run name mapping for pretty labels
    name_map = {
        "run1_full": "Full ConflictNet (Ours)",
        "run2_no_sep": r"$\backslash$ Separation Wall",
        "run3_no_temporal": r"$\backslash$ Temporal Context",
        "run5_baseline": "ConflictNet-mini",
    }
    
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"  \centering")
    latex.append(r"  \caption{Ablation study on conflict detection components.}")
    latex.append(r"  \label{tab:ablation}")
    latex.append(r"  \begin{tabular}{lcccc}")
    latex.append(r"    \toprule")
    latex.append(r"    \textbf{Model} & \textbf{Binary F1} & \textbf{Type F1 ($\mu$)} & \textbf{Type F1 (M)} & \textbf{Severity MSE} \\")
    latex.append(r"    \midrule")
    
    for run in runs_data:
        if run is None:
            continue
        pretty_name = name_map.get(run["name"], run["name"])
        latex.append(
            f"    {pretty_name} & "
            f"{run['best_val_f1']:.4f} & "
            f"{run['best_val_type_f1_micro']:.4f} & "
            f"{run['best_val_type_f1_macro']:.4f} & "
            f"{run['best_val_severity_mse']:.4f} \\\\"
        )
    
    latex.append(r"    \bottomrule")
    latex.append(r"  \end{tabular}")
    latex.append(r"\end{table}")
    
    with open(output_path, "w") as f:
        f.write("\n".join(latex))
    
    logger.info(f"LaTeX table saved to {output_path}")


def main():
    args = parse_args()
    
    run_names = [r.strip() for r in args.runs.split(",")]
    runs_data = []
    
    for run_name in run_names:
        metrics = load_run_metrics(run_name, args.checkpoints_dir)
        if metrics:
            runs_data.append(metrics)
            logger.info(f"Loaded {run_name}: F1={metrics['best_val_f1']:.4f}")
    
    if not runs_data:
        logger.error("No valid runs found!")
        return
    
    generate_latex_table(runs_data, args.output)


if __name__ == "__main__":
    main()