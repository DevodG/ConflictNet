#!/usr/bin/env python3
"""ConflictNet Training Script for Kaggle GPU.

Usage: python train_conflictnet.py
"""

import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

KAGGLE_INPUT = Path("/kaggle/input")
WORK_DIR = Path("/kaggle/working")
CREMAD_DIR = WORK_DIR / "cremad"
MUSTARD_DIR = WORK_DIR / "mustard"
CODE_DIR = KAGGLE_INPUT / "conflictnet-code"
OUTPUT_DIR = WORK_DIR / "output"


def install_deps():
    logger.info("Installing dependencies...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "torch", "torchaudio", "torchvision",
        "transformers", "librosa", "praat-parselmouth",
        "sentencepiece", "tiktoken", "kagglehub",
        "tqdm", "pandas", "scikit-learn",
        "accelerate",
    ], check=True, capture_output=False)
    logger.info("Dependencies installed.")


def setup_data():
    logger.info("Setting up data directories...")
    CREMAD_DIR.mkdir(parents=True, exist_ok=True)
    MUSTARD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # CREMA-D from Kaggle input
    cremad_input = list(KAGGLE_INPUT.glob("**/ejlok1/cremad/AudioWAV"))
    if cremad_input:
        logger.info(f"Found CREMA-D at {cremad_input[0]}")
        (CREMAD_DIR / "AudioWAV").symlink_to(cremad_input[0])
    else:
        # Try other paths
        for p in KAGGLE_INPUT.rglob("AudioWAV"):
            if p.is_dir():
                (CREMAD_DIR / "AudioWAV").symlink_to(p)
                logger.info(f"Found CREMA-D at {p}")
                break
        else:
            logger.error("CREMA-D AudioWAV not found in Kaggle inputs!")
            sys.exit(1)

    # MUStARD++ from our code dataset
    mustard_json_src = CODE_DIR / "data" / "mustard" / "mustard++_raw_data.json"
    if mustard_json_src.exists():
        (MUSTARD_DIR / "mustard++_raw_data.json").symlink_to(mustard_json_src.resolve())

    # Try downloading MUStARD++ audio from Google Drive
    try:
        import gdown
        gdown.download_folder(
            "https://drive.google.com/drive/folders/1kUdT2yU7ERJ5KdauObTj5oQsBlSrvTlW",
            output=str(MUSTARD_DIR / "utterances_final"),
        )
        logger.info("MUStARD++ audio downloaded")
    except Exception as e:
        logger.warning(f"MUStARD++ audio download failed: {e}")
        logger.warning("Will train on CREMA-D only")

    # Copy prosody stats
    for fname in ["prosody_stats.json", "prosody_stats.zscores.json"]:
        src = CODE_DIR / fname
        if src.exists():
            (WORK_DIR / fname).symlink_to(src.resolve())


def run_training():
    logger.info("Starting training...")
    args = [
        sys.executable, "-m", "scripts.train",
        "--cremad_root", str(CREMAD_DIR),
        "--epochs", "30",
        "--batch_size", "16",
        "--lr", "5e-5",
        "--prosody_stats", str(WORK_DIR / "prosody_stats.json"),
        "--output_dir", str(OUTPUT_DIR),
        "--log_interval", "10",
        "--eval_interval", "500",
        "--save_interval", "1000",
        "--num_workers", "2",
    ]

    mustard_json = MUSTARD_DIR / "mustard++_raw_data.json"
    if mustard_json.exists():
        mustard_audio = MUSTARD_DIR / "utterances_final"
        if mustard_audio.exists() and list(mustard_audio.glob("*")):
            args.extend(["--mustard_root", str(MUSTARD_DIR)])
            logger.info("MUStARD++ root found, including in training")

    logger.info(f"Command: {' '.join(args)}")
    result = subprocess.run(args, cwd=str(CODE_DIR))
    if result.returncode != 0:
        logger.error(f"Training failed with code {result.returncode}")
        sys.exit(result.returncode)
    logger.info("Training completed!")


def save_outputs():
    logger.info("Saving outputs...")
    ckpts = list(OUTPUT_DIR.glob("*.pt")) + list(OUTPUT_DIR.glob("*.pth"))
    for ckpt in ckpts:
        final_path = WORK_DIR / ckpt.name
        ckpt.rename(final_path)
        logger.info(f"Checkpoint saved: {final_path}")

    logs = list(OUTPUT_DIR.glob("*.json")) + list(OUTPUT_DIR.glob("*.log"))
    for log in logs:
        final_path = WORK_DIR / log.name
        log.rename(final_path)
        logger.info(f"Log saved: {final_path}")


def main():
    install_deps()
    setup_data()
    run_training()
    save_outputs()
    logger.info("All done!")


if __name__ == "__main__":
    main()
