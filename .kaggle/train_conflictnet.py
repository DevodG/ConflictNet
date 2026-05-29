#!/usr/bin/env python3
"""ConflictNet Training Script for Kaggle GPU.

Usage: python train_conflictnet.py
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

KAGGLE_INPUT = Path("/kaggle/input")
WORK_DIR = Path("/kaggle/working")
CREMAD_DIR = WORK_DIR / "cremad"
MUSTARD_DIR = WORK_DIR / "mustard"
OUTPUT_DIR = WORK_DIR / "output"


def install_deps():
    logger.info("Installing dependencies...")
    deps = ["sentencepiece", "tiktoken", "kagglehub"]
    subprocess.run(
        [sys.executable, "-m", "pip", "install"] + deps,
        check=False, capture_output=False,
    )
    logger.info("Dependencies installed.")


def find_code_dir():
    for p in KAGGLE_INPUT.rglob("scripts/train.py"):
        return p.parent.parent
    for p in KAGGLE_INPUT.rglob("prosody_stats.json"):
        return p.parent
    logger.error("Cannot find code directory in Kaggle inputs!")
    sys.exit(1)


def setup_models():
    """Extract pretrained model tarballs and set up env vars."""
    models_dir = WORK_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    hf_cache = WORK_DIR / "hf_cache"
    hf_cache.mkdir(parents=True, exist_ok=True)

    import tarfile
    for model_key, tar_name, env_var in [
        ("wavlm", "wavlm-base-plus.tar.gz", "CONFLICTNET_WAVLM_PATH"),
        ("deberta", "deberta-v3-large.tar.gz", "CONFLICTNET_DEBERTA_PATH"),
    ]:
        # Find the tarball in Kaggle inputs
        tarball = None
        for p in KAGGLE_INPUT.rglob(tar_name):
            tarball = p
            break
        if tarball is None:
            logger.error(f"Model tarball {tar_name} not found in Kaggle inputs!")
            sys.exit(1)

        extract_dir = models_dir / model_key
        extract_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Extracting {tar_name} to {extract_dir}...")
        with tarfile.open(str(tarball), "r:gz") as tar:
            tar.extractall(path=str(extract_dir))

        # The tarball contains a subdirectory; find it
        subdirs = [d for d in extract_dir.iterdir() if d.is_dir()]
        if subdirs:
            model_path = str(subdirs[0])
        else:
            model_path = str(extract_dir)
        os.environ[env_var] = model_path
        logger.info(f"Set {env_var}={model_path}")

    # Also set HF cache
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache)
    logger.info(f"HF_HOME set to {hf_cache}")

    # Tokenizer comes from the DeBERTa model dir
    tok_dir = os.environ.get("CONFLICTNET_DEBERTA_PATH")
    if tok_dir:
        import transformers
        tokenizer = transformers.AutoTokenizer.from_pretrained(tok_dir)
        logger.info(f"Tokenizer: {type(tokenizer).__name__} (vocab_size={tokenizer.vocab_size})")
    return tok_dir


def setup_data():
    logger.info("Setting up data directories...")
    CREMAD_DIR.mkdir(parents=True, exist_ok=True)
    MUSTARD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    code_dir = find_code_dir()
    logger.info(f"Code directory: {code_dir}")

    # CREMA-D from Kaggle input
    cremad_input = list(KAGGLE_INPUT.rglob("ejlok1/cremad/AudioWAV"))
    if cremad_input:
        logger.info(f"Found CREMA-D at {cremad_input[0]}")
        (CREMAD_DIR / "AudioWAV").symlink_to(cremad_input[0])
    else:
        for p in KAGGLE_INPUT.rglob("AudioWAV"):
            if p.is_dir():
                (CREMAD_DIR / "AudioWAV").symlink_to(p)
                logger.info(f"Found CREMA-D at {p}")
                break
        else:
            logger.error("CREMA-D AudioWAV not found in Kaggle inputs!")
            sys.exit(1)

    # MUStARD++ data from code dataset
    mustard_json_src = code_dir / "data" / "mustard" / "mustard++_raw_data.json"
    if mustard_json_src.exists():
        (MUSTARD_DIR / "mustard++_raw_data.json").symlink_to(mustard_json_src.resolve())

    # MUStARD++ audio unavailable — CREMA-D only
    logger.info("MUStARD++ audio unavailable — training on CREMA-D only")

    # Copy prosody stats
    for fname in ["prosody_stats.json", "prosody_stats.zscores.json"]:
        src = code_dir / fname
        if src.exists():
            (WORK_DIR / fname).symlink_to(src.resolve())

    tok_dir = setup_models()
    return code_dir, tok_dir


def run_training(code_dir, tok_dir):
    logger.info("Starting training...")
    train_script = code_dir / "scripts" / "train.py"
    if not train_script.exists():
        logger.error(f"Training script not found at {train_script}")
        sys.exit(1)

    args = [
        sys.executable, str(train_script),
        "--cremad_root", str(CREMAD_DIR),
        "--epochs", "30",
        "--batch_size", "16",
        "--lr", "5e-5",
        "--audio_encoder", "wavlm",
        "--no_word_divergence",
        "--tokenizer_path", str(tok_dir) if tok_dir else "",
        "--prosody_stats", str(WORK_DIR / "prosody_stats.json"),
        "--output_dir", str(OUTPUT_DIR),
    ]

    mustard_json = MUSTARD_DIR / "mustard++_raw_data.json"
    if mustard_json.exists():
        mustard_audio = MUSTARD_DIR / "utterances_final"
        if mustard_audio.exists() and list(mustard_audio.glob("*")):
            args.extend(["--mustard_root", str(MUSTARD_DIR)])
            logger.info("MUStARD++ root found, including in training")

    logger.info(f"Command: {' '.join(args)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{code_dir}:{env.get('PYTHONPATH', '')}"
    result = subprocess.run(args, cwd=str(code_dir), env=env)
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
    code_dir, tok_dir = setup_data()
    run_training(code_dir, tok_dir)
    save_outputs()
    logger.info("All done!")


if __name__ == "__main__":
    main()
