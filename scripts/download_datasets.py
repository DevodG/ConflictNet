#!/usr/bin/env python3
"""Download missing ConflictNet datasets.

Handles:
  - MELD: from Kaggle (kagglehub) or direct GitHub download
  - MUStARD++ audio: from GitHub releases
  - CREMA-D: verify existing (already present)
  - IEMOCAP: skip (requires USC license)
"""

import os
import sys
import shutil
import subprocess
import zipfile
import tarfile
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def check_cremad():
    """Verify CREMA-D is present."""
    wav_dir = DATA_ROOT / "cremad" / "AudioWAV"
    if wav_dir.exists():
        count = len(list(wav_dir.glob("*.wav")))
        logger.info(f"✅ CREMA-D: {count} WAV files found in {wav_dir}")
        return count > 0
    logger.warning(f"❌ CREMA-D: AudioWAV directory not found at {wav_dir}")
    return False


def download_cremad_kaggle():
    """Download CREMA-D from Kaggle if missing."""
    wav_dir = DATA_ROOT / "cremad" / "AudioWAV"
    if wav_dir.exists() and len(list(wav_dir.glob("*.wav"))) > 7000:
        logger.info("CREMA-D already present, skipping download.")
        return True
    
    try:
        import kagglehub
        logger.info("Downloading CREMA-D from Kaggle...")
        path = kagglehub.dataset_download("ejlok1/cremad")
        logger.info(f"Downloaded to: {path}")
        
        # Copy to data/cremad/
        src = Path(path) / "AudioWAV"
        if not src.exists():
            # Sometimes kagglehub puts it in a different structure
            for candidate in Path(path).rglob("AudioWAV"):
                src = candidate
                break
        
        if src.exists():
            wav_dir.mkdir(parents=True, exist_ok=True)
            for wav in src.glob("*.wav"):
                shutil.copy2(wav, wav_dir / wav.name)
            logger.info(f"✅ CREMA-D installed: {len(list(wav_dir.glob('*.wav')))} files")
            return True
        else:
            logger.error(f"AudioWAV not found in downloaded path: {path}")
            return False
    except ImportError:
        logger.error("kagglehub not installed. Run: pip install kagglehub")
        return False


def download_mustard_audio():
    """Download MUStARD++ audio from GitHub."""
    mustard_root = DATA_ROOT / "mustard"
    utterances_dir = mustard_root / "utterances_final"
    
    # Check if audio already exists
    if utterances_dir.exists():
        wav_count = len(list(utterances_dir.rglob("*.wav")))
        if wav_count > 0:
            logger.info(f"✅ MUStARD++ audio: {wav_count} WAV files found")
            return True
    
    # Check JSON exists
    json_path = mustard_root / "mustard++_raw_data.json"
    if not json_path.exists():
        logger.warning("MUStARD++ JSON not found, downloading full dataset...")
    
    logger.info("Downloading MUStARD++ audio from GitHub...")
    
    # Clone the repo to get utterances_final
    tmp_dir = DATA_ROOT / "_tmp_mustard"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Try git clone with sparse checkout (only utterances_final)
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/soujanyaporia/MUStARD.git",
            str(tmp_dir / "MUStARD")
        ], check=True, capture_output=True, text=True, timeout=600)
        
        src_utterances = tmp_dir / "MUStARD" / "utterances_final"
        if src_utterances.exists():
            utterances_dir.mkdir(parents=True, exist_ok=True)
            for f in src_utterances.rglob("*"):
                if f.is_file():
                    dest = utterances_dir / f.relative_to(src_utterances)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dest)
            wav_count = len(list(utterances_dir.rglob("*.wav")))
            logger.info(f"✅ MUStARD++ audio installed: {wav_count} files")
        else:
            # The repo may use a different structure or have the audio in releases
            logger.warning(
                "utterances_final not found in MUStARD repo.\n"
                "This dataset may require manual download:\n"
                "  1. Visit https://github.com/soujanyaporia/MUStARD\n"
                "  2. Download utterances_final.zip from Releases or linked Drive\n"
                "  3. Extract to data/mustard/utterances_final/"
            )
            return False
    except subprocess.TimeoutExpired:
        logger.error("Git clone timed out (>10 min)")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Git clone failed: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("git not found. Install git first.")
        return False
    finally:
        # Cleanup
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
    
    return True


def download_meld():
    """Download MELD dataset."""
    meld_root = DATA_ROOT / "meld"
    
    # Check if already present
    train_csv = meld_root / "train" / "train_sent_emo.csv"
    if train_csv.exists():
        train_wavs = meld_root / "train" / "train_splits"
        if train_wavs.exists():
            wav_count = len(list(train_wavs.glob("*.wav")))
            if wav_count > 0:
                logger.info(f"✅ MELD: {wav_count} train WAVs found")
                return True
    
    logger.info("Downloading MELD dataset...")
    
    # Try kagglehub first (most reliable)
    try:
        import kagglehub
        logger.info("Trying Kaggle download for MELD...")
        path = kagglehub.dataset_download("zaber666/meld-dataset")
        logger.info(f"Downloaded to: {path}")
        
        # Copy to data/meld/ preserving structure
        src = Path(path)
        meld_root.mkdir(parents=True, exist_ok=True)
        
        # Look for the expected structure
        for split in ["train", "dev", "test"]:
            # Find CSV
            csv_candidates = list(src.rglob(f"{split}_sent_emo.csv"))
            if csv_candidates:
                dest_dir = meld_root / split
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(csv_candidates[0], dest_dir / f"{split}_sent_emo.csv")
                logger.info(f"  Copied {split}_sent_emo.csv")
            
            # Find audio splits directory
            splits_candidates = list(src.rglob(f"{split}_splits"))
            if not splits_candidates:
                # Try alternate names
                splits_candidates = list(src.rglob(f"{split.replace('val','dev')}_splits"))
            if splits_candidates:
                dest_splits = meld_root / split / f"{split}_splits"
                if not dest_splits.exists():
                    shutil.copytree(splits_candidates[0], dest_splits)
                wav_count = len(list(dest_splits.glob("*.wav")))
                logger.info(f"  Copied {split}_splits: {wav_count} WAVs")
        
        # Verify
        if (meld_root / "train" / "train_sent_emo.csv").exists():
            logger.info("✅ MELD installed successfully")
            return True
        
    except ImportError:
        logger.warning("kagglehub not installed, trying direct download...")
    except Exception as e:
        logger.warning(f"Kaggle download failed: {e}")
    
    # Fallback: direct download from MELD GitHub releases
    logger.info("Trying direct download from MELD GitHub...")
    meld_root.mkdir(parents=True, exist_ok=True)
    
    urls = {
        "train": "https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz",
    }
    
    try:
        import urllib.request
        tar_path = meld_root / "MELD.Raw.tar.gz"
        
        logger.info("Downloading MELD.Raw.tar.gz (this may take a while, ~4GB)...")
        urllib.request.urlretrieve(urls["train"], str(tar_path))
        
        logger.info("Extracting...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=meld_root)
        
        tar_path.unlink()
        logger.info("✅ MELD extracted")
        return True
    except Exception as e:
        logger.error(f"Direct download failed: {e}")
        logger.info(
            "Manual download instructions:\n"
            "  1. Visit https://affective-meld.github.io/\n"
            "  2. Download train, dev, test splits\n"
            "  3. Extract to data/meld/{train,dev,test}/\n"
            "  OR\n"
            "  1. pip install kagglehub\n"
            "  2. python -c \"import kagglehub; kagglehub.dataset_download('zaber666/meld-dataset')\"\n"
            "  3. Copy to data/meld/"
        )
        return False


def check_iemocap():
    """Check IEMOCAP (cannot auto-download — requires USC license)."""
    iemocap_root = DATA_ROOT / "iemocap"
    if iemocap_root.exists():
        sessions = list(iemocap_root.glob("Session*"))
        if sessions:
            logger.info(f"✅ IEMOCAP: {len(sessions)} sessions found")
            return True
    
    logger.warning(
        "❌ IEMOCAP: Not found (requires USC academic license)\n"
        "   Request access at: https://sail.usc.edu/iemocap/iemocap_release.htm\n"
        "   Training can proceed WITHOUT IEMOCAP using the other 3 datasets."
    )
    return False


def main():
    logger.info("=" * 60)
    logger.info("ConflictNet Dataset Downloader")
    logger.info("=" * 60)
    logger.info(f"Data root: {DATA_ROOT}")
    print()
    
    results = {}
    
    # 1. CREMA-D
    logger.info("-" * 40)
    logger.info("1/4: CREMA-D")
    if check_cremad():
        results["CREMA-D"] = "✅ Ready"
    else:
        if download_cremad_kaggle():
            results["CREMA-D"] = "✅ Downloaded"
        else:
            results["CREMA-D"] = "❌ Failed"
    print()
    
    # 2. MUStARD++
    logger.info("-" * 40)
    logger.info("2/4: MUStARD++ Audio")
    if download_mustard_audio():
        results["MUStARD++"] = "✅ Ready"
    else:
        results["MUStARD++"] = "⚠️ Manual download needed"
    print()
    
    # 3. MELD
    logger.info("-" * 40)
    logger.info("3/4: MELD")
    if download_meld():
        results["MELD"] = "✅ Ready"
    else:
        results["MELD"] = "❌ Failed"
    print()
    
    # 4. IEMOCAP
    logger.info("-" * 40)
    logger.info("4/4: IEMOCAP")
    if check_iemocap():
        results["IEMOCAP"] = "✅ Ready"
    else:
        results["IEMOCAP"] = "⚠️ Requires USC license"
    print()
    
    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for name, status in results.items():
        logger.info(f"  {name:15s} {status}")
    
    ready_count = sum(1 for v in results.values() if "✅" in v)
    logger.info(f"\n  {ready_count}/4 datasets ready for training")
    
    if ready_count >= 2:
        logger.info("  You can start training with the available datasets!")
    
    return 0 if ready_count >= 2 else 1


if __name__ == "__main__":
    sys.exit(main())
