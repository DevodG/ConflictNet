#!/usr/bin/env python3
"""Connect to Lightning AI and run ConflictNet training on H200.

All credentials come from the environment — never hardcode secrets::

    export LIGHTNING_USER=<studio username>
    export LIGHTNING_STUDIO=training-devbox
    export LIGHTNING_TEAMSPACE=<teamspace>
    pip install lightning_sdk
    python lightning_train.py
"""

from __future__ import annotations

import os
import time


def main() -> None:
    try:
        from lightning_sdk import Studio, Teamspace, User
    except ImportError:
        raise SystemExit("pip install lightning_sdk to use the remote launcher")

    studio_name = os.environ.get("LIGHTNING_STUDIO", "training-devbox")
    teamspace_name = os.environ.get("LIGHTNING_TEAMSPACE", "financial-llm-training-project")
    username = os.environ.get("LIGHTNING_USER")
    if not username:
        raise SystemExit("Set LIGHTNING_USER to your Lightning AI username")

    print("Connecting to Lightning AI...")
    user = User(name=username)
    teamspace = Teamspace(name=teamspace_name, user=user)
    studio = Studio(name=studio_name, teamspace=teamspace)

    if studio.status != "Running":
        print(f"Starting studio '{studio_name}'...")
        studio.start()
        print("Waiting for studio to be ready...")
        time.sleep(30)

    commands = [
        "cd /teamspace/studios/this_studio",
        "git clone https://github.com/DevodG/ConflictNet.git conflictnet 2>/dev/null || true",
        "cd conflictnet",
        "git pull origin main",
        "pip install -q -r requirements.txt sentencepiece tiktoken kagglehub speechbrain audiomentations",
    ]

    print("Running setup commands...")
    for cmd in commands:
        print(f"  $ {cmd}")
        studio.run(cmd)

    print("Downloading CREMA-D dataset...")
    r = studio.run("python -c \"import kagglehub; print('Downloaded to:', kagglehub.dataset_download('ejlok1/cremad'))\" 2>&1 | tail -3")
    print(r)

    print("Starting training on H200...")
    cremad = "/teamspace/studios/this_studio/.cache/kagglehub/datasets/ejlok1/cremad/versions/1"
    cmd = (
        "cd /teamspace/studios/this_studio/conflictnet && "
        "nohup python scripts/train.py "
        f"--cremad_root {cremad} "
        "--epochs 30 "
        "--batch_size 32 "
        "--lr 3e-5 "
        "--audio_encoder wavlm "
        "--gradient_accumulation_steps 1 "
        "--pretrain_epochs 5 "
        "--amp "
        "--output_dir /teamspace/studios/this_studio/conflictnet/checkpoints "
        "--target_f1 0.78 "
        "--max_retries 2 "
        "--resume_epochs 10 "
        "> training.log 2>&1 &"
    )
    studio.run(cmd)
    print("Training launched. Check progress with: tail -f training.log")


if __name__ == "__main__":
    main()
