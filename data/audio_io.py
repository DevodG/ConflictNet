"""Backend-free audio IO for ConflictNet.

The torchaudio file loaders require the TorchCodec backend on recent
torchaudio versions, which is a heavy optional dependency that is absent
from minimal installs (and CI). All waveform file IO in this repo goes
through :mod:`soundfile` (libsndfile) instead; resampling stays in
pure-torch (torchaudio.functional), which needs no backend.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

PathLike = Union[str, Path, BinaryIO]

SAMPLE_RATE = 16000
MAX_AUDIO_LEN = 10.0  # seconds


def read_mono(
    path_or_buf: PathLike,
    target_sr: int = SAMPLE_RATE,
    max_len_s: float = MAX_AUDIO_LEN,
) -> torch.Tensor:
    """Read an audio file as a mono waveform tensor.

    Args:
        path_or_buf: File path or binary buffer (WAV/FLAC/OGG via libsndfile).
        target_sr: Resample target. ``0``/``None`` keeps the native rate.
        max_len_s: Truncate to this many seconds (``<= 0`` disables).

    Returns:
        ``(T,)`` float32 mono tensor.
    """
    if isinstance(path_or_buf, io.BytesIO):
        path_or_buf.seek(0)
    data, sr = sf.read(path_or_buf, dtype="float32", always_2d=True)  # (T, C)
    mono = data.mean(axis=1)  # stereo → mono; (T,)
    waveform = torch.from_numpy(np.ascontiguousarray(mono)).unsqueeze(0)  # (1, T)
    if target_sr and sr != target_sr:
        waveform = AF.resample(waveform, sr, target_sr)
    if max_len_s and max_len_s > 0:
        max_samples = int(max_len_s * (target_sr or sr))
        waveform = waveform[:, :max_samples]
    return waveform.squeeze(0)


def write_wav(
    path_or_buf: PathLike,
    waveform: torch.Tensor,
    sample_rate: int,
) -> None:
    """Write a waveform tensor as 16-bit PCM WAV.

    Args:
        path_or_buf: File path or binary buffer.
        waveform: ``(T,)`` or ``(C, T)`` float tensor.
        sample_rate: Sample rate in Hz.
    """
    wav = waveform.detach().cpu()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    kwargs = {} if isinstance(path_or_buf, (str, Path)) else {"format": "WAV"}
    sf.write(path_or_buf, wav.transpose(0, 1).numpy(), sample_rate, subtype="PCM_16", **kwargs)


def read_native(path_or_buf: PathLike) -> tuple[torch.Tensor, int]:
    """Read an audio file at its native rate as ``((C, T), sr)``."""
    if isinstance(path_or_buf, io.BytesIO):
        path_or_buf.seek(0)
    data, sr = sf.read(path_or_buf, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(np.ascontiguousarray(data)).transpose(0, 1)
    return waveform, sr


def sample_rate_of(path_or_buf: PathLike) -> int:
    """Return the sample rate of an audio file without reading samples."""
    if isinstance(path_or_buf, io.BytesIO):
        path_or_buf.seek(0)
    with sf.SoundFile(path_or_buf) as f:
        return f.samplerate


__all__ = [
    "SAMPLE_RATE",
    "MAX_AUDIO_LEN",
    "read_mono",
    "read_native",
    "write_wav",
    "sample_rate_of",
]
