"""Audio decoding via ffmpeg.

We shell out to ffmpeg rather than depending on libsndfile/librosa so that any
format ffmpeg understands (mp3, m4a, flac, wav, ogg, ...) decodes the same way,
and so the project installs cleanly on bleeding-edge Python where the DSP wheels
may not exist yet.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np

from . import CHANNELS, SAMPLE_RATE


class FfmpegMissing(RuntimeError):
    pass


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def decode(path: str, sr: int = SAMPLE_RATE, channels: int = CHANNELS) -> np.ndarray:
    """Decode an audio file to a float32 array of shape (frames, channels), range ~[-1, 1]."""
    if not have_ffmpeg():
        raise FfmpegMissing("ffmpeg not found on PATH; install it (brew install ffmpeg).")

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v", "error",
        "-i", path,
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", str(channels),
        "-ar", str(sr),
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode {path!r}: {msg}")

    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if channels > 1:
        # Trailing partial frame guard, then reshape interleaved samples.
        usable = (audio.size // channels) * channels
        audio = audio[:usable].reshape(-1, channels)
    else:
        audio = audio.reshape(-1, 1)
    return np.ascontiguousarray(audio)


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1)
