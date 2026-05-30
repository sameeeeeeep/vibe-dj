"""Generate a few synthetic tracks (known BPMs/energies) for testing the DJ.

Writes WAV files into an output folder. Each track is a four-on-the-floor kick
plus a hat and a bass tone, so the tempo estimator has something real to lock to.

    python tools/gen_demo_tracks.py demo_tracks
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.io import wavfile

SR = 44100


def kick(t: np.ndarray, dur=0.18) -> np.ndarray:
    env = np.exp(-t / 0.06) * (t < dur)
    freq = 120 * np.exp(-t / 0.03) + 45
    return np.sin(2 * np.pi * freq * t) * env


def hat(n: int) -> np.ndarray:
    env = np.exp(-np.linspace(0, 1, n) / 0.02)
    return (np.random.randn(n) * env) * 0.3


def make_track(bpm: float, seconds: float, energy: float) -> np.ndarray:
    beat = 60.0 / bpm
    n = int(seconds * SR)
    out = np.zeros(n, dtype=np.float32)
    tk = np.arange(int(0.25 * SR)) / SR
    one_kick = kick(tk).astype(np.float32)

    step = beat / 4.0  # 16th-note grid
    for i in range(int(seconds / step)):
        pos = int(i * step * SR)
        if i % 4 == 0:  # kick on every beat
            end = min(n, pos + one_kick.size)
            out[pos:end] += one_kick[: end - pos]
        if energy > 0.5 and i % 2 == 1:  # offbeat hats on busier tracks
            hn = int(0.05 * SR)
            end = min(n, pos + hn)
            out[pos:end] += hat(end - pos).astype(np.float32)

    # Bass pad whose brightness scales with energy.
    t = np.arange(n) / SR
    bass = 0.15 * np.sin(2 * np.pi * (55 + 110 * energy) * t)
    out += bass.astype(np.float32)

    out *= 0.6 + 0.4 * energy
    peak = np.abs(out).max() or 1.0
    return (out / peak * 0.9).astype(np.float32)


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "demo_tracks"
    os.makedirs(out_dir, exist_ok=True)
    specs = [
        ("warmup_100bpm", 100, 0.25),
        ("groove_118bpm", 118, 0.5),
        ("peak_124bpm", 124, 0.8),
        ("banger_128bpm", 128, 0.95),
    ]
    for name, bpm, energy in specs:
        audio = make_track(bpm, seconds=40, energy=energy)
        path = os.path.join(out_dir, f"{name}.wav")
        wavfile.write(path, SR, (audio * 32767).astype(np.int16))
        print(f"wrote {path}  ({bpm} BPM, energy {energy})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
