"""Scan a folder, analyse each track (cached), and score energy across the set."""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import SAMPLE_RATE
from .analysis import Analysis, analyze
from .audio_io import decode

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".oga",
              ".opus", ".webm", ".mp4", ".aiff", ".aif"}
CACHE_NAME = ".dj_cache.json"


def score_energy(tracks: list["Track"]) -> None:
    """Rank-normalise each track's energy to 0..1 across the given set.

    Used for both the static library and the streaming pool, so a pool can
    re-score itself whenever a track is added or released.
    """
    if not tracks:
        return
    rms = np.array([t.analysis.rms for t in tracks])
    cen = np.array([t.analysis.centroid for t in tracks])
    ons = np.array([t.analysis.onset_rate for t in tracks])

    def rank(x: np.ndarray) -> np.ndarray:
        if len(x) == 1:
            return np.array([0.5])
        order = x.argsort()
        r = np.empty_like(order, dtype=float)
        r[order] = np.linspace(0.0, 1.0, len(x))
        return r

    score = 0.4 * rank(rms) + 0.4 * rank(ons) + 0.2 * rank(cen)
    for t, s in zip(tracks, score):
        t.energy = float(s)


@dataclass
class Track:
    path: str
    name: str
    analysis: Analysis
    energy: float = 0.5         # 0..1, normalised across the library
    play_count: int = 0
    last_played_at: float = 0.0  # monotonic seconds; 0 = never

    @property
    def bpm(self) -> float:
        return self.analysis.bpm


def _iter_audio_files(folder: str):
    for root, _dirs, files in os.walk(folder):
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() in AUDIO_EXTS:
                yield os.path.join(root, fn)


def _load_cache(folder: str) -> dict:
    path = os.path.join(folder, CACHE_NAME)
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(folder: str, cache: dict) -> None:
    path = os.path.join(folder, CACHE_NAME)
    try:
        with open(path, "w") as fh:
            json.dump(cache, fh)
    except OSError:
        pass


@dataclass
class Library:
    folder: str
    tracks: list[Track] = field(default_factory=list)

    def scan(self, progress: Optional[Callable[[int, int, str], None]] = None) -> "Library":
        files = list(_iter_audio_files(self.folder))
        cache = _load_cache(self.folder)
        new_cache: dict = {}
        self.tracks = []

        for i, path in enumerate(files, 1):
            name = os.path.splitext(os.path.basename(path))[0]
            if progress:
                progress(i, len(files), name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            key = path
            cached = cache.get(key)
            sig = {"mtime": st.st_mtime, "size": st.st_size}
            if cached and cached.get("mtime") == sig["mtime"] and cached.get("size") == sig["size"]:
                an = Analysis(**cached["analysis"])
            else:
                try:
                    audio = decode(path, sr=SAMPLE_RATE)
                except Exception:
                    continue
                an = analyze(audio, sr=SAMPLE_RATE)
            new_cache[key] = {**sig, "analysis": dataclasses.asdict(an)}
            self.tracks.append(Track(path=path, name=name, analysis=an))

        _save_cache(self.folder, new_cache)
        self._score_energy()
        return self

    def _score_energy(self) -> None:
        score_energy(self.tracks)

    def load_audio(self, track: Track) -> np.ndarray:
        return decode(track.path, sr=SAMPLE_RATE)

    def release(self, track: Track) -> None:
        """No-op for a static folder — we never delete the user's own files."""
        return
