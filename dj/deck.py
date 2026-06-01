"""A single deck: an in-memory track with a playhead, gain and varispeed rate.

Beatmatching is done by varispeed (resampling the read), which shifts pitch
slightly like a turntable pitch fader. For the small tempo deltas between
adjacent tracks this is musically acceptable and avoids a phase vocoder.
"""

from __future__ import annotations

import threading

import numpy as np

from . import CHANNELS, SAMPLE_RATE
from .analysis import Analysis


class Deck:
    def __init__(self, name: str):
        self.name = name
        self._lock = threading.Lock()
        self.samples = np.zeros((0, CHANNELS), dtype=np.float32)
        self.analysis: Analysis | None = None
        self.title = ""
        self.pos = 0.0          # fractional frame index into samples
        self.rate = 1.0         # beatmatch playback speed (1.0 == original tempo)
        self.bend = 0.0         # manual pitch-fader offset (fraction, ±), DJ-driven
        self.gain = 0.0         # linear gain applied by the mixer
        self.playing = False

    def load(self, samples: np.ndarray, analysis: Analysis, title: str) -> None:
        with self._lock:
            self.samples = np.ascontiguousarray(samples, dtype=np.float32)
            self.analysis = analysis
            self.title = title
            self.pos = 0.0
            self.rate = 1.0
            self.bend = 0.0
            self.gain = 0.0
            self.playing = False

    @property
    def eff_rate(self) -> float:
        """Beatmatch rate combined with the manual pitch bend; always > 0."""
        r = self.rate * (1.0 + self.bend)
        return r if r > 0 else 1e-6

    @property
    def effective_bpm(self) -> float:
        if not self.analysis:
            return 0.0
        return self.analysis.bpm * self.eff_rate

    @property
    def position_sec(self) -> float:
        return self.pos / SAMPLE_RATE

    @property
    def remaining_sec(self) -> float:
        with self._lock:
            n = len(self.samples)
            if n == 0:
                return 0.0
            return max(0.0, (n - self.pos) / (SAMPLE_RATE * self.eff_rate))

    def seconds_to_next_beat(self) -> float:
        """Wall-clock seconds until this deck's next beat at its current rate."""
        if not self.analysis or self.analysis.beat_period <= 0:
            return 0.0
        src_period = self.analysis.beat_period          # seconds in source timeline
        t = self.position_sec - self.analysis.beat_offset
        into = t % src_period
        src_to_next = (src_period - into) % src_period
        return src_to_next / self.eff_rate              # compress by varispeed

    def read(self, n: int) -> np.ndarray:
        out = np.zeros((n, CHANNELS), dtype=np.float32)
        with self._lock:
            samples = self.samples
            length = len(samples)
            if not self.playing or length == 0:
                return out
            r = self.eff_rate
            positions = self.pos + np.arange(n) * r
            self.pos = self.pos + n * r
            idx0 = np.floor(positions).astype(np.int64)
            frac = (positions - idx0).astype(np.float32)[:, None]
            valid = (idx0 >= 0) & (idx0 < length - 1)
            if valid.any():
                i0 = idx0[valid]
                s0 = samples[i0]
                s1 = samples[i0 + 1]
                out[valid] = s0 * (1.0 - frac[valid]) + s1 * frac[valid]
            if self.pos >= length - 1:
                self.playing = False
            return out
