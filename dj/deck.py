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

WAVE_BINS = 64


def _wave_envelope(samples: np.ndarray, bins: int = WAVE_BINS) -> list[float]:
    """Peak envelope of a track, downsampled to `bins` values in 0..1, for the
    dashboard waveform. Peak (not RMS) so drops and transients read clearly; a
    small floor keeps quiet bins visible. Computed once per load, never in the
    audio callback."""
    n = len(samples)
    if n < bins:
        return [0.0] * bins
    mono = np.abs(samples).max(axis=1) if samples.ndim > 1 else np.abs(samples)
    edges = np.linspace(0, n, bins + 1).astype(np.int64)
    env = np.maximum.reduceat(mono, edges[:-1]).astype(np.float32)
    peak = float(env.max())
    if peak > 0:
        env = env / peak
    env = 0.06 + 0.94 * env
    return [round(float(x), 3) for x in env]


class Deck:
    def __init__(self, name: str):
        self.name = name
        self._lock = threading.Lock()
        self.samples = np.zeros((0, CHANNELS), dtype=np.float32)
        self.analysis: Analysis | None = None
        self.title = ""
        self.wave: list[float] = [0.0] * WAVE_BINS  # peak envelope for the UI
        self.pos = 0.0          # fractional frame index into samples
        self.rate = 1.0         # beatmatch playback speed (1.0 == original tempo)
        self.bend = 0.0         # manual pitch-fader offset (fraction, ±), DJ-driven
        self.gain = 0.0         # linear gain applied by the mixer
        self.playing = False
        # DJ overrides for the phrase-aligned mix cues (seconds). None = use the
        # analysis defaults; set by the dashboard's draggable cue markers so the
        # operator can hand-tune where the incoming track drops (mix_in) and where
        # the outgoing one starts its fade (mix_out). The live transition AND the
        # headphone-cue audition both read these, so what you align by ear is what
        # actually fires.
        self.mix_in_override: float | None = None
        self.mix_out_override: float | None = None

    def load(self, samples: np.ndarray, analysis: Analysis, title: str) -> None:
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        wave = _wave_envelope(samples)   # outside the lock; never stall the callback
        with self._lock:
            self.samples = samples
            self.wave = wave
            self.analysis = analysis
            self.title = title
            self.pos = 0.0
            self.rate = 1.0
            self.bend = 0.0
            self.gain = 0.0
            self.playing = False
            # A fresh track gets fresh cues: drop any hand-tuned overrides from
            # the track that was here before.
            self.mix_in_override = None
            self.mix_out_override = None

    def seek_fraction(self, frac: float) -> None:
        """Jump the playhead to a fraction (0..1) of the track. Drives the
        dashboard's waveform-seek and jog-wheel scrub. Resumes playback if the
        deck is audible (seeking the live track keeps it playing); a silent cued
        deck just repositions without starting."""
        with self._lock:
            n = len(self.samples)
            if n <= 1:
                return
            self.pos = float(min(1.0, max(0.0, frac))) * (n - 1)
            if self.gain > 0.0 or self.playing:
                self.playing = self.pos < n - 1

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

    # ---- cue points ------------------------------------------------------
    def eff_mix_in(self) -> float:
        """Seconds at which to bring this track IN — the DJ's hand-set marker if
        there is one, else the phrase-aligned analysis default."""
        if self.mix_in_override is not None:
            return self.mix_in_override
        return self.analysis.mix_in_sec() if self.analysis else 0.0

    def eff_mix_out(self) -> float:
        """Seconds at which to start mixing this track OUT — DJ override if set,
        else the analysis default (which is the duration when no outro found)."""
        if self.mix_out_override is not None:
            return self.mix_out_override
        return self.analysis.mix_out_sec() if self.analysis else 0.0

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

    def tail_slice(self, n_samples: int) -> np.ndarray:
        """A contiguous copy of up to `n_samples` frames ending at the current
        playhead. Used by the morph transition to capture the outgoing track's
        last beat for its stutter-loop riser. Returns a copy (not a view) so the
        caller can hold it without racing the deck; empty if nothing is loaded."""
        with self._lock:
            length = len(self.samples)
            if length == 0 or n_samples <= 0:
                return np.zeros((0, CHANNELS), dtype=np.float32)
            end = int(min(length, max(1, self.pos)))
            start = max(0, end - int(n_samples))
            return np.array(self.samples[start:end], dtype=np.float32)

    def read_preview(self, pos: float, n: int, rate: float) -> tuple[np.ndarray, float]:
        """Read `n` frames starting at fractional index `pos` at the given rate,
        WITHOUT touching the deck's live playhead/gain/playing state. Used by the
        headphone-cue engine (a second output device) to monitor or audition a
        deck independently of what's going to the speakers. Returns the audio plus
        the next position so the caller can carry its own preview playhead."""
        out = np.zeros((n, CHANNELS), dtype=np.float32)
        r = rate if rate > 0 else 1e-6
        with self._lock:
            samples = self.samples
            length = len(samples)
            if length == 0:
                return out, pos
            positions = pos + np.arange(n) * r
            idx0 = np.floor(positions).astype(np.int64)
            frac = (positions - idx0).astype(np.float32)[:, None]
            valid = (idx0 >= 0) & (idx0 < length - 1)
            if valid.any():
                i0 = idx0[valid]
                out[valid] = samples[i0] * (1.0 - frac[valid]) + samples[i0 + 1] * frac[valid]
        return out, pos + n * r
