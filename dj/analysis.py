"""Tempo, beat-grid and energy analysis using only numpy/scipy.

This intentionally avoids librosa/numba so it installs on any Python. The tempo
estimator is the classic pipeline: spectral-flux onset envelope -> autocorrelation
with a log-Gaussian tempo prior -> comb-filter phase search for the beat grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve, resample_poly, stft

from . import SAMPLE_RATE
from .audio_io import to_mono

ANALYSIS_SR = 22050
HOP = 512
NFFT = 2048
MIN_BPM = 70.0
MAX_BPM = 180.0
PRIOR_CENTER_BPM = 120.0
PRIOR_WIDTH = 0.9  # in octaves

# Phrase grid: 4/4 bars grouped into phrases. Mix points snap to this grid so
# blends land on musical "sentence" boundaries, not mid-phrase.
BEATS_PER_BAR = 4
PHRASE_BARS = 8

# Section detection (intro / outro) from a smoothed loudness envelope.
SECTION_SMOOTH_SEC = 1.5   # smooth past beat-level flicker
LOUD_FRAC = 0.45           # fraction of peak energy that counts as "full"
SUSTAIN_SEC = 4.0          # how long energy must hold to call the intro over
INTRO_MAX_FRAC = 0.4       # ignore "intros" longer than this share of the track
OUTRO_MIN_FRAC = 0.5       # an outro must sit in the back half to be real


@dataclass
class Analysis:
    bpm: float
    beat_offset: float      # seconds: time of the first beat in the grid
    beat_period: float      # seconds between beats (== 60 / bpm)
    duration: float         # seconds
    # Raw energy components; normalised into a 0..1 score by the library.
    rms: float
    centroid: float         # mean spectral centroid (Hz) -> brightness
    onset_rate: float       # mean onset-envelope value -> rhythmic density
    # Structure (0/duration defaults => "no section found", callers fall back).
    intro_end: float = 0.0      # seconds: where the first full-energy section starts
    outro_start: float = 0.0    # seconds: where the final wind-down begins
    phrase_period: float = 0.0  # seconds per phrase (BEATS_PER_BAR * PHRASE_BARS beats)

    def beat_times(self, until: float) -> np.ndarray:
        if self.beat_period <= 0:
            return np.array([])
        n = int(max(0, (until - self.beat_offset) / self.beat_period)) + 1
        return self.beat_offset + np.arange(n) * self.beat_period

    def phrase_times(self, until: float) -> np.ndarray:
        if self.phrase_period <= 0:
            return np.array([])
        n = int(max(0, (until - self.beat_offset) / self.phrase_period)) + 1
        return self.beat_offset + np.arange(n) * self.phrase_period

    def _snap_phrase(self, t: float) -> float:
        """Nearest phrase boundary to t, clamped within the track."""
        if self.phrase_period <= 0:
            return t
        k = max(0, round((t - self.beat_offset) / self.phrase_period))
        return float(min(max(0.0, self.beat_offset + k * self.phrase_period), self.duration))

    def mix_in_sec(self) -> float:
        """Phrase-aligned point to bring this track IN (past its intro). Falls
        back to the first beat when no structure was detected."""
        if self.phrase_period <= 0:
            return self.beat_offset
        return self._snap_phrase(self.intro_end)

    def mix_out_sec(self) -> float:
        """Phrase-aligned point to start mixing this track OUT (into its outro).
        Returns duration when no outro was detected, so the caller mixes at the
        track's end instead."""
        if self.phrase_period <= 0 or self.outro_start >= self.duration:
            return self.duration
        return self._snap_phrase(self.outro_start)


def _onset_envelope(mono: np.ndarray, sr: int):
    f, _, Z = stft(mono, fs=sr, nperseg=NFFT, noverlap=NFFT - HOP, boundary=None)
    S = np.abs(Z)
    # Spectral flux: positive frame-to-frame magnitude increases, summed over freq.
    flux = np.diff(S, axis=1)
    flux[flux < 0] = 0.0
    oenv = flux.sum(axis=0)
    # Normalise so autocorrelation isn't dominated by overall loudness.
    if oenv.std() > 1e-9:
        oenv = (oenv - oenv.mean()) / oenv.std()
        oenv[oenv < 0] = 0.0
    fps = sr / HOP
    return oenv, S, f, fps


def _estimate_tempo(oenv: np.ndarray, fps: float) -> float:
    if oenv.size < 4:
        return PRIOR_CENTER_BPM
    ac = fftconvolve(oenv, oenv[::-1], mode="full")
    ac = ac[oenv.size - 1:]          # lags 0..N-1
    ac[0] = 0.0
    lags = np.arange(ac.size)
    with np.errstate(divide="ignore"):
        bpm = 60.0 * fps / lags
    valid = (bpm >= MIN_BPM) & (bpm <= MAX_BPM) & (ac > 0)
    if not valid.any():
        return PRIOR_CENTER_BPM
    prior = np.exp(-0.5 * (np.log2(bpm[valid] / PRIOR_CENTER_BPM) / PRIOR_WIDTH) ** 2)
    score = ac[valid] * prior
    return float(bpm[valid][int(np.argmax(score))])


def _estimate_phase(oenv: np.ndarray, fps: float, bpm: float) -> float:
    period = 60.0 * fps / bpm           # frames per beat
    if period < 1:
        return 0.0
    candidates = max(1, int(round(period)))
    best_phase, best_score = 0, -1.0
    idx = np.arange(oenv.size)
    for phase in range(candidates):
        beats = np.round(np.arange(phase, oenv.size, period)).astype(int)
        beats = beats[beats < oenv.size]
        score = float(oenv[beats].sum())
        if score > best_score:
            best_score, best_phase = score, phase
    return best_phase / fps             # seconds


def _detect_sections(S: np.ndarray, fps: float, duration: float) -> tuple[float, float]:
    """Coarse intro/outro boundaries from a smoothed loudness envelope.

    Returns (intro_end, outro_start) in seconds. intro_end is 0 and outro_start
    is the duration when no clear section is found, so callers fall back to
    start-of-track / end-of-track behaviour.
    """
    if S.size == 0 or fps <= 0:
        return 0.0, duration
    e = (S.astype(np.float64) ** 2).sum(axis=0)
    win = max(1, int(round(SECTION_SMOOTH_SEC * fps)))
    if win > 1:
        e = np.convolve(e, np.ones(win) / win, mode="same")
    n = e.size
    peak = float(np.percentile(e, 90))
    if n < 4 or peak <= 0:
        return 0.0, duration
    loud = (e / peak) >= LOUD_FRAC
    if not loud.any():
        return 0.0, duration

    # intro_end: first frame where the track stays mostly loud for SUSTAIN_SEC.
    intro_end = 0.0
    sustain = max(1, int(round(SUSTAIN_SEC * fps)))
    if n >= sustain:
        csum = np.concatenate([[0], np.cumsum(loud.astype(np.int64))])
        intro_frame = 0
        for i in range(0, n - sustain + 1):
            if csum[i + sustain] - csum[i] >= 0.8 * sustain:
                intro_frame = i
                break
        intro_end = intro_frame / fps
        if intro_end > INTRO_MAX_FRAC * duration:
            intro_end = 0.0

    # outro_start: after the last loud frame the track has wound down for good.
    outro_start = int(np.flatnonzero(loud)[-1]) / fps
    if outro_start < OUTRO_MIN_FRAC * duration or outro_start >= duration - 0.5:
        outro_start = duration

    return float(intro_end), float(outro_start)


def analyze(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Analysis:
    mono = to_mono(audio).astype(np.float32)
    duration = mono.size / sr
    if sr != ANALYSIS_SR and mono.size:
        # Rational resample to the analysis rate (e.g. 44100 -> 22050 is 1/2).
        g = np.gcd(int(sr), ANALYSIS_SR)
        mono = resample_poly(mono, ANALYSIS_SR // g, sr // g)
    asr = ANALYSIS_SR

    if mono.size < NFFT * 2:
        return Analysis(PRIOR_CENTER_BPM, 0.0, 60.0 / PRIOR_CENTER_BPM,
                        duration, 0.0, 0.0, 0.0)

    oenv, S, freqs, fps = _onset_envelope(mono, asr)
    bpm = _estimate_tempo(oenv, fps)
    beat_offset = _estimate_phase(oenv, fps, bpm)

    rms = float(np.sqrt(np.mean(mono ** 2)) + 1e-12)
    mag = S.sum(axis=0)
    nz = mag > 1e-9
    centroid = float((freqs[:, None] * S)[:, nz].sum() / mag[nz].sum()) if nz.any() else 0.0
    onset_rate = float(oenv.mean())

    intro_end, outro_start = _detect_sections(S, fps, duration)
    phrase_period = (60.0 / bpm) * BEATS_PER_BAR * PHRASE_BARS if bpm > 0 else 0.0

    return Analysis(
        bpm=round(bpm, 2),
        beat_offset=beat_offset,
        beat_period=60.0 / bpm,
        duration=duration,
        rms=rms,
        centroid=centroid,
        onset_rate=onset_rate,
        intro_end=intro_end,
        outro_start=outro_start,
        phrase_period=phrase_period,
    )
