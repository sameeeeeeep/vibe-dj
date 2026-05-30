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

    def beat_times(self, until: float) -> np.ndarray:
        if self.beat_period <= 0:
            return np.array([])
        n = int(max(0, (until - self.beat_offset) / self.beat_period)) + 1
        return self.beat_offset + np.arange(n) * self.beat_period


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

    return Analysis(
        bpm=round(bpm, 2),
        beat_offset=beat_offset,
        beat_period=60.0 / bpm,
        duration=duration,
        rms=rms,
        centroid=centroid,
        onset_rate=onset_rate,
    )
