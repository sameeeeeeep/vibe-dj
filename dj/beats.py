"""A procedural drum-loop layer — a "beats" pad the DJ drops over the live track.

No samples on disk (disk is tight) and no librosa: the voices (kick/clap/hat/…)
are synthesised once with numpy, then sequenced into a one-bar loop that is
*beat-locked* to the live deck and *tempo-matched* by sizing the bar to the
deck's heard tempo. Intensity (how busy the pattern is) follows the song's vibe
— minimal on mellow cuts, driving on bangers — or a manual setting.

How the lock works: each audio block we read the live deck's bar phase (0..1
from its beat grid, beat_offset assumed a downbeat) and map it onto the loop. If
the loop length equals the heard bar length, the read index advances by exactly
`n` per block, so playback is contiguous AND stays pinned to the deck's downbeat
for free. Tempo follows because the bar is re-sized when the deck's varispeed
changes; intensity changes just re-sequence the cached one-shots.
"""

from __future__ import annotations

import threading

import numpy as np

from . import CHANNELS, SAMPLE_RATE


# ---- one-shot voice synthesis (numpy only) -------------------------------
def _t(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.float32) / SAMPLE_RATE


def _noise(n: int, seed: int) -> np.ndarray:
    return np.random.RandomState(seed).randn(n).astype(np.float32)


def _hp(x: np.ndarray) -> np.ndarray:
    """Crude one-pole high-pass (first difference) — enough to make noise hiss
    read as a hi-hat rather than a rumble."""
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - 0.97 * x[:-1]
    return y


def _norm(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(x))) + 1e-9
    return (x / m * peak).astype(np.float32)


def _kick(dur: float = 0.23) -> np.ndarray:
    n = int(dur * SAMPLE_RATE)
    t = _t(n)
    f = 48.0 + (118.0 - 48.0) * np.exp(-t * 42.0)         # pitch drop 118→48 Hz
    phase = 2 * np.pi * np.cumsum(f) / SAMPLE_RATE
    body = np.sin(phase) * np.exp(-t * 7.0)
    click = _noise(n, 1) * np.exp(-t * 240.0) * 0.35       # beater transient
    return _norm(body + click)


def _closed_hat(dur: float = 0.05) -> np.ndarray:
    n = int(dur * SAMPLE_RATE)
    return _norm(_hp(_noise(n, 2)) * np.exp(-_t(n) * 90.0), 0.7)


def _open_hat(dur: float = 0.20) -> np.ndarray:
    n = int(dur * SAMPLE_RATE)
    return _norm(_hp(_noise(n, 3)) * np.exp(-_t(n) * 14.0), 0.6)


def _clap(dur: float = 0.18) -> np.ndarray:
    n = int(dur * SAMPLE_RATE)
    t = _t(n)
    # Three quick noise bursts (the hand-slap stutter) into a short tail.
    env = np.zeros(n, dtype=np.float32)
    for d in (0.0, 0.008, 0.016):
        k = int(d * SAMPLE_RATE)
        env[k:] += np.exp(-_t(n - k) * 130.0)
    env += np.exp(-t * 24.0) * 0.6
    body = _hp(_noise(n, 4)) * env
    return _norm(body, 0.8)


def _snare(dur: float = 0.14) -> np.ndarray:
    n = int(dur * SAMPLE_RATE)
    t = _t(n)
    tone = np.sin(2 * np.pi * 190.0 * t) * np.exp(-t * 30.0)
    noise = _hp(_noise(n, 5)) * np.exp(-t * 22.0)
    return _norm(tone * 0.5 + noise)


def _perc(dur: float = 0.11) -> np.ndarray:
    n = int(dur * SAMPLE_RATE)
    t = _t(n)
    tone = np.sin(2 * np.pi * 330.0 * t) * np.exp(-t * 38.0)
    noise = _noise(n, 6) * np.exp(-t * 70.0) * 0.25
    return _norm(tone + noise, 0.8)


# ---- patterns: which voices hit on which 16th step, per style + intensity --
# Each builder returns a one-bar grid as (voice, step, velocity). `step` may be
# fractional (e.g. 12.5 for a 32nd-note in a hat roll) — _synth_bar places it at
# step/16 of the bar, so half-steps land between the 16th grid lines. Patterns
# get busier + louder as intensity rises (minimal cut → full-kit banger).

def _p_four_floor(inten: float) -> list[tuple[float, ...]]:
    """The original house-leaning four-on-the-floor: soft kick + offbeat hats
    when minimal, full kit with 16th hats, ghost claps and perc up top."""
    hits: list[tuple] = []
    kv = 0.85 if inten >= 0.33 else 0.68
    for s in (0, 4, 8, 12):
        hits.append(("kick", s, kv))
    if inten < 0.33:
        for s in (2, 6, 10, 14):
            hits.append(("chat", s, 0.32))
    elif inten < 0.66:
        for s in (2, 6, 10, 14):
            hits.append(("chat", s, 0.5))
        for s in (0, 4, 8, 12):
            hits.append(("chat", s, 0.22))
    else:
        for s in range(0, 16, 2):
            hits.append(("chat", s, 0.5))
        for s in range(1, 16, 2):
            hits.append(("chat", s, 0.28))
    if inten >= 0.33:
        hits.append(("clap", 4, 0.8))
        hits.append(("clap", 12, 0.8))
    if inten >= 0.66:
        hits.append(("clap", 7, 0.28))      # ghost
        hits.append(("ohat", 14, 0.5))      # open-hat lift into the 1
        hits.append(("perc", 10, 0.45))
        hits.append(("perc", 3, 0.4))
    elif inten >= 0.33:
        hits.append(("ohat", 14, 0.38))
    return hits


def _p_house(inten: float) -> list[tuple[float, ...]]:
    """Classic house: four-on-the-floor with the signature open-hat on every
    offbeat 8th, clap backbeat, swung closed-hat fills as it builds."""
    hits: list[tuple] = []
    for s in (0, 4, 8, 12):
        hits.append(("kick", s, 0.82))
    ov = 0.32 + 0.22 * inten
    for s in (2, 6, 10, 14):                 # the house "tss" offbeat open hat
        hits.append(("ohat", s, ov))
    if inten >= 0.25:
        hits.append(("clap", 4, 0.75))
        hits.append(("clap", 12, 0.75))
    if inten >= 0.5:
        for s in (3, 7, 11, 15):
            hits.append(("chat", s, 0.3))
    if inten >= 0.75:
        for s in (1, 5, 9, 13):
            hits.append(("chat", s, 0.2))
        hits.append(("perc", 10, 0.4))
    return hits


def _p_techno(inten: float) -> list[tuple[float, ...]]:
    """Driving techno: relentless kick, offbeat open hat, 16th closed hats that
    fill in with energy, perc stabs and a clap accent up top."""
    hits: list[tuple] = []
    for s in (0, 4, 8, 12):
        hits.append(("kick", s, 0.9))
    for s in (2, 6, 10, 14):
        hits.append(("ohat", s, 0.3 + 0.2 * inten))
    if inten < 0.4:
        for s in range(2, 16, 4):
            hits.append(("chat", s, 0.3))
    elif inten < 0.7:
        for s in range(0, 16, 2):
            hits.append(("chat", s, 0.4))
    else:
        for s in range(0, 16):
            hits.append(("chat", s, 0.38 if s % 2 == 0 else 0.22))
    if inten >= 0.5:
        hits.append(("perc", 6, 0.4))
        hits.append(("perc", 14, 0.4))
    if inten >= 0.75:
        hits.append(("clap", 12, 0.5))
        hits.append(("perc", 3, 0.35))
    return hits


def _p_breakbeat(inten: float) -> list[tuple[float, ...]]:
    """Broken beat (amen-ish): syncopated kick, snare backbone on 2 & 4, busy
    off-grid hats with ghost snares as it intensifies."""
    hits: list[tuple] = []
    hits.append(("kick", 0, 0.9))
    hits.append(("kick", 6, 0.7))
    hits.append(("kick", 10, 0.6))
    if inten >= 0.5:
        hits.append(("kick", 11, 0.5))
    hits.append(("snare", 4, 0.85))          # backbeat (beats 2 & 4)
    hits.append(("snare", 12, 0.85))
    if inten >= 0.6:
        hits.append(("snare", 14, 0.4))      # ghost
        hits.append(("snare", 7, 0.3))
    if inten < 0.4:
        for s in (2, 6, 10, 14):
            hits.append(("chat", s, 0.3))
    else:
        for s in range(0, 16, 2):
            hits.append(("chat", s, 0.38))
        if inten >= 0.7:
            for s in range(1, 16, 2):
                hits.append(("chat", s, 0.22))
            hits.append(("ohat", 14, 0.45))
    return hits


def _p_trap(inten: float) -> list[tuple[float, ...]]:
    """Half-time trap: booming syncopated kick, snare/clap on the 3, rolling
    hi-hats that subdivide into 32nd-note rolls (fractional steps) up top."""
    hits: list[tuple] = []
    hits.append(("kick", 0, 0.95))
    hits.append(("kick", 3, 0.6))
    hits.append(("kick", 8, 0.7))
    hits.append(("kick", 11, 0.55))
    if inten >= 0.6:
        hits.append(("kick", 14, 0.5))
    hits.append(("snare", 8, 0.9))           # half-time backbeat on the 3
    if inten >= 0.5:
        hits.append(("clap", 8, 0.5))
    if inten < 0.4:
        for s in range(0, 16, 2):
            hits.append(("chat", s, 0.32))
    elif inten < 0.7:
        for s in range(0, 16):
            hits.append(("chat", s, 0.26))
    else:
        for s in range(0, 16):
            hits.append(("chat", s, 0.26))
        for s in (12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5):
            hits.append(("chat", s, 0.24))   # 32nd-note roll into the 1
        hits.append(("ohat", 6, 0.4))
    return hits


def _p_afro(inten: float) -> list[tuple[float, ...]]:
    """Afro house: four-on-the-floor under a syncopated conga/shaker groove with
    offbeat open hats — rolling and hypnotic, building extra perc up top."""
    hits: list[tuple] = []
    for s in (0, 4, 8, 12):
        hits.append(("kick", s, 0.78))
    for s in (2, 6, 10, 14):                 # shaker = swung closed hat
        hits.append(("chat", s, 0.34))
    if inten >= 0.4:
        for s in (3, 7, 11, 15):
            hits.append(("chat", s, 0.2))
    for s in (2, 10):
        hits.append(("ohat", s, 0.32))
    hits.append(("perc", 3, 0.5))            # conga / clave-ish syncopation
    hits.append(("perc", 6, 0.42))
    hits.append(("perc", 11, 0.5))
    if inten >= 0.5:
        hits.append(("perc", 7, 0.36))
        hits.append(("perc", 14, 0.4))
        hits.append(("clap", 12, 0.45))
    if inten >= 0.75:
        hits.append(("perc", 1, 0.3))
        hits.append(("perc", 9, 0.34))
    return hits


# Order = chill → hype; AUTO walks this 4-on-the-floor family by vibe. Breakbeat
# and trap change the feel too drastically (broken / half-time) to impose
# automatically, so they're deliberate manual picks only.
STYLES = ("four_floor", "house", "techno", "breakbeat", "trap", "afro")
_BUILDERS = {
    "four_floor": _p_four_floor, "house": _p_house, "techno": _p_techno,
    "breakbeat": _p_breakbeat, "trap": _p_trap, "afro": _p_afro,
}


def _pattern(style: str, inten: float) -> list[tuple[float, ...]]:
    return _BUILDERS.get(style, _p_four_floor)(inten)


class BeatMachine:
    """The drum layer. Thread-safe enough for the audio callback: control state
    is plain attributes (atomic assigns under the GIL) and render() is allocation
    -light + wrapped so it can never crash the callback."""

    def __init__(self) -> None:
        self.enabled = False
        self.level = 0.55          # mix gain of the whole drum layer (0..1)
        self.mode = "auto"         # "auto" (follow song vibe) | "manual" intensity
        self.manual_intensity = 0.5
        self.vibe = 0.5            # song energy 0..1, pushed in by the controller
        self.style = "auto"        # "auto" (pick by vibe) | a name from STYLES

        self._voices = {
            "kick": _kick(), "chat": _closed_hat(), "ohat": _open_hat(),
            "clap": _clap(), "snare": _snare(), "perc": _perc(),
        }
        self._lock = threading.Lock()
        self._loop: np.ndarray | None = None   # (bar_frames, CHANNELS)
        self._loop_frames = 0
        self._loop_inten = -1.0
        self._loop_style = ""

    # ---- controls --------------------------------------------------------
    def set_enabled(self, on: bool) -> None:
        self.enabled = bool(on)

    def toggle(self) -> None:
        self.enabled = not self.enabled

    def set_level(self, v: float) -> None:
        self.level = float(min(1.0, max(0.0, v)))

    def set_mode(self, m: str) -> None:
        if m in ("auto", "manual"):
            self.mode = m

    def set_intensity(self, v: float) -> None:
        self.manual_intensity = float(min(1.0, max(0.0, v)))

    def set_vibe(self, v: float) -> None:
        self.vibe = float(min(1.0, max(0.0, v)))

    def set_style(self, name: str) -> None:
        if name == "auto" or name in STYLES:
            self.style = name

    def intensity(self) -> float:
        return self.vibe if self.mode == "auto" else self.manual_intensity

    def _auto_style(self) -> str:
        """Walk the 4-on-the-floor family by vibe: mellow→afro, building→house,
        peak→four_floor, banger→techno."""
        v = self.vibe
        if v < 0.30:
            return "afro"
        if v < 0.52:
            return "house"
        if v < 0.74:
            return "four_floor"
        return "techno"

    def effective_style(self) -> str:
        return self._auto_style() if self.style == "auto" else self.style

    def state(self) -> dict:
        return {
            "enabled": self.enabled,
            "level": round(self.level, 3),
            "mode": self.mode,
            "intensity": round(self.intensity(), 3),
            "manual_intensity": round(self.manual_intensity, 3),
            "vibe": round(self.vibe, 3),
            "style": self.style,
            "style_eff": self.effective_style(),
        }

    # ---- audio -----------------------------------------------------------
    def _synth_bar(self, bar_frames: int, style: str, inten: float) -> np.ndarray:
        buf = np.zeros(bar_frames, dtype=np.float32)
        sf = bar_frames / 16.0
        for voice, step, vel in _pattern(style, inten):
            v = self._voices.get(voice)
            if v is None:
                continue
            off = int(round(step * sf)) % bar_frames
            end = off + len(v)
            if end <= bar_frames:
                buf[off:end] += v * vel
            else:                                   # wrap a late hit's tail round
                k = bar_frames - off
                buf[off:] += v[:k] * vel
                buf[:end - bar_frames] += v[k:] * vel
        np.clip(buf, -1.0, 1.0, out=buf)
        return np.stack([buf, buf], axis=1).astype(np.float32)

    def render(self, n: int, deck) -> np.ndarray:
        """One block of the drum loop, tempo-matched + beat-locked to `deck`.
        Returns silence when off, or when the deck has no grid / isn't playing."""
        try:
            if not self.enabled:
                return np.zeros((n, CHANNELS), dtype=np.float32)
            an = getattr(deck, "analysis", None)
            if an is None or not deck.playing or an.beat_period <= 0:
                return np.zeros((n, CHANNELS), dtype=np.float32)
            eff = deck.eff_rate
            bar_frames = int(round(4 * an.beat_period / eff * SAMPLE_RATE))
            if bar_frames < 2048:
                return np.zeros((n, CHANNELS), dtype=np.float32)
            inten = self.intensity()
            style = self.effective_style()
            ib = round(inten * 4) / 4.0             # quantise to 5 intensity steps
            # Re-sequence only on a real change: style switch, intensity step, or
            # the heard tempo drifting > ~0.6% (a varispeed change) — not every
            # block.
            if (self._loop is None or self._loop_frames == 0
                    or abs(bar_frames - self._loop_frames) > max(64, 0.006 * self._loop_frames)
                    or ib != self._loop_inten or style != self._loop_style):
                self._loop = self._synth_bar(bar_frames, style, inten)
                self._loop_frames = bar_frames
                self._loop_inten = ib
                self._loop_style = style
            loop = self._loop
            L = self._loop_frames
            t = deck.position_sec - an.beat_offset
            bar_phase = ((t / an.beat_period) / 4.0) % 1.0
            if bar_phase < 0:
                bar_phase += 1.0
            start = int(bar_phase * L) % L
            idx = (start + np.arange(n)) % L
            return loop[idx] * self.level
        except Exception:  # noqa: BLE001 - never break the audio callback
            return np.zeros((n, CHANNELS), dtype=np.float32)
