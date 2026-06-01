"""Two-deck mixer: equal-power, beat-aligned crossfades over a real or dummy output."""

from __future__ import annotations

import threading
import time

import numpy as np

from . import CHANNELS, SAMPLE_RATE
from .analysis import Analysis
from .deck import Deck
from .eq import ThreeBandEQ

BLOCK = 1024
RATE_LIMIT = 0.08  # max +/- varispeed (~8%) so beatmatching doesn't sound chipmunky

# Where in a crossfade the basslines trade. Before the start only the outgoing
# deck owns the lows; after the end only the incoming deck does — so two
# basslines never play at once (the classic "bass swap" that keeps blends clean).
BASS_SWAP_START = 0.45
BASS_SWAP_END = 0.60


class Mixer:
    def __init__(self, dry_run: bool = False):
        self.decks = {"A": Deck("A"), "B": Deck("B")}
        self.current = "A"
        self.dry_run = dry_run

        # Per-deck 3-band EQ. Two sources drive the band gains and compose by
        # multiplication so they never fight each other:
        #   eq_manual — the DJ's EQ (low/mid/high, 1.0 = unity, 0.0 = killed)
        #   bass_auto — the bass-swap automation's low-band factor during a fade
        # Effective low gain = eq_manual.low * bass_auto; mid/high = eq_manual.
        self.eqs = {"A": ThreeBandEQ(), "B": ThreeBandEQ()}
        self.eq_manual = {"A": [1.0, 1.0, 1.0], "B": [1.0, 1.0, 1.0]}
        self.bass_auto = {"A": 1.0, "B": 1.0}

        # Per-channel volume trim (the DJ's line faders) and a master pause that
        # freezes both decks and mutes the output, resuming exactly in place.
        self.trim = {"A": 1.0, "B": 1.0}
        self.paused = False

        self._xf_lock = threading.Lock()
        self._xf_active = False
        self._xf_from = ""
        self._xf_to = ""
        self._xf_total = 1
        self._xf_done = 0

        self._stream = None
        self._thread = None
        self._stop = threading.Event()

    # ---- track placement -------------------------------------------------
    @property
    def live_deck(self) -> Deck:
        return self.decks[self.current]

    @property
    def idle_deck(self) -> Deck:
        return self.decks["B" if self.current == "A" else "A"]

    @property
    def idle_name(self) -> str:
        return "B" if self.current == "A" else "A"

    def _reset_eq(self, name: str) -> None:
        self.eqs[name].reset()
        self.eq_manual[name] = [1.0, 1.0, 1.0]
        self.bass_auto[name] = 1.0
        # Trim is a physical line fader: it persists across track loads.

    def load_idle(self, samples: np.ndarray, analysis: Analysis, title: str) -> None:
        self.idle_deck.load(samples, analysis, title)
        self._reset_eq(self.idle_name)

    def start_first(self, samples: np.ndarray, analysis: Analysis, title: str) -> None:
        d = self.live_deck
        d.load(samples, analysis, title)
        self._reset_eq(self.current)
        d.pos = max(0.0, analysis.beat_offset * SAMPLE_RATE)
        d.gain = 1.0
        d.playing = True

    # ---- transitions -----------------------------------------------------
    def is_transitioning(self) -> bool:
        with self._xf_lock:
            return self._xf_active

    def transition_state(self) -> dict:
        """Snapshot of the crossfade for observers (e.g. the dashboard)."""
        with self._xf_lock:
            return {
                "active": self._xf_active,
                "progress": min(1.0, self._xf_done / self._xf_total) if self._xf_active else 0.0,
                "from": self._xf_from if self._xf_active else "",
                "to": self._xf_to if self._xf_active else "",
            }

    def start_transition(self, duration_sec: float) -> None:
        live = self.live_deck
        nxt = self.idle_deck
        if nxt.analysis is None or live.analysis is None:
            return

        target_bpm = live.effective_bpm
        rate = target_bpm / nxt.analysis.bpm if nxt.analysis.bpm > 0 else 1.0
        nxt.rate = float(np.clip(rate, 1 - RATE_LIMIT, 1 + RATE_LIMIT))

        # Drop the incoming track in at its mix-in cue (past the intro, on a
        # phrase boundary) and align its next beat to the outgoing track's next
        # beat. The cue sits on a beat, so adding the sub-beat phase offset keeps
        # the blend beatmatched.
        src_period = nxt.analysis.beat_period
        from_next = live.seconds_to_next_beat()
        if src_period > 0:
            x = (src_period - (from_next * nxt.rate) % src_period) % src_period
            nxt.pos = max(0.0, (nxt.analysis.mix_in_sec() + x) * SAMPLE_RATE)
        nxt.gain = 0.0
        nxt.playing = True

        with self._xf_lock:
            self._xf_from = self.current
            self._xf_to = self.idle_name
            self._xf_total = max(1, int(duration_sec * SAMPLE_RATE))
            self._xf_done = 0
            self._xf_active = True

    def _advance_crossfade(self, n: int) -> None:
        with self._xf_lock:
            if not self._xf_active:
                return
            self._xf_done += n
            p = min(1.0, self._xf_done / self._xf_total)
            frm = self.decks[self._xf_from]
            to = self.decks[self._xf_to]
            frm.gain = float(np.cos(p * np.pi / 2))
            to.gain = float(np.sin(p * np.pi / 2))

            # Bass swap: trade the lows over the swap window so only one bassline
            # plays at a time. Mids/highs ride the equal-power volume fade above.
            # This drives only the auto factor; the DJ's manual EQ multiplies on
            # top in mix(), so a manual kill and the swap coexist.
            swap = (p - BASS_SWAP_START) / (BASS_SWAP_END - BASS_SWAP_START)
            swap = min(1.0, max(0.0, swap))
            self.bass_auto[self._xf_from] = 1.0 - swap
            self.bass_auto[self._xf_to] = swap

            if p >= 1.0:
                frm.gain = 0.0
                frm.playing = False
                to.gain = 1.0
                self.bass_auto[self._xf_from] = 1.0
                self.bass_auto[self._xf_to] = 1.0
                self.current = self._xf_to
                self._xf_active = False

    def effective_bands(self, name: str) -> list[float]:
        """The (low, mid, high) gains actually applied to a deck right now:
        the DJ's EQ with the bass-swap automation folded into the low band."""
        lo, mid, hi = self.eq_manual[name]
        return [lo * self.bass_auto[name], mid, hi]

    # ---- manual controls (dashboard) ------------------------------------
    _BAND_IX = {"low": 0, "mid": 1, "high": 2}

    def set_paused(self, on: bool) -> None:
        self.paused = bool(on)

    def toggle_pause(self) -> bool:
        self.paused = not self.paused
        return self.paused

    def set_eq(self, deck: str, band: str, gain: float) -> None:
        i = self._BAND_IX.get(band)
        if i is None or deck not in self.eq_manual:
            return
        self.eq_manual[deck][i] = float(min(1.5, max(0.0, gain)))

    def toggle_kill(self, deck: str, band: str) -> None:
        i = self._BAND_IX.get(band)
        if i is None or deck not in self.eq_manual:
            return
        self.eq_manual[deck][i] = 0.0 if self.eq_manual[deck][i] > 0.0 else 1.0

    def set_trim(self, deck: str, value: float) -> None:
        if deck in self.trim:
            self.trim[deck] = float(min(1.0, max(0.0, value)))

    def set_bend(self, deck: str, frac: float) -> None:
        d = self.decks.get(deck)
        if d is not None:
            d.bend = float(min(RATE_LIMIT, max(-RATE_LIMIT, frac)))

    def scrub_crossfade(self, frac: float) -> None:
        """Push/pull a live transition by hand (no-op when not transitioning)."""
        with self._xf_lock:
            if self._xf_active:
                self._xf_done = int(min(1.0, max(0.0, frac)) * self._xf_total)

    # ---- audio generation ------------------------------------------------
    def mix(self, n: int) -> np.ndarray:
        if self.paused:
            # Frozen: no deck advance, silent output. Resumes exactly in place.
            return np.zeros((n, CHANNELS), dtype=np.float32)
        self._advance_crossfade(n)
        out = np.zeros((n, CHANNELS), dtype=np.float32)
        for name, d in self.decks.items():
            g = d.gain
            if g <= 0.0 and not d.playing:
                continue
            lg, mg, hg = self.effective_bands(name)
            out += self.eqs[name].process(d.read(n), lg, mg, hg) * (g * self.trim[name])
        np.clip(out, -1.0, 1.0, out=out)
        return out

    # ---- backends --------------------------------------------------------
    def start(self) -> None:
        if self.dry_run:
            self._thread = threading.Thread(target=self._dummy_loop, daemon=True)
            self._thread.start()
            return
        import sounddevice as sd

        def callback(outdata, frames, time_info, status):
            outdata[:] = self.mix(frames)

        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS,
            dtype="float32", blocksize=BLOCK, callback=callback,
        )
        self._stream.start()

    def _dummy_loop(self) -> None:
        period = BLOCK / SAMPLE_RATE
        nxt = time.monotonic()
        while not self._stop.is_set():
            self.mix(BLOCK)
            nxt += period
            time.sleep(max(0, nxt - time.monotonic()))

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
