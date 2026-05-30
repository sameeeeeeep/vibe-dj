"""Two-deck mixer: equal-power, beat-aligned crossfades over a real or dummy output."""

from __future__ import annotations

import threading
import time

import numpy as np

from . import CHANNELS, SAMPLE_RATE
from .analysis import Analysis
from .deck import Deck

BLOCK = 1024
RATE_LIMIT = 0.08  # max +/- varispeed (~8%) so beatmatching doesn't sound chipmunky


class Mixer:
    def __init__(self, dry_run: bool = False):
        self.decks = {"A": Deck("A"), "B": Deck("B")}
        self.current = "A"
        self.dry_run = dry_run

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

    def load_idle(self, samples: np.ndarray, analysis: Analysis, title: str) -> None:
        self.idle_deck.load(samples, analysis, title)

    def start_first(self, samples: np.ndarray, analysis: Analysis, title: str) -> None:
        d = self.live_deck
        d.load(samples, analysis, title)
        d.pos = max(0.0, analysis.beat_offset * SAMPLE_RATE)
        d.gain = 1.0
        d.playing = True

    # ---- transitions -----------------------------------------------------
    def is_transitioning(self) -> bool:
        with self._xf_lock:
            return self._xf_active

    def start_transition(self, duration_sec: float) -> None:
        live = self.live_deck
        nxt = self.idle_deck
        if nxt.analysis is None or live.analysis is None:
            return

        target_bpm = live.effective_bpm
        rate = target_bpm / nxt.analysis.bpm if nxt.analysis.bpm > 0 else 1.0
        nxt.rate = float(np.clip(rate, 1 - RATE_LIMIT, 1 + RATE_LIMIT))

        # Align the incoming track's next beat to the outgoing track's next beat.
        src_period = nxt.analysis.beat_period
        from_next = live.seconds_to_next_beat()
        if src_period > 0:
            x = (src_period - (from_next * nxt.rate) % src_period) % src_period
            nxt.pos = max(0.0, (nxt.analysis.beat_offset + x) * SAMPLE_RATE)
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
            if p >= 1.0:
                frm.gain = 0.0
                frm.playing = False
                to.gain = 1.0
                self.current = self._xf_to
                self._xf_active = False

    # ---- audio generation ------------------------------------------------
    def mix(self, n: int) -> np.ndarray:
        self._advance_crossfade(n)
        out = np.zeros((n, CHANNELS), dtype=np.float32)
        for d in self.decks.values():
            g = d.gain
            if g <= 0.0 and not d.playing:
                continue
            out += d.read(n) * g
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
