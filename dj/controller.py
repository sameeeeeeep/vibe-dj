"""Autopilot: read the crowd, pick what plays next, and fire transitions.

The set is steered by a single rule — keep momentum when the room is hot, ease
down toward the crowd when it's cooling — while preferring tempo-adjacent tracks
so the beatmatched blends stay clean.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from .crowd import CrowdSensor
from .library import Library, Track
from .mixer import Mixer


def target_energy(crowd: float, current: float, max_step: float = 0.25) -> float:
    """Move the energy toward the crowd, but cap the jump so the set ramps
    instead of whiplashing. A hot room escalates; a cooling room eases down."""
    delta = max(-max_step, min(max_step, crowd - current))
    return min(1.0, max(0.0, current + delta))


class Controller:
    def __init__(
        self,
        library: Library,
        mixer: Mixer,
        crowd: CrowdSensor,
        crossfade_sec: float = 12.0,
        cue_lead_sec: float = 25.0,
        log: Optional[Callable[[str], None]] = None,
    ):
        self.library = library
        self.mixer = mixer
        self.crowd = crowd
        self.crossfade_sec = crossfade_sec
        self.cue_lead_sec = cue_lead_sec
        self.log = log or (lambda m: None)

        self.deck_tracks: dict[str, Optional[Track]] = {"A": None, "B": None}
        self._was_transitioning = False
        self._stop = threading.Event()

        # Manual controls (driven by the dashboard). A nonzero bias shifts the
        # crowd reading the DJ steers toward; a requested crossfade fires the
        # next transition on the operator's command instead of at track-end.
        self.energy_bias = 0.0
        self.skip_crossfade_sec = 4.0
        self._cmd_lock = threading.Lock()
        self._requested_xf: Optional[float] = None
        # Manual crowd override: when set, the DJ is dictating the room's vibe
        # and the autopilot ignores the sensor. None = follow the live sensor.
        self._crowd_override: Optional[float] = None

    # ---- selection -------------------------------------------------------
    def _pick(self, target: float, live_bpm: float, exclude: set[int]) -> Optional[Track]:
        best, best_score = None, -1e9
        for t in self.library.tracks:
            if id(t) in exclude:
                continue
            score = -abs(t.energy - target)
            if live_bpm > 0:
                score -= 0.5 * abs(t.bpm - live_bpm) / live_bpm
            score -= 0.05 * t.play_count
            if score > best_score:
                best, best_score = t, score
        return best

    def _loaded_ids(self) -> set[int]:
        return {id(t) for t in self.deck_tracks.values() if t is not None}

    def _biased_crowd(self, crowd: float) -> float:
        return min(1.0, max(0.0, crowd + self.energy_bias))

    def effective_crowd(self) -> float:
        """The crowd reading the autopilot acts on: the DJ's manual vibe when
        the override is engaged, otherwise the live sensor."""
        return self.crowd.energy if self._crowd_override is None else self._crowd_override

    @property
    def crowd_manual(self) -> bool:
        return self._crowd_override is not None

    # ---- manual controls (dashboard) ------------------------------------
    def nudge_energy(self, delta: float) -> None:
        self.energy_bias = float(min(0.5, max(-0.5, self.energy_bias + delta)))

    def set_crowd_manual(self, on: bool) -> None:
        """Engage/release manual crowd control. Engaging seeds the manual vibe
        with the current sensor reading so the set doesn't jump on takeover."""
        if on:
            if self._crowd_override is None:
                self._crowd_override = self.crowd.energy
        else:
            self._crowd_override = None

    def set_crowd_energy(self, value: float) -> None:
        """Pin the crowd vibe to a DJ-chosen 0..1 level (engages manual mode)."""
        self._crowd_override = float(min(1.0, max(0.0, value)))

    def request_transition(self, duration: Optional[float] = None) -> None:
        """Ask for the next crossfade to fire on the next tick. `None` uses the
        configured crossfade length; a shorter value gives a quick skip."""
        with self._cmd_lock:
            self._requested_xf = self.crossfade_sec if duration is None else duration

    def request_skip(self) -> None:
        self.request_transition(self.skip_crossfade_sec)

    def current_target(self) -> float:
        """Where the autopilot is steering energy right now, bias included."""
        live = self.deck_tracks[self.mixer.current]
        cur_energy = live.energy if live else 0.5
        return target_energy(self._biased_crowd(self.effective_crowd()), cur_energy)

    # ---- lifecycle -------------------------------------------------------
    def start_set(self) -> None:
        first = min(self.library.tracks, key=lambda t: abs(t.energy - 0.5))
        samples = self.library.load_audio(first)
        self.mixer.start_first(samples, first.analysis, first.name)
        self.deck_tracks[self.mixer.current] = first
        first.play_count += 1
        first.last_played_at = time.monotonic()
        self.log(f"[start] {first.name}  {first.bpm:.0f} BPM  energy {first.energy:.2f}")

    def _cue_next(self, crowd: float) -> None:
        live_track = self.deck_tracks[self.mixer.current]
        cur_energy = live_track.energy if live_track else 0.5
        target = target_energy(self._biased_crowd(crowd), cur_energy)
        nxt = self._pick(target, self.mixer.live_deck.effective_bpm, self._loaded_ids())
        if nxt is None:
            return
        samples = self.library.load_audio(nxt)
        self.mixer.load_idle(samples, nxt.analysis, nxt.name)
        self.deck_tracks[self.mixer.idle_name] = nxt
        nxt.play_count += 1
        nxt.last_played_at = time.monotonic()
        self.log(
            f"[cue]   {nxt.name}  {nxt.bpm:.0f} BPM  energy {nxt.energy:.2f}  "
            f"(target {target:.2f}, crowd {crowd:.2f})"
        )

    def _on_transition_done(self) -> None:
        # The deck that just faded out is now free; release its track so a
        # streaming pool can delete the file and pull the next one.
        freed = self.mixer.idle_name
        old = self.deck_tracks[freed]
        self.deck_tracks[freed] = None
        if old is not None:
            self.library.release(old)
        live = self.deck_tracks[self.mixer.current]
        if live:
            self.log(f"[live]  {live.name}  now playing")

    def tick(self) -> None:
        crowd = self.effective_crowd()
        transitioning = self.mixer.is_transitioning()
        if self._was_transitioning and not transitioning:
            self._on_transition_done()
        self._was_transitioning = transitioning

        # Operator-requested transition (skip / force) takes priority. Cue a
        # track first if the idle deck is empty, then fire immediately.
        with self._cmd_lock:
            requested_xf = self._requested_xf
            self._requested_xf = None
        if requested_xf is not None and not transitioning:
            if self.deck_tracks[self.mixer.idle_name] is None:
                self._cue_next(crowd)
            if self.deck_tracks[self.mixer.idle_name] is not None:
                self.log(f"[mix]   manual transition over {requested_xf:.0f}s")
                self.mixer.start_transition(requested_xf)
            return

        live = self.mixer.live_deck
        rem = live.remaining_sec
        pos = live.position_sec
        idle_loaded = self.deck_tracks[self.mixer.idle_name] is not None

        # Prefer mixing out at the live track's outro (phrase-aligned) over
        # waiting for the file to end. A real cue sits before the natural end;
        # otherwise mix_out is the duration and only the end fallback fires.
        live_track = self.deck_tracks[self.mixer.current]
        an = live_track.analysis if live_track else None
        mix_out = an.mix_out_sec() if an else None
        have_outro = an is not None and mix_out is not None and mix_out < an.duration - 0.5

        if not transitioning and not idle_loaded:
            if rem < self.cue_lead_sec or (have_outro and pos >= mix_out - self.cue_lead_sec):
                self._cue_next(crowd)
                idle_loaded = self.deck_tracks[self.mixer.idle_name] is not None

        if not transitioning and idle_loaded:
            at_outro = have_outro and pos >= mix_out and rem >= self.crossfade_sec
            at_end = rem <= self.crossfade_sec + 0.5
            if at_outro or at_end:
                why = "outro" if at_outro and not at_end else "track end"
                self.log(f"[mix]   crossfading over {self.crossfade_sec:.0f}s (on {why})")
                self.mixer.start_transition(self.crossfade_sec)

    def run(self, status_every: float = 4.0) -> None:
        last_status = 0.0
        while not self._stop.is_set():
            self.tick()
            now = time.monotonic()
            if now - last_status >= status_every:
                last_status = now
                live = self.deck_tracks[self.mixer.current]
                name = live.name if live else "-"
                self.log(
                    f"[stat]  crowd {self.crowd.energy:.2f}  playing {name}  "
                    f"{self.mixer.live_deck.remaining_sec:5.1f}s left  "
                    f"{self.mixer.live_deck.effective_bpm:.1f} BPM"
                )
            time.sleep(0.5)

    def stop(self) -> None:
        self._stop.set()
