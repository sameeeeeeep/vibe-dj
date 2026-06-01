"""Autopilot: read the crowd, pick what plays next, and fire transitions.

The set is steered by a single rule — keep momentum when the room is hot, ease
down toward the crowd when it's cooling — while preferring tempo-adjacent tracks
so the beatmatched blends stay clean.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from typing import Callable, Optional

from .crowd import CrowdSensor
from .library import Library, Track
from .mixer import Mixer


def target_energy(crowd: float, current: float, max_step: float = 0.25) -> float:
    """Move the energy toward the crowd, but cap the jump so the set ramps
    instead of whiplashing. A hot room escalates; a cooling room eases down."""
    delta = max(-max_step, min(max_step, crowd - current))
    return min(1.0, max(0.0, current + delta))


# Re-pick tuning: a staged (cued) track is only swapped for a fresh pick when
# the room has drifted enough that the new pick beats it by a clear margin
# (avoids flapping between near-ties), and at most once per cooldown (caps how
# often we re-decode a track onto the idle deck). The cooldown is deliberately
# long: every swap decodes a whole track on the control thread, and a tight
# loop of re-cues was both churny *and* a source of audio hiccups, so we only
# re-cue when the room has really moved.
REPICK_MARGIN = 0.08
REPICK_COOLDOWN = 12.0

# Don't replay a track (by normalised title, so duplicate uploads count as the
# same song) until this many other tracks have played. Falls back gracefully on
# a small library: if every candidate is "recent", the filter is relaxed.
NO_REPEAT_WINDOW = 20


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

        # Tee every log line into a ring buffer so the dashboard can show a live
        # feed of what the autopilot just decided, without changing call sites.
        self._log_cb = log or (lambda m: None)
        self.events: deque[tuple[float, str]] = deque(maxlen=300)
        self._evt_lock = threading.Lock()
        self.log = self._log

        self.deck_tracks: dict[str, Optional[Track]] = {"A": None, "B": None}
        self._was_transitioning = False
        self._last_repick = 0.0
        self._stop = threading.Event()

        # Manual controls (driven by the dashboard). A nonzero bias shifts the
        # crowd reading the DJ steers toward; a requested crossfade fires the
        # next transition on the operator's command instead of at track-end.
        self.energy_bias = 0.0
        self.skip_crossfade_sec = 4.0
        self._cmd_lock = threading.Lock()
        self._requested_xf: Optional[float] = None
        self._requested_cue = False

        # Transition style. "auto" lets the autopilot choose a style to fit each
        # energy move; any specific style name (see mixer.TRANSITIONS) forces
        # every transition to use it. The rotor keeps auto picks from leaning on
        # one move when several fit.
        self.transition_kind = "auto"
        self._kind_rotor = 0
        self._last_auto_kind = ""
        # Manual crowd override: when set, the DJ is dictating the room's vibe
        # and the autopilot ignores the sensor. None = follow the live sensor.
        self._crowd_override: Optional[float] = None

        # DJ manual queue: an ordered list of tracks the operator has lined up.
        # When non-empty it overrides the autopilot's energy pick — the head is
        # staged onto the idle deck next, in order. Empty = pure autopilot. The
        # _idle_queued flag remembers that the staged track was a DJ pick so the
        # autopilot's re-pick won't quietly swap it back out.
        self.queue: list[Track] = []
        self._queue_lock = threading.Lock()
        self._idle_queued = False

        # Rolling memory of recently-played (normalised) titles so the autopilot
        # doesn't replay the same song — or a duplicate upload of it — too soon.
        # Touched only on the control thread (tick), so no lock needed.
        self._recent_titles: deque[str] = deque(maxlen=NO_REPEAT_WINDOW)

    # ---- event log -------------------------------------------------------
    def _log(self, message: str) -> None:
        with self._evt_lock:
            self.events.append((time.time(), message))
        self._log_cb(message)

    def recent_events(self, n: int = 14) -> list[tuple[float, str]]:
        with self._evt_lock:
            return list(self.events)[-n:]

    # ---- selection -------------------------------------------------------
    def _score(self, t: Track, target: float, live_bpm: float) -> float:
        """How good a next-track candidate is: close to the target energy,
        tempo-adjacent for a clean beatmatch, and not played too recently."""
        score = -abs(t.energy - target)
        if live_bpm > 0:
            score -= 0.5 * abs(t.bpm - live_bpm) / live_bpm
        score -= 0.05 * t.play_count
        return score

    @staticmethod
    def _norm_title(name: str) -> str:
        """Collapse a filename/title to a song identity for repeat-avoidance:
        drop the trailing ``[videoid]`` tag, lowercase, squeeze whitespace. Two
        different uploads of the same song normalise to the same key."""
        s = re.sub(r"\s*\[[^\]]*\]\s*", " ", name or "")
        return re.sub(r"\s+", " ", s).strip().lower()

    def _pick(self, target: float, live_bpm: float, exclude: set[int]) -> Optional[Track]:
        recent = set(self._recent_titles)
        # Prefer a track we haven't played lately; keep the best "recent" one only
        # as a fallback for when everything fresh is excluded (small library).
        best = best_recent = None
        best_score = best_recent_score = -1e9
        for t in self.library.tracks:
            if id(t) in exclude:
                continue
            score = self._score(t, target, live_bpm)
            if score > best_recent_score:
                best_recent, best_recent_score = t, score
            if self._norm_title(t.name) in recent:
                continue
            if score > best_score:
                best, best_score = t, score
        return best if best is not None else best_recent

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

    def request_cue(self) -> None:
        """Ask the autopilot to load the next track onto the idle deck now
        (preview the upcoming mix) without firing the crossfade."""
        with self._cmd_lock:
            self._requested_cue = True

    def set_transition_kind(self, kind: str) -> None:
        """Set the transition style the autopilot uses: 'auto' (pick one to fit
        the energy move) or a specific style every mix will use."""
        from .mixer import TRANSITIONS
        if kind == "auto" or kind in TRANSITIONS:
            self.transition_kind = kind

    def _choose_kind(self, incoming: Optional[Track]) -> str:
        """Pick a transition style. A forced style wins; otherwise choose by the
        energy move into the incoming track, rotating within each bucket so the
        set doesn't lean on a single move when several fit."""
        if self.transition_kind != "auto":
            return self.transition_kind
        live = self.deck_tracks[self.mixer.current]
        cur_e = live.energy if live else 0.5
        nxt_e = incoming.energy if incoming else cur_e
        delta = nxt_e - cur_e
        if delta > 0.14:
            bucket = ("cut", "echo")                     # lifting hard: punch it in
        elif delta < -0.14:
            bucket = ("filter", "brake")                 # cooling: wind it down
        else:
            bucket = ("smooth", "bass_swap", "filter")   # cruising: musical blends
        self._kind_rotor += 1
        kind = bucket[self._kind_rotor % len(bucket)]
        if kind == self._last_auto_kind and len(bucket) > 1:
            kind = bucket[(self._kind_rotor + 1) % len(bucket)]
        self._last_auto_kind = kind
        return kind

    def current_target(self) -> float:
        """Where the autopilot is steering energy right now, bias included."""
        live = self.deck_tracks[self.mixer.current]
        cur_energy = live.energy if live else 0.5
        return target_energy(self._biased_crowd(self.effective_crowd()), cur_energy)

    # ---- manual queue (dashboard) ---------------------------------------
    # Tracks are addressed by id() (sent to the browser as a string) so the UI
    # never has to deal with names that aren't unique. Everything compares by
    # identity, never ==, because Track.__eq__ would drag the numpy-backed
    # Analysis into the comparison.
    def _resolve_tid(self, tid: int) -> Optional[Track]:
        for t in self.library.tracks:
            if id(t) == tid:
                return t
        return None

    def queue_add(self, tid: int) -> None:
        t = self._resolve_tid(tid)
        if t is None:
            return
        with self._queue_lock:
            if any(x is t for x in self.queue):
                return
            self.queue.append(t)
            pos = len(self.queue)
        self.log(f"[queue] +{t.name[:34]}  (#{pos})")

    def queue_remove(self, tid: int) -> None:
        removed = None
        with self._queue_lock:
            for i, x in enumerate(self.queue):
                if id(x) == tid:
                    removed = self.queue.pop(i)
                    break
        if removed is not None:
            self.log(f"[queue] -{removed.name[:34]}")

    def queue_move(self, tid: int, delta: float) -> None:
        """Nudge a queued track one slot toward the front (delta<0) or back."""
        step = 1 if delta > 0 else -1
        with self._queue_lock:
            ids = [id(x) for x in self.queue]
            if tid not in ids:
                return
            i = ids.index(tid)
            j = max(0, min(len(self.queue) - 1, i + step))
            if i != j:
                self.queue.insert(j, self.queue.pop(i))

    def queue_clear(self) -> None:
        with self._queue_lock:
            self.queue.clear()

    def queue_view(self) -> list[Track]:
        with self._queue_lock:
            return list(self.queue)

    def _peek_queue(self, loaded_ids: set[int]) -> Optional[Track]:
        """The next queued track ready to stage: the first entry not already on
        a deck. Prunes entries that have left the library (e.g. a streaming pool
        deleted the file) so the queue can't wedge on a dead reference."""
        with self._queue_lock:
            lib_ids = {id(t) for t in self.library.tracks}
            self.queue = [t for t in self.queue if id(t) in lib_ids]
            for t in self.queue:
                if id(t) not in loaded_ids:
                    return t
            return None

    def _pop_queue(self, track: Track) -> None:
        with self._queue_lock:
            for i, x in enumerate(self.queue):
                if x is track:
                    self.queue.pop(i)
                    return

    def _restage(self, idle: str, cued: Track, new: Track) -> None:
        """Swap the staged idle-deck track for `new`: decode it first, load it
        onto the (silent) idle deck so there's no gap, then un-stage the old."""
        samples = self.library.load_audio(new)
        self.mixer.load_idle(samples, new.analysis, new.name)
        self.deck_tracks[idle] = new
        new.play_count += 1
        new.last_played_at = time.monotonic()
        self._unstage(cued)

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
        # The DJ's queue wins: stage its head if there is one, else let the
        # autopilot pick by energy/tempo.
        queued = self._peek_queue(self._loaded_ids())
        nxt = queued or self._pick(target, self.mixer.live_deck.effective_bpm, self._loaded_ids())
        if nxt is None:
            return
        samples = self.library.load_audio(nxt)
        self.mixer.load_idle(samples, nxt.analysis, nxt.name)
        self.deck_tracks[self.mixer.idle_name] = nxt
        nxt.play_count += 1
        nxt.last_played_at = time.monotonic()
        if queued is not None:
            self._pop_queue(nxt)
            self._idle_queued = True
            self.log(f"[cue]   {nxt.name}  {nxt.bpm:.0f} BPM  energy {nxt.energy:.2f}  (from queue)")
        else:
            self._idle_queued = False
            self.log(
                f"[cue]   {nxt.name}  {nxt.bpm:.0f} BPM  energy {nxt.energy:.2f}  "
                f"(target {target:.2f}, crowd {crowd:.2f})"
            )

    def _unstage(self, track: Track) -> None:
        """Undo the cue-time bookkeeping for a staged track that never played
        (it's being swapped out), and release it back to the library/pool."""
        track.play_count = max(0, track.play_count - 1)
        self.library.release(track)

    def _maybe_repick(self, crowd: float) -> None:
        """The idle deck is staged but the mix is still a way off. If the room
        has drifted so another track now beats the staged one by a clear margin,
        swap it in — keeps the on-deck pick matching where the crowd is heading.
        Rate-limited so we don't thrash the idle deck with re-decodes."""
        idle = self.mixer.idle_name
        cued = self.deck_tracks[idle]
        if cued is None:
            return
        # The DJ's queue overrides the autopilot. If its head isn't what's
        # staged, swap it in at once (the operator's order beats the energy pick
        # and skips the re-pick cooldown).
        head = self._peek_queue(self._loaded_ids())
        if head is not None:
            if head is cued:
                return
            self._restage(idle, cued, head)
            self._pop_queue(head)
            self._idle_queued = True
            self.log(f"[swap]  queued {head.name}  {head.bpm:.0f} BPM  energy {head.energy:.2f}")
            return
        # A DJ-chosen track is staged and the queue is now empty — leave it; the
        # operator picked it deliberately, so the autopilot shouldn't override it.
        if self._idle_queued:
            return
        now = time.monotonic()
        if now - self._last_repick < REPICK_COOLDOWN:
            return
        live_track = self.deck_tracks[self.mixer.current]
        cur_energy = live_track.energy if live_track else 0.5
        target = target_energy(self._biased_crowd(crowd), cur_energy)
        live_bpm = self.mixer.live_deck.effective_bpm
        # Candidates are everything but the live track; the staged track is
        # allowed to win again (then it's a no-op).
        exclude = {id(live_track)} if live_track is not None else set()
        best = self._pick(target, live_bpm, exclude)
        if best is None or best is cued:
            return
        if self._score(best, target, live_bpm) - self._score(cued, target, live_bpm) <= REPICK_MARGIN:
            return
        # Decode the better pick first, then swap it onto the (silent) idle deck
        # so there's no empty gap, and un-stage the one it replaces.
        self._last_repick = now
        self._restage(idle, cued, best)
        self.log(
            f"[swap]  re-cue {best.name}  {best.bpm:.0f} BPM  energy {best.energy:.2f}  "
            f"(was {cued.energy:.2f}, target {target:.2f}, crowd {crowd:.2f})"
        )

    def _on_transition_done(self) -> None:
        # The deck that just faded out is now free; release its track so a
        # streaming pool can delete the file and pull the next one.
        freed = self.mixer.idle_name
        old = self.deck_tracks[freed]
        self.deck_tracks[freed] = None
        # The just-promoted track is live now; the freed deck is empty, so any
        # "this staged track was a DJ pick" flag no longer applies.
        self._idle_queued = False
        if old is not None:
            # Remember it just played so the autopilot won't pick it (or a
            # duplicate upload of it) again until the no-repeat window rolls over.
            self._recent_titles.append(self._norm_title(old.name))
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

        # Feed the drum layer its "vibe" so (in AUTO mode) it thickens up on
        # bangers and stays minimal on mellow cuts — mostly the live song's own
        # energy, nudged a touch toward where the room is heading.
        live_track = self.deck_tracks.get(self.mixer.current)
        if live_track is not None:
            self.mixer.beats.set_vibe(0.7 * live_track.energy + 0.3 * crowd)

        # Operator-requested transition (skip / force) takes priority. Cue a
        # track first if the idle deck is empty, then fire immediately.
        with self._cmd_lock:
            requested_xf = self._requested_xf
            self._requested_xf = None
            requested_cue = self._requested_cue
            self._requested_cue = False
        # A bare cue request (no mix) just stages the next track on the idle deck.
        if requested_cue and not transitioning \
                and self.deck_tracks[self.mixer.idle_name] is None:
            self._cue_next(crowd)
        if requested_xf is not None and not transitioning:
            if self.deck_tracks[self.mixer.idle_name] is None:
                self._cue_next(crowd)
            if self.deck_tracks[self.mixer.idle_name] is not None:
                kind = self._choose_kind(self.deck_tracks[self.mixer.idle_name])
                self.log(f"[mix]   manual {kind} transition over {requested_xf:.0f}s")
                self.mixer.start_transition(requested_xf, kind)
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
        # Honour the DJ's hand-set mix-out marker (Deck.eff_mix_out) over the
        # analysis default, so the crossfade fires where the operator aligned it.
        mix_out = self.mixer.live_deck.eff_mix_out() if an else None
        have_outro = an is not None and mix_out is not None and mix_out < an.duration - 0.5

        if not transitioning:
            if not idle_loaded:
                # Eager pre-load: the moment the idle deck is free, stage the
                # next track so a deck is always cued and ready (like a DJ
                # loading the next tune right after a mix).
                self._cue_next(crowd)
                idle_loaded = self.deck_tracks[self.mixer.idle_name] is not None
            else:
                # Staged, but the mix is still a way off: let the pick track the
                # crowd. Once inside the cue-lead window we lock it so the deck
                # is stable and ready when the crossfade fires.
                approaching = (rem <= self.cue_lead_sec
                               or (have_outro and pos >= mix_out - self.cue_lead_sec))
                if not approaching:
                    self._maybe_repick(crowd)

        if not transitioning and idle_loaded:
            at_outro = have_outro and pos >= mix_out and rem >= self.crossfade_sec
            at_end = rem <= self.crossfade_sec + 0.5
            if at_outro or at_end:
                why = "outro" if at_outro and not at_end else "track end"
                kind = self._choose_kind(self.deck_tracks[self.mixer.idle_name])
                self.log(f"[mix]   {kind} over {self.crossfade_sec:.0f}s (on {why})")
                self.mixer.start_transition(self.crossfade_sec, kind)

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
