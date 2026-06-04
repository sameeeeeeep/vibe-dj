"""Two-deck mixer: equal-power, beat-aligned crossfades over a real or dummy output."""

from __future__ import annotations

import threading
import time

import numpy as np

from . import CHANNELS, SAMPLE_RATE
from .analysis import Analysis
from .beats import BeatMachine
from .deck import Deck
from .fx import FXRack
from .midi import MelodyLayer
from .eq import ThreeBandEQ

BLOCK = 2048       # ~46ms/callback: more headroom so a busy control thread
                   # (track decode, dashboard serialisation) can't starve the
                   # audio callback into an underrun/glitch. Still low-latency
                   # enough for hands-on EQ/crossfader.
RATE_LIMIT = 0.08  # max +/- varispeed (~8%) so beatmatching doesn't sound chipmunky

# Where in a crossfade the basslines trade. Before the start only the outgoing
# deck owns the lows; after the end only the incoming deck does — so two
# basslines never play at once (the classic "bass swap" that keeps blends clean).
BASS_SWAP_START = 0.45
BASS_SWAP_END = 0.60

BEATS_PER_BAR = 4   # 4/4 — used to size audition pre/post-roll in bars

# The transition styles the engine can run, in the order the dashboard lists
# them. Each is a musically distinct way to hand the room from the outgoing
# track to the incoming one (see _advance_crossfade for the automation):
#   smooth    — equal-power blend + bass swap (the clean default)
#   bass_swap — both grooves coexist at volume; basslines trade, outgoing drops late
#   filter    — sweep the outgoing up through a high-pass / open the incoming up
#   cut       — hard downbeat cut (a few ms, just click-free)
#   echo      — throw the outgoing into a beat-echo and pull it down under the tail
#   brake     — tape-stop the outgoing to a halt, then slam the incoming in
#   morph     — stutter-loop the outgoing's last beat with a shrinking window +
#               rising pitch (the "ee-ee-eeeh-EHH" build), then SLAM the incoming
TRANSITIONS = ("smooth", "bass_swap", "filter", "cut", "echo", "brake", "morph")

# Per-style length (seconds). None = use the caller's crossfade length (the long
# musical blends); the punchy styles pin their own short length regardless.
_KIND_SECONDS = {
    "smooth": None, "bass_swap": None, "filter": None,
    "cut": 0.035, "echo": 6.0, "brake": 3.0, "morph": 3.2,
}


class _FeedbackDelay:
    """A stereo feedback delay for the 'echo' transition throw.

    Vectorised block-wise: the delay time is one beat, always far longer than an
    audio block, so a block never reads samples it is writing in the same call —
    no per-sample Python loop runs in the audio callback."""

    def __init__(self, max_sec: float = 3.0):
        self._buf = np.zeros((int(max_sec * SAMPLE_RATE), CHANNELS), dtype=np.float32)
        self._w = 0

    def reset(self) -> None:
        self._buf[:] = 0.0
        self._w = 0

    def process(self, x: np.ndarray, delay_frames: int, feedback: float) -> np.ndarray:
        n = len(x)
        L = len(self._buf)
        d = int(max(n, min(L - n - 1, delay_frames)))   # >= block so no aliasing
        w = self._w
        wet = self._buf[(w - d + np.arange(n)) % L]      # fancy-index = fresh copy
        self._buf[(w + np.arange(n)) % L] = x + feedback * wet
        self._w = (w + n) % L
        return wet


class _MorphRiser:
    """The 'morph' transition's riser voice — the "ee-ee-eeeh-EHH" build.

    Captures one beat of the OUTGOING track and stutter-loops it with a window
    that shrinks in musical steps (1 -> 1/2 -> 1/4 -> 1/8 beat) while the pitch
    sweeps up (resampling faster). Shorter window = faster retrigger, so the
    stutter accelerates into the slam. Vectorised per block — the loop wraps with
    a modulo, no per-sample Python loop runs in the audio callback."""

    def __init__(self):
        self._slice = np.zeros((0, CHANNELS), dtype=np.float32)
        self._beat = 1       # one captured beat, in frames (the base window)
        self._pos = 0.0      # fractional read position, wrapped into the window

    def load(self, slice_samples: np.ndarray, beat_frames: int) -> None:
        self._slice = np.ascontiguousarray(slice_samples, dtype=np.float32)
        self._beat = max(2, int(beat_frames))
        self._pos = 0.0

    def process(self, n: int, p: float) -> np.ndarray:
        out = np.zeros((n, CHANNELS), dtype=np.float32)
        L = len(self._slice)
        if L < 2:
            return out
        # Window shrinks in discrete musical steps as the build rises.
        if p < 0.30:
            frac = 1.0
        elif p < 0.55:
            frac = 0.5
        elif p < 0.78:
            frac = 0.25
        else:
            frac = 0.125
        win = max(2, min(L, int(self._beat * frac)))
        rate = 1.0 + 1.0 * min(1.0, p)        # pitch sweeps up ~1.0 -> 2.0
        start = self._pos % win               # re-anchor when the window changes
        positions = (start + np.arange(n) * rate) % win
        idx0 = np.floor(positions).astype(np.int64)
        frac_part = (positions - idx0).astype(np.float32)[:, None]
        i0 = idx0 % win
        i1 = (idx0 + 1) % win
        out[:] = self._slice[i0] * (1.0 - frac_part) + self._slice[i1] * frac_part
        self._pos = start + n * rate
        return out


class Mixer:
    def __init__(self, dry_run: bool = False, cue_device=None, master_device=None):
        self.decks = {"A": Deck("A"), "B": Deck("B")}
        self.current = "A"
        self.dry_run = dry_run
        # Pin the master (room) output to a specific device instead of following
        # the system default. This matters for the AirPods workflow: to make
        # AirPods appear to CoreAudio you select them as the Mac's output, which
        # makes them the default — without this pin the room mix would jump onto
        # the AirPods too. None = system default (unchanged behaviour).
        self.master_device = master_device

        # Per-deck 3-band EQ. Two sources drive the band gains and compose by
        # multiplication so they never fight each other:
        #   eq_manual — the DJ's EQ (low/mid/high, 1.0 = unity, 0.0 = killed)
        #   bass_auto — the bass-swap automation's low-band factor during a fade
        # Effective low gain = eq_manual.low * bass_auto; mid/high = eq_manual.
        self.eqs = {"A": ThreeBandEQ(), "B": ThreeBandEQ()}
        self.eq_manual = {"A": [1.0, 1.0, 1.0], "B": [1.0, 1.0, 1.0]}
        # Auto-EQ a transition drives on TOP of the DJ's manual EQ (per band,
        # 1.0 = untouched). The bass swap writes the low band; the filter style
        # sweeps all three. Composes by multiplication in effective_bands so it
        # never fights the DJ's hand EQ. (Replaces the old low-only bass_auto.)
        self.eq_auto = {"A": [1.0, 1.0, 1.0], "B": [1.0, 1.0, 1.0]}

        # Per-channel volume trim (the DJ's line faders) and a master pause that
        # freezes both decks and mutes the output, resuming exactly in place.
        self.trim = {"A": 1.0, "B": 1.0}
        self.paused = False

        # Procedural drum-loop layer — a "beats" pad summed into the master,
        # beat-locked to the live deck and tempo-matched, intensity following
        # the song's vibe. Off by default (the DJ drops it in).
        self.beats = BeatMachine()

        # Sound-FX rack — triggerable one-shot pads loaded from pasted links.
        # Summed flat into the master (not beat-locked); the DJ fires them.
        self.fx = FXRack()

        # MIDI melody layer — a loaded .mid played through synth voices, looped
        # beat-locked + tempo-matched to the live deck. Off until a file loads.
        self.melody = MelodyLayer()

        self._xf_lock = threading.Lock()
        self._xf_active = False
        self._xf_from = ""
        self._xf_to = ""
        self._xf_total = 1
        self._xf_done = 0
        self._xf_type = "smooth"     # which transition style is running
        self._xf_secs = 0.0          # its length in seconds (dashboard readout)
        self._last_kind = "smooth"   # last style that fired
        self._echo = _FeedbackDelay(3.0)   # delay line for the 'echo' throw
        self._echo_delay = 1         # one-beat delay (frames) for this throw
        self._brake_rate = 1.0       # outgoing deck's rate captured for a brake
        self._morph = _MorphRiser()  # riser voice for the 'morph' build

        self._stream = None
        self._thread = None
        self._stop = threading.Event()

        # ---- headphone cue: a SECOND output device (e.g. AirPods) ----------
        # The master mix goes to the speakers; this monitor feed goes to a
        # separate device so the DJ can pre-listen to the incoming track and
        # audition the transition privately. It reads decks via read_preview()
        # on its OWN playheads, so it never disturbs what the crowd hears.
        self.cue_device = cue_device   # device index/name for the monitor, or None
        self.cue_enabled = False       # is the second stream actually open & running
        self.cue_level = 1.0           # monitor volume (0..1)
        self.cue_source = "cued"       # "cued" (incoming) | "live" | "master"
        self._cue_error = ""           # last device-open failure, surfaced to the UI
        self._cue_stream = None
        self._cue_lock = threading.Lock()
        self._cue_pos = 0.0            # preview playhead for pre-listening one deck
        self._cue_mon_deck = ""        # which deck that playhead is currently tracking
        self._last_master = None       # cached master block for the "master" monitor
        # Optional LAN broadcaster (dj/netcast.py). When set, every mixed block is
        # fed to it for MP3 streaming to other machines. None = not broadcasting.
        self.netcast = None
        # One-shot audition: render the real blend (outgoing@mix_out -> incoming
        # @mix_in over the crossfade) to the monitor only, on preview playheads.
        self._aud_active = False
        self._aud_from = ""
        self._aud_to = ""
        self._aud_from_pos = 0.0
        self._aud_to_pos = 0.0
        self._aud_to_rate = 1.0
        self._aud_done = 0
        self._aud_pre = 0              # outgoing-solo pre-roll (frames)
        self._aud_total = 1            # crossfade length (frames)
        self._aud_post = 0             # incoming-solo post-roll (frames)

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
        self.eq_auto[name] = [1.0, 1.0, 1.0]
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
                "kind": self._xf_type if self._xf_active else "",
                "secs": round(self._xf_secs, 2) if self._xf_active else 0.0,
                "last_kind": self._last_kind,
            }

    def start_transition(self, duration_sec: float, kind: str = "smooth") -> None:
        live = self.live_deck
        nxt = self.idle_deck
        if nxt.analysis is None or live.analysis is None:
            return
        if kind not in TRANSITIONS:
            kind = "smooth"
        secs = _KIND_SECONDS.get(kind)
        dur = duration_sec if secs is None else secs

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
            nxt.pos = max(0.0, (nxt.eff_mix_in() + x) * SAMPLE_RATE)
        nxt.gain = 0.0
        # A brake stops the room dead and a morph rides a riser up; both drop the
        # new track in only at the end — so hold the incoming frozen at its cue
        # (read() won't advance a non-playing deck) until the style releases it in
        # _advance_crossfade, where it starts cleanly.
        nxt.playing = kind not in ("brake", "morph")

        # Fresh auto-EQ on both decks so the previous style leaves nothing behind.
        self.eq_auto[self.current] = [1.0, 1.0, 1.0]
        self.eq_auto[self.idle_name] = [1.0, 1.0, 1.0]

        if kind == "echo":
            # One beat of the OUTGOING track, fed back on itself.
            self._echo.reset()
            bp = live.analysis.beat_period / max(1e-6, live.eff_rate)
            self._echo_delay = max(1, int(bp * SAMPLE_RATE))
        elif kind == "brake":
            self._brake_rate = max(0.1, live.rate)
        elif kind == "morph":
            # Grab the outgoing track's last beat (source frames) for the riser to
            # stutter-loop. One source beat == beat_period seconds of samples.
            beat_frames = max(2, int(live.analysis.beat_period * SAMPLE_RATE))
            self._morph.load(live.tail_slice(beat_frames), beat_frames)

        with self._xf_lock:
            self._xf_from = self.current
            self._xf_to = self.idle_name
            self._xf_total = max(1, int(dur * SAMPLE_RATE))
            self._xf_done = 0
            self._xf_type = kind
            self._xf_secs = dur
            self._last_kind = kind
            self._xf_active = True

    def _advance_crossfade(self, n: int) -> None:
        with self._xf_lock:
            if not self._xf_active:
                return
            self._xf_done += n
            p = min(1.0, self._xf_done / self._xf_total)
            frm = self.decks[self._xf_from]
            to = self.decks[self._xf_to]
            kind = self._xf_type

            if kind == "cut":
                # Hard downbeat cut: a few-ms equal-power swap, just enough to
                # dodge a click. No EQ moves.
                frm.gain = float(np.cos(p * np.pi / 2))
                to.gain = float(np.sin(p * np.pi / 2))

            elif kind == "filter":
                # Filter transition. Volumes stay near full so the *filter* does
                # the handoff: sweep the outgoing UP through a high-pass (lows
                # gone by ~0.5, mids by ~0.85, thin top fades out) while the
                # incoming opens UP from a low-pass (lows first, mids/highs bloom).
                frm.gain = float(min(1.0, 2.0 * (1.0 - p)))
                to.gain = float(min(1.0, 2.0 * p))
                self.eq_auto[self._xf_from] = [
                    max(0.0, 1.0 - p / 0.5),
                    max(0.0, 1.0 - max(0.0, p - 0.4) / 0.45),
                    1.0,
                ]
                self.eq_auto[self._xf_to] = [
                    1.0,
                    min(1.0, p / 0.6),
                    min(1.0, max(0.0, p - 0.4) / 0.45),
                ]

            elif kind == "echo":
                # Outgoing is thrown into a feedback echo (rung out in mix()) and
                # pulled down fast; the incoming rises under the tail.
                frm.gain = float(max(0.0, 1.0 - p / 0.35))
                to.gain = float(np.sin(min(1.0, p / 0.85) * np.pi / 2))
                sw = min(1.0, max(0.0, (p - 0.2) / 0.25))
                self.eq_auto[self._xf_from][0] = 1.0 - sw
                self.eq_auto[self._xf_to][0] = sw

            elif kind == "brake":
                # Tape-stop: brake the outgoing to a halt over the first ~55% (its
                # pitch sags as the platter stops), a beat of near-silence, then
                # the incoming slams in for the back third.
                b = min(1.0, p / 0.55)
                frm.rate = float(max(0.0, self._brake_rate * (1.0 - b)))
                frm.gain = float(max(0.0, 1.0 - b))
                if p >= 0.62 and not to.playing:
                    to.playing = True            # release the frozen incoming
                tp = min(1.0, max(0.0, (p - 0.62) / 0.30))
                to.gain = float(np.sin(tp * np.pi / 2))

            elif kind == "morph":
                # The riser (built in mix() from the captured beat) IS the
                # outgoing voice, so mute the deck's own read. Near the top of the
                # build the incoming is released from its frozen cue and slams in.
                frm.gain = 0.0
                if p >= 0.86 and not to.playing:
                    to.playing = True
                tp = min(1.0, max(0.0, (p - 0.86) / 0.14))
                to.gain = float(np.sin(tp * np.pi / 2))

            else:   # "smooth" and "bass_swap"
                frm.gain = float(np.cos(p * np.pi / 2))
                to.gain = float(np.sin(p * np.pi / 2))
                if kind == "bass_swap":
                    # Both grooves coexist: incoming up quickly (bass killed),
                    # outgoing held, basslines trade, outgoing falls away late.
                    frm.gain = float(min(1.0, 1.6 * (1.0 - p)))
                    to.gain = float(min(1.0, 1.6 * p))
                # Trade the lows over the swap window so only one bassline plays
                # at a time. Drives only the auto factor; the DJ's manual EQ
                # multiplies on top in mix(), so a manual kill and the swap coexist.
                swap = (p - BASS_SWAP_START) / (BASS_SWAP_END - BASS_SWAP_START)
                swap = min(1.0, max(0.0, swap))
                self.eq_auto[self._xf_from][0] = 1.0 - swap
                self.eq_auto[self._xf_to][0] = swap

            if p >= 1.0:
                frm.gain = 0.0
                frm.playing = False
                frm.rate = 1.0                   # release any brake
                to.gain = 1.0
                to.playing = True
                self.eq_auto[self._xf_from] = [1.0, 1.0, 1.0]
                self.eq_auto[self._xf_to] = [1.0, 1.0, 1.0]
                self.current = self._xf_to
                self._xf_active = False

    def effective_bands(self, name: str) -> list[float]:
        """The (low, mid, high) gains actually applied to a deck right now: the
        DJ's manual EQ with the transition's auto-EQ folded in per band."""
        eqm = self.eq_manual[name]
        eqa = self.eq_auto[name]
        return [eqm[0] * eqa[0], eqm[1] * eqa[1], eqm[2] * eqa[2]]

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

    def seek(self, deck: str, frac: float) -> None:
        """Move a deck's playhead to a 0..1 fraction of its track (dashboard
        waveform-seek / jog scrub)."""
        d = self.decks.get(deck)
        if d is not None:
            d.seek_fraction(frac)

    def scrub_crossfade(self, frac: float) -> None:
        """Push/pull a live transition by hand (no-op when not transitioning)."""
        with self._xf_lock:
            if self._xf_active:
                self._xf_done = int(min(1.0, max(0.0, frac)) * self._xf_total)

    # ---- custom transition points ---------------------------------------
    def set_mix_in(self, deck: str, sec: float) -> None:
        """Hand-set where the incoming track on `deck` drops (seconds). A
        negative value clears the override back to the analysis default."""
        d = self.decks.get(deck)
        if d is None or d.analysis is None:
            return
        d.mix_in_override = None if sec < 0 else float(min(d.analysis.duration, max(0.0, sec)))

    def set_mix_out(self, deck: str, sec: float) -> None:
        """Hand-set where the outgoing track on `deck` starts its fade (seconds).
        Negative clears the override."""
        d = self.decks.get(deck)
        if d is None or d.analysis is None:
            return
        d.mix_out_override = None if sec < 0 else float(min(d.analysis.duration, max(0.0, sec)))

    # ---- drum-loop layer ("beats" pad) ----------------------------------
    def beats_state(self) -> dict:
        return self.beats.state()

    # ---- sound-FX rack --------------------------------------------------
    def fx_state(self) -> dict:
        return self.fx.state()

    # ---- MIDI melody layer ----------------------------------------------
    def melody_state(self) -> dict:
        st = self.melody.state()
        # Attach the live track's detected key so the dashboard can show what
        # autotune is snapping the melody into.
        an = getattr(self.live_deck, "analysis", None)
        st["track_key"] = an.key_name() if an is not None else ""
        return st

    # ---- headphone cue (second output) ----------------------------------
    @staticmethod
    def list_output_devices() -> list[dict]:
        """Output-capable audio devices, for the dashboard's monitor picker."""
        try:
            import sounddevice as sd
            out = []
            for i, d in enumerate(sd.query_devices()):
                if d.get("max_output_channels", 0) > 0:
                    out.append({"index": i, "name": d["name"]})
            return out
        except Exception:  # noqa: BLE001
            return []

    def rescan_devices(self) -> list[dict]:
        """Force PortAudio to re-enumerate devices, then return the fresh
        output list. macOS only exposes a device (e.g. AirPods) to CoreAudio
        once it's actively routed, and PortAudio snapshots the device list at
        init — so anything connected after launch never appears in
        list_output_devices() until PortAudio is re-initialized. We drop the
        open streams (so there are no handles when we terminate), reinit, then
        reopen on the same devices. The master output blips briefly."""
        import sounddevice as sd
        had_cue = self.cue_device is not None and self.cue_enabled
        self._close_cue_stream()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None
        try:
            sd._terminate()
        except Exception:  # noqa: BLE001 - not initialized yet (dry-run); ignore
            pass
        try:
            sd._initialize()
        except Exception:  # noqa: BLE001
            pass
        if not self.dry_run:
            try:
                self._open_master_stream()
            except Exception as exc:  # noqa: BLE001
                self._cue_error = f"master reopen failed: {exc}"
            if had_cue:
                self._open_cue_stream()
        return self.list_output_devices()

    def _resolve_device(self, device):
        """Map an index or a name-substring to a PortAudio device index."""
        if device is None:
            return None
        try:
            return int(device)
        except (TypeError, ValueError):
            pass
        try:
            import sounddevice as sd
            needle = str(device).lower()
            for i, d in enumerate(sd.query_devices()):
                if d.get("max_output_channels", 0) > 0 and needle in d["name"].lower():
                    return i
        except Exception:  # noqa: BLE001
            pass
        return None

    def set_cue_device(self, device) -> bool:
        """(Re)point the monitor feed at an output device, opening the second
        stream. Pass None/""/"off" to turn the monitor off. Returns success."""
        with self._cue_lock:
            self._aud_active = False
            self._cue_mon_deck = ""
        self._close_cue_stream()
        off = device in (None, "", "none", "off", "None")
        self.cue_device = None if off else device
        self._cue_error = ""
        if self.cue_device is None or self.dry_run:
            return True
        return self._open_cue_stream()

    def set_cue_source(self, src: str) -> None:
        if src in ("cued", "live", "master"):
            with self._cue_lock:
                self.cue_source = src
                self._cue_mon_deck = ""   # force the preview playhead to re-anchor

    def set_cue_level(self, value: float) -> None:
        self.cue_level = float(min(1.0, max(0.0, value)))

    def start_audition(self, duration_sec: float = 12.0,
                       pre_bars: int = 2, post_bars: int = 2) -> bool:
        """Play the real transition — outgoing@mix_out blending into
        incoming@mix_in over `duration_sec`, with a couple of bars of context
        either side — to the MONITOR only. Lets the DJ hear how the blend lands
        at the current cue points, then adjust them, without touching the floor."""
        live = self.live_deck
        nxt = self.idle_deck
        if live.analysis is None or nxt.analysis is None:
            return False
        target_bpm = live.effective_bpm
        rate = target_bpm / nxt.analysis.bpm if nxt.analysis.bpm > 0 else 1.0
        rate = float(np.clip(rate, 1 - RATE_LIMIT, 1 + RATE_LIMIT))
        bp_out = live.analysis.beat_period if live.analysis.beat_period > 0 else 0.5
        pre = int(pre_bars * BEATS_PER_BAR * bp_out * SAMPLE_RATE)
        post = int(post_bars * BEATS_PER_BAR * bp_out * SAMPLE_RATE)
        total = int(max(1, duration_sec * SAMPLE_RATE))
        # Outgoing starts a couple of bars before its fade point; incoming is
        # offset so it lands exactly on mix_in at the moment the fade begins.
        from_start = max(0.0, live.eff_mix_out() * SAMPLE_RATE - pre)
        to_start = nxt.eff_mix_in() * SAMPLE_RATE - pre * rate
        with self._cue_lock:
            self._aud_from = self.current
            self._aud_to = self.idle_name
            self._aud_from_pos = from_start
            self._aud_to_pos = to_start
            self._aud_to_rate = rate
            self._aud_pre = pre
            self._aud_total = total
            self._aud_post = post
            self._aud_done = 0
            self._aud_active = True
        return True

    def stop_audition(self) -> None:
        with self._cue_lock:
            self._aud_active = False

    def cue_state(self) -> dict:
        with self._cue_lock:
            aud = self._aud_active
            span = max(1, self._aud_pre + self._aud_total + self._aud_post)
            prog = min(1.0, self._aud_done / span) if aud else 0.0
        return {
            "enabled": self.cue_enabled,
            "device": self.cue_device,
            "source": self.cue_source,
            "level": round(self.cue_level, 3),
            "auditioning": aud,
            "audition_progress": round(prog, 3),
            "error": self._cue_error,
        }

    def _open_cue_stream(self) -> bool:
        import sounddevice as sd
        dev = self._resolve_device(self.cue_device)

        def cue_callback(outdata, frames, time_info, status):
            outdata[:] = self.cue_mix(frames)

        try:
            self._cue_stream = sd.OutputStream(
                samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32",
                blocksize=BLOCK, callback=cue_callback, latency="high",
                device=dev,
            )
            self._cue_stream.start()
            self.cue_enabled = True
            return True
        except Exception as exc:  # noqa: BLE001 - report, never crash the engine
            self._cue_stream = None
            self.cue_enabled = False
            self._cue_error = str(exc)
            return False

    def _close_cue_stream(self) -> None:
        s = self._cue_stream
        self._cue_stream = None
        self.cue_enabled = False
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception:  # noqa: BLE001
                pass

    def cue_mix(self, n: int) -> np.ndarray:
        """Render the monitor block for the second output. Never advances a live
        playhead; never raises (a throw here would kill the audio thread)."""
        try:
            with self._cue_lock:
                auditioning = self._aud_active
            if auditioning:
                out = self._render_audition(n)
            else:
                out = self._render_monitor(n)
            out = out * self.cue_level
            np.clip(out, -1.0, 1.0, out=out)
            return out
        except Exception:  # noqa: BLE001
            return np.zeros((n, CHANNELS), dtype=np.float32)

    def _render_monitor(self, n: int) -> np.ndarray:
        src = self.cue_source
        if src == "master":
            lm = self._last_master
            out = np.zeros((n, CHANNELS), dtype=np.float32)
            if lm is not None:
                m = min(n, len(lm))
                out[:m] = lm[:m]
            return out
        # Pre-listen one deck on an independent preview playhead, looping from
        # its mix-in (for the cued/incoming deck) or current spot (for live).
        name = self.current if src == "live" else self.idle_name
        d = self.decks.get(name)
        if d is None or len(d.samples) <= 1:
            return np.zeros((n, CHANNELS), dtype=np.float32)
        if name != self._cue_mon_deck:
            self._cue_mon_deck = name
            anchor = d.eff_mix_in() if src == "cued" else d.position_sec
            self._cue_pos = max(0.0, anchor * SAMPLE_RATE)
        block, newpos = d.read_preview(self._cue_pos, n, d.eff_rate)
        if newpos >= len(d.samples) - 1:
            anchor = d.eff_mix_in() if src == "cued" else 0.0
            newpos = max(0.0, anchor * SAMPLE_RATE)
        self._cue_pos = newpos
        return block

    def _render_audition(self, n: int) -> np.ndarray:
        with self._cue_lock:
            if not self._aud_active:
                return np.zeros((n, CHANNELS), dtype=np.float32)
            frm = self.decks.get(self._aud_from)
            to = self.decks.get(self._aud_to)
            if frm is None or to is None:
                self._aud_active = False
                return np.zeros((n, CHANNELS), dtype=np.float32)
            done = self._aud_done
            pre, total, post = self._aud_pre, self._aud_total, self._aud_post
            from_block, self._aud_from_pos = frm.read_preview(self._aud_from_pos, n, frm.eff_rate)
            to_block, self._aud_to_pos = to.read_preview(self._aud_to_pos, n, self._aud_to_rate)
            # Equal-power gains: clamping progress to [0,1] makes the pre-roll
            # (p=0 -> outgoing solo) and post-roll (p=1 -> incoming solo) fall
            # out for free.
            i = (np.arange(n) + done).astype(np.float32)
            p = np.clip((i - pre) / float(total), 0.0, 1.0)
            gf = np.cos(p * np.pi / 2).astype(np.float32)[:, None]
            gt = np.sin(p * np.pi / 2).astype(np.float32)[:, None]
            out = from_block * gf + to_block * gt
            self._aud_done = done + n
            if self._aud_done >= pre + total + post:
                self._aud_active = False
        return out

    # ---- audio generation ------------------------------------------------
    def mix(self, n: int) -> np.ndarray:
        if self.paused:
            # Frozen: no deck advance, silent output. Resumes exactly in place.
            silent = np.zeros((n, CHANNELS), dtype=np.float32)
            self._last_master = silent
            if self.netcast is not None:
                self.netcast.feed(silent)  # keep listeners connected through a pause
            return silent
        self._advance_crossfade(n)
        out = np.zeros((n, CHANNELS), dtype=np.float32)
        with self._xf_lock:
            echoing = self._xf_active and self._xf_type == "echo"
            efrom = self._xf_from
            edelay = self._echo_delay
            morphing = self._xf_active and self._xf_type == "morph"
            mp = min(1.0, self._xf_done / self._xf_total) if morphing else 0.0
        echo_src = None
        for name, d in self.decks.items():
            g = d.gain
            if g <= 0.0 and not d.playing:
                continue
            lg, mg, hg = self.effective_bands(name)
            blk = self.eqs[name].process(d.read(n), lg, mg, hg) * (g * self.trim[name])
            if echoing and name == efrom:
                echo_src = blk
            out += blk
        # Echo throw: feed the (fast-fading) outgoing block into the delay so its
        # tail rings out under the incoming. As its gain hits zero the fresh feed
        # stops and the feedback tail decays — the classic echo-out.
        if echoing:
            try:
                src = echo_src if echo_src is not None else np.zeros((n, CHANNELS), dtype=np.float32)
                out += self._echo.process(src, edelay, 0.55) * 0.9
            except Exception:  # noqa: BLE001 - never break the audio callback
                pass
        # Morph riser: stutter-looped beat carries the outgoing voice through the
        # build, rides up, then ducks out as the incoming slams in past p~0.86.
        if morphing:
            try:
                if mp < 0.86:
                    mlevel = 0.85 + 0.15 * (mp / 0.86)
                else:
                    mlevel = max(0.0, 1.0 - (mp - 0.86) / 0.14)
                out += self._morph.process(n, mp) * mlevel
            except Exception:  # noqa: BLE001 - never break the audio callback
                pass
        # Drum layer rides on top of the deck blend, locked to the live deck.
        out += self.beats.render(n, self.live_deck)
        # MIDI melody layer, beat-locked to the live deck like the drums.
        out += self.melody.render(n, self.live_deck)
        # Triggered sound-FX one-shots, summed flat on top.
        out += self.fx.render(n)
        np.clip(out, -1.0, 1.0, out=out)
        # Cache for the "master" monitor source (the cue stream reads this, one
        # block stale at worst — fine for monitoring).
        self._last_master = out
        if self.netcast is not None:
            self.netcast.feed(out)     # broadcast to LAN listeners, if any
        return out

    # ---- backends --------------------------------------------------------
    def start(self) -> None:
        if self.dry_run:
            self._thread = threading.Thread(target=self._dummy_loop, daemon=True)
            self._thread.start()
            return
        self._open_master_stream()
        # Bring up the monitor feed too if a cue device was chosen at launch
        # (it can also be selected later from the dashboard).
        if self.cue_device is not None:
            self._open_cue_stream()

    def _open_master_stream(self) -> None:
        import sounddevice as sd

        def callback(outdata, frames, time_info, status):
            outdata[:] = self.mix(frames)

        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS,
            dtype="float32", blocksize=BLOCK, callback=callback,
            latency="high",   # deeper device buffer = more underrun cushion
            device=self._resolve_device(self.master_device),  # None = default
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
        self._close_cue_stream()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
