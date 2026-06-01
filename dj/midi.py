"""Load a Standard MIDI File and play it as a beat-locked synth melody layer.

Two pieces:
  * parse_midi() — a tiny SMF reader (stdlib bytes only, NO mido/pretty_midi). It
    returns notes positioned in *beats* (quarter-notes), independent of the file's
    own tempo, because the layer is slaved to the live deck's tempo — not the
    file's. So we only need the division (ticks-per-quarter), never the tempo map.
  * MelodyLayer — synthesises those notes into a one-loop buffer (numpy poly-osc +
    envelope), sized to the deck's heard bar length and beat-locked the same way
    the drum layer is. Tempo follows varispeed for free; transpose just rebuilds.

render() is wrapped so it can never crash the audio callback.
"""

from __future__ import annotations

import math
import os
import threading

import numpy as np

from . import CHANNELS, SAMPLE_RATE

_DRUM_CHANNEL = 9          # GM percussion channel — skipped (it's not a melody)
MAX_NOTES = 4000          # safety cap so a pathological file can't blow up RAM


# ---- Standard MIDI File parsing ------------------------------------------
def _read_vlq(data: bytes, i: int) -> tuple[int, int]:
    """Read a variable-length quantity; return (value, next_index)."""
    val = 0
    while i < len(data):
        b = data[i]
        i += 1
        val = (val << 7) | (b & 0x7F)
        if not (b & 0x80):
            break
    return val, i


def _parse_track(data: bytes) -> list[tuple]:
    """Return absolute-tick events: (tick, kind, pitch, vel, channel)."""
    events: list[tuple] = []
    i = 0
    tick = 0
    status = 0
    n = len(data)
    while i < n:
        dt, i = _read_vlq(data, i)
        tick += dt
        if i >= n:
            break
        b0 = data[i]
        if b0 == 0xFF:                     # meta event
            i += 1
            mtype = data[i]
            i += 1
            length, i = _read_vlq(data, i)
            i += length
            if mtype == 0x2F:              # end of track
                break
            continue
        if b0 in (0xF0, 0xF7):            # sysex
            i += 1
            length, i = _read_vlq(data, i)
            i += length
            continue
        if b0 & 0x80:                     # new status byte
            status = b0
            i += 1
        # else: running status — reuse previous status, b0 is the first data byte
        et = status & 0xF0
        ch = status & 0x0F
        if et in (0xC0, 0xD0):           # program / channel-pressure: 1 data byte
            i += 1
            continue
        if i >= n:                       # truncated tail — stop cleanly
            break
        a = data[i] if i < n else 0
        i += 1
        b = data[i] if i < n else 0
        i += 1
        if et == 0x90 and b > 0:
            events.append((tick, "on", a, b, ch))
        elif et == 0x80 or (et == 0x90 and b == 0):
            events.append((tick, "off", a, 0, ch))
        # other channel messages (CC, pitch-bend, aftertouch) are ignored
    return events


def parse_midi(path: str) -> dict:
    """Parse an SMF into {notes, length_beats, name}. notes is a list of
    (pitch, start_beat, dur_beats, velocity). Raises ValueError on a bad file."""
    with open(path, "rb") as fh:
        blob = fh.read()
    if len(blob) < 14 or blob[:4] != b"MThd":
        raise ValueError("not a Standard MIDI File (missing MThd)")
    hlen = int.from_bytes(blob[4:8], "big")
    division = int.from_bytes(blob[12:14], "big")
    if division & 0x8000:                 # SMPTE timing — rare for melodies
        ppq = 480                         # fall back to a sane default
    else:
        ppq = division or 480
    pos = 8 + hlen

    events: list[tuple] = []
    while pos + 8 <= len(blob):
        if blob[pos:pos + 4] != b"MTrk":
            break
        tlen = int.from_bytes(blob[pos + 4:pos + 8], "big")
        body = blob[pos + 8:pos + 8 + tlen]
        pos += 8 + tlen
        events.extend(_parse_track(body))

    # Pair note-ons with note-offs (FIFO per (channel,pitch) for overlaps).
    events.sort(key=lambda e: e[0])
    pending: dict[tuple, list[tuple]] = {}
    notes: list[tuple] = []
    max_tick = 0
    for tick, kind, pitch, vel, ch in events:
        max_tick = max(max_tick, tick)
        if ch == _DRUM_CHANNEL:
            continue
        key = (ch, pitch)
        if kind == "on":
            pending.setdefault(key, []).append((tick, vel))
        else:  # off
            stack = pending.get(key)
            if stack:
                start_tick, v = stack.pop(0)
                notes.append((tick, start_tick, pitch, v))
    # Close any hanging note-ons at the last event tick.
    for (ch, pitch), stack in pending.items():
        for start_tick, v in stack:
            notes.append((max_tick, start_tick, pitch, v))

    out: list[tuple] = []
    for end_tick, start_tick, pitch, v in notes:
        sb = start_tick / ppq
        db = max(end_tick - start_tick, ppq // 16) / ppq   # min ~1/16-note
        out.append((int(pitch), float(sb), float(db), int(v)))
    out.sort(key=lambda nt: nt[1])
    if len(out) > MAX_NOTES:
        out = out[:MAX_NOTES]

    if not out:
        raise ValueError("no playable melodic notes found")
    max_end = max(nt[1] + nt[2] for nt in out)
    # Loop on a whole bar (4 beats), min one bar.
    length_beats = max(4, int(math.ceil(max_end / 4.0 - 1e-6) * 4))
    return {"notes": out, "length_beats": length_beats,
            "name": os.path.basename(path)}


# ---- the synth melody layer ----------------------------------------------
class MelodyLayer:
    """A loaded MIDI melody, synthesised and looped beat-locked to the live deck.
    Off until a file is loaded and enabled. Thread-safe for the audio callback:
    the loaded song is published as one immutable tuple, and render() works on a
    cached loop rebuilt only on a real change (new file / transpose / tempo)."""

    def __init__(self) -> None:
        self.enabled = False
        self.level = 0.6
        self.transpose = 0          # semitones, clamped +/- 24
        # Published atomically by load(): (notes, length_beats, name, rev) or None
        self._song: tuple | None = None
        self._rev = 0
        self._lock = threading.Lock()
        # render cache
        self._loop: np.ndarray | None = None
        self._loop_frames = 0
        self._loop_transpose = 0
        self._loop_rev = -1

    # ---- controls --------------------------------------------------------
    def set_enabled(self, on: bool) -> None:
        self.enabled = bool(on)

    def toggle(self) -> None:
        self.enabled = not self.enabled

    def set_level(self, v: float) -> None:
        self.level = float(min(1.0, max(0.0, v)))

    def set_transpose(self, semitones: float) -> None:
        self.transpose = int(max(-24, min(24, round(semitones))))

    def clear(self) -> None:
        with self._lock:
            self._song = None
            self.enabled = False
            self._rev += 1

    def load(self, path: str, log=None) -> bool:
        log = log or (lambda _m: None)
        try:
            parsed = parse_midi(path)
        except Exception as exc:  # noqa: BLE001 - surface, keep current melody
            log(f"[midi]  load failed: {exc}")
            return False
        with self._lock:
            self._rev += 1
            self._song = (tuple(parsed["notes"]), parsed["length_beats"],
                          parsed["name"], self._rev)
        log(f"[midi]  loaded '{parsed['name']}'  {len(parsed['notes'])} notes  "
            f"{parsed['length_beats'] // 4} bar(s)")
        return True

    def state(self) -> dict:
        song = self._song
        return {
            "enabled": self.enabled,
            "level": round(self.level, 3),
            "transpose": self.transpose,
            "loaded": song is not None,
            "name": song[2] if song else "",
            "notes": len(song[0]) if song else 0,
            "bars": (song[1] // 4) if song else 0,
        }

    # ---- audio -----------------------------------------------------------
    def _synth_loop(self, loop_frames: int, length_beats: int,
                    notes: tuple, transpose: int) -> np.ndarray:
        buf = np.zeros(loop_frames, dtype=np.float32)
        fpb = loop_frames / length_beats          # frames per beat
        atk = max(1, int(0.008 * SAMPLE_RATE))
        rel = max(1, int(0.030 * SAMPLE_RATE))
        for pitch, sb, db, vel in notes:
            start = int(sb * fpb) % loop_frames
            ln = max(1, int(db * fpb))
            f = 440.0 * 2.0 ** ((pitch - 69 + transpose) / 12.0)
            t = np.arange(ln, dtype=np.float32) / SAMPLE_RATE
            ph = 2 * np.pi * f * t
            wave = np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.12 * np.sin(3 * ph)
            env = np.exp(-t * 1.8)                 # gentle pluck/pad decay
            a = min(atk, ln)
            if a > 1:
                env[:a] *= np.linspace(0.0, 1.0, a, dtype=np.float32)
            r = min(rel, ln)
            if r > 1:
                env[-r:] *= np.linspace(1.0, 0.0, r, dtype=np.float32)
            seg = (wave * env * (0.18 * vel / 127.0)).astype(np.float32)
            end = start + ln
            if end <= loop_frames:
                buf[start:end] += seg
            else:                                  # wrap the tail past the loop end
                k = loop_frames - start
                buf[start:] += seg[:k]
                buf[:end - loop_frames] += seg[k:]
        np.clip(buf, -1.0, 1.0, out=buf)
        return np.stack([buf, buf], axis=1).astype(np.float32)

    def render(self, n: int, deck) -> np.ndarray:
        try:
            if not self.enabled:
                return np.zeros((n, CHANNELS), dtype=np.float32)
            song = self._song
            an = getattr(deck, "analysis", None)
            if song is None or an is None or not deck.playing or an.beat_period <= 0:
                return np.zeros((n, CHANNELS), dtype=np.float32)
            notes, length_beats, _name, rev = song
            eff = deck.eff_rate
            beat_frames = an.beat_period / eff * SAMPLE_RATE
            loop_frames = int(round(length_beats * beat_frames))
            if loop_frames < 2048:
                return np.zeros((n, CHANNELS), dtype=np.float32)
            tr = self.transpose
            if (self._loop is None or self._loop_rev != rev
                    or self._loop_transpose != tr
                    or abs(loop_frames - self._loop_frames)
                    > max(64, 0.006 * max(1, self._loop_frames))):
                self._loop = self._synth_loop(loop_frames, length_beats, notes, tr)
                self._loop_frames = loop_frames
                self._loop_transpose = tr
                self._loop_rev = rev
            loop = self._loop
            L = self._loop_frames
            t = deck.position_sec - an.beat_offset
            phase = ((t / an.beat_period) / length_beats) % 1.0
            if phase < 0:
                phase += 1.0
            start = int(phase * L) % L
            idx = (start + np.arange(n)) % L
            return loop[idx] * self.level
        except Exception:  # noqa: BLE001 - never break the audio callback
            return np.zeros((n, CHANNELS), dtype=np.float32)
