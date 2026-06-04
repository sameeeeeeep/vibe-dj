"""Sound-FX rack: paste a link, get a triggerable one-shot pad.

Download a short clip from any URL yt-dlp understands (Instagram reels, YouTube,
TikTok, a direct media link), decode it with ffmpeg, trim it to a punchy
one-shot and keep it in RAM as a pad the DJ fires over the live mix. The
downloaded source file is deleted straight after decode — disk stays clean, only
the trimmed in-memory sample survives.

Each pad fires as a one-shot or, latched into loop mode, repeats forever until
tapped off. Loaded effects are also saved to a small on-disk library (int16,
gitignored) and reload as ready pads on the next startup, so the rack is a
growing, reusable effect library rather than a scratchpad.

No new deps: yt-dlp (already the track source) + the ffmpeg decode path. render()
is allocation-light and wrapped so it can never crash the audio callback.

Note on Instagram: yt-dlp can pull *public* reels/posts, but private or
login-walled media needs cookies and will simply fail here (surfaced in the log,
never by asking for credentials). Public reels and ordinary YouTube/TikTok links
are the happy path.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import uuid
from urllib.parse import urlparse

import numpy as np

from . import CHANNELS, SAMPLE_RATE
from .audio_io import decode

FX_MAX_SEC = 6.0       # hard cap on a one-shot's length
MAX_SLOTS = 24         # pads kept in the rack/library at once
_SILENCE = 0.02        # lead-in trim threshold (fraction of peak)
_FADE_SEC = 0.006      # click-guard fade at both edges
LIBRARY_DIR = "fx_library"   # persisted, reloaded on startup (gitignored)


def _trim_oneshot(audio: np.ndarray) -> np.ndarray:
    """Trim leading silence, cap length, normalise, and fade the very edges so
    triggering is click-free. Returns (frames, CHANNELS) float32 (empty if the
    clip is unusably short)."""
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.shape[1] == 1:
        audio = np.repeat(audio, CHANNELS, axis=1)
    elif audio.shape[1] > CHANNELS:
        audio = audio[:, :CHANNELS]
    peak = float(np.max(np.abs(audio))) + 1e-9
    # Start on the first transient (skip dead air / fade-ins at the head).
    mono = np.max(np.abs(audio), axis=1)
    above = np.nonzero(mono > _SILENCE * peak)[0]
    start = int(above[0]) if above.size else 0
    audio = audio[start:]
    maxn = int(FX_MAX_SEC * SAMPLE_RATE)
    if len(audio) > maxn:
        audio = audio[:maxn]
    if len(audio) < 16:
        return np.zeros((0, CHANNELS), dtype=np.float32)
    audio = (audio / peak * 0.9).astype(np.float32)
    f = min(int(_FADE_SEC * SAMPLE_RATE), len(audio) // 2)
    if f > 1:
        ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)[:, None]
        audio[:f] *= ramp
        audio[-f:] *= ramp[::-1]
    return np.ascontiguousarray(audio)


class _Slot:
    __slots__ = ("name", "status", "samples", "pos", "active", "gain",
                 "loop", "lib_id")

    def __init__(self, name: str) -> None:
        self.name = name
        self.status = "loading"        # loading | ready | error
        self.samples: np.ndarray | None = None
        self.pos = 0
        self.active = False
        self.gain = 1.0
        self.loop = False              # latched: wrap forever until tapped off
        self.lib_id = ""               # id of this clip's file in the library


class FXRack:
    """A handful of one-shot pads + a master FX level. Thread-safe: the small
    slot list is mutated only under a lock, and render() works on a snapshot."""

    def __init__(self, library_dir: str = LIBRARY_DIR) -> None:
        self.level = 0.8
        self._slots: list[_Slot] = []
        self._lock = threading.Lock()
        self._lib_dir = library_dir
        self._load_library()           # restore previously-loaded effects

    # ---- controls --------------------------------------------------------
    def set_level(self, v: float) -> None:
        # Ceiling > 1.0 so pads can be pushed *louder* than unity (up to +6 dB).
        self.level = float(min(2.0, max(0.0, v)))

    def trigger(self, idx: int) -> None:
        """Fire a pad. A one-shot (re)starts from its head — retriggering chokes
        the prior hit. A looping pad is *latched*: tapping it while it's running
        stops it, so the same tap both arms and disarms the loop."""
        with self._lock:
            if 0 <= idx < len(self._slots):
                sl = self._slots[idx]
                if sl.status == "ready" and sl.samples is not None:
                    if sl.loop and sl.active:
                        sl.active = False        # latched loop: tap again = stop
                    else:
                        sl.pos = 0
                        sl.active = True

    def set_loop(self, idx: int, on: bool) -> None:
        """Arm/disarm loop mode. Arming a loop *starts it immediately* from the
        head (so the one ⟳ tap both turns the mode on and gets it playing);
        disarming stops it now. No separate trigger needed."""
        with self._lock:
            if 0 <= idx < len(self._slots):
                self._apply_loop(self._slots[idx], bool(on))

    def toggle_loop(self, idx: int) -> None:
        with self._lock:
            if 0 <= idx < len(self._slots):
                sl = self._slots[idx]
                self._apply_loop(sl, not sl.loop)

    @staticmethod
    def _apply_loop(sl: "_Slot", on: bool) -> None:
        sl.loop = on
        if on:
            if sl.status == "ready" and sl.samples is not None:
                sl.pos = 0
                sl.active = True          # one tap arms *and* starts the loop
        else:
            sl.active = False             # disarming halts it right away

    def clear(self, idx: int) -> None:
        lib_id = ""
        with self._lock:
            if 0 <= idx < len(self._slots):
                lib_id = self._slots[idx].lib_id
                self._slots.pop(idx)
        if lib_id:
            self._delete_lib_file(lib_id)
            self._write_index()

    def clear_all(self) -> None:
        with self._lock:
            ids = [sl.lib_id for sl in self._slots if sl.lib_id]
            self._slots = []
        for lib_id in ids:
            self._delete_lib_file(lib_id)
        self._write_index()

    def state(self) -> dict:
        with self._lock:
            slots = list(self._slots)
        return {
            "level": round(self.level, 3),
            "max": MAX_SLOTS,
            "slots": [
                {
                    "name": sl.name,
                    "status": sl.status,
                    "dur": round(len(sl.samples) / SAMPLE_RATE, 2)
                    if sl.samples is not None else 0.0,
                    "active": bool(sl.active),
                    "loop": bool(sl.loop),
                }
                for sl in slots
            ],
        }

    # ---- loading ---------------------------------------------------------
    def add_from_url(self, url: str, log=None):
        """Download → decode → trim a one-shot into a fresh pad. Meant to run on
        a worker thread (it blocks on the network). The source file is deleted
        once decoded. Returns the slot on success, else None."""
        log = log or (lambda _m: None)
        slot = _Slot(self._short_name(url))
        if not self._append(slot):
            log("[fx]    rack full — clear a pad first")
            return None
        log(f"[fx]    fetching {url} …")
        tmp = tempfile.mkdtemp(prefix="fx_")
        try:
            path, title = self._download(url, tmp, log)
            if not path:
                self._fail(slot, log, "download failed (private/login-walled?)")
                return None
            if title:
                slot.name = title[:40]
            oneshot = _trim_oneshot(decode(path))
            if len(oneshot) == 0:
                self._fail(slot, log, "clip too short / silent")
                return None
            slot.samples = oneshot
            slot.status = "ready"
            self._save_to_library(slot)          # persist for reuse / restart
            log(f"[fx]    ready '{slot.name}'  {len(oneshot) / SAMPLE_RATE:.1f}s")
            return slot
        except Exception as exc:  # noqa: BLE001 - surface, don't crash the server
            self._fail(slot, log, str(exc))
            return None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)   # self-cleaning: drop source

    def _download(self, url: str, cache_dir: str, log):
        import yt_dlp

        opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(cache_dir, "fx.%(ext)s"),
            "restrictfilenames": True,
            "quiet": True, "no_warnings": True, "noprogress": True,
            "ignoreerrors": True, "noplaylist": True,
        }
        title = None
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info and info.get("entries"):
                info = info["entries"][0] if info["entries"] else None
            if info:
                title = info.get("title")
                path = ydl.prepare_filename(info)
                if path and os.path.exists(path):
                    return path, title
        # yt-dlp may remux/rename; grab whatever single file landed in the dir.
        for f in sorted(os.listdir(cache_dir)):
            full = os.path.join(cache_dir, f)
            if os.path.isfile(full):
                return full, title
        return None, title

    # ---- audio -----------------------------------------------------------
    def render(self, n: int) -> np.ndarray:
        """Sum every playing pad's next block. Silence when idle. A looping pad
        wraps back to its head and fills the whole block (the 6 ms edge fades on
        each clip keep the seam click-free); a one-shot stops at its tail."""
        try:
            with self._lock:
                slots = list(self._slots)
            out = None
            for sl in slots:
                if not sl.active or sl.samples is None:
                    continue
                s = sl.samples
                L = len(s)
                p = sl.pos
                if out is None:
                    out = np.zeros((n, CHANNELS), dtype=np.float32)
                if sl.loop:
                    filled = 0
                    while filled < n:
                        if p >= L:
                            p = 0
                        k = min(n - filled, L - p)
                        out[filled:filled + k] += s[p:p + k] * sl.gain
                        p += k
                        filled += k
                    sl.pos = p
                else:
                    if p >= L:
                        sl.active = False
                        continue
                    k = min(n, L - p)
                    out[:k] += s[p:p + k] * sl.gain
                    sl.pos = p + k
                    if sl.pos >= L:
                        sl.active = False
            if out is None:
                return np.zeros((n, CHANNELS), dtype=np.float32)
            return out * self.level
        except Exception:  # noqa: BLE001 - never break the audio callback
            return np.zeros((n, CHANNELS), dtype=np.float32)

    # ---- library (persist / restore loaded effects) ----------------------
    def _save_to_library(self, slot: _Slot) -> None:
        """Persist a ready pad's trimmed clip to disk (int16, ~half the size of
        float32) and refresh the index, so it reloads as a pad next startup."""
        if slot.samples is None:
            return
        if not slot.lib_id:
            slot.lib_id = uuid.uuid4().hex[:8]
        try:
            os.makedirs(self._lib_dir, exist_ok=True)
            arr = np.clip(slot.samples * 32767.0, -32768, 32767).astype(np.int16)
            np.save(os.path.join(self._lib_dir, slot.lib_id + ".npy"), arr)
        except OSError:
            return
        self._write_index()

    def _write_index(self) -> None:
        with self._lock:
            idx = [{"id": sl.lib_id, "name": sl.name}
                   for sl in self._slots if sl.lib_id]
        try:
            os.makedirs(self._lib_dir, exist_ok=True)
            with open(os.path.join(self._lib_dir, "index.json"), "w") as fh:
                json.dump(idx, fh)
        except OSError:
            pass

    def _delete_lib_file(self, lib_id: str) -> None:
        try:
            os.remove(os.path.join(self._lib_dir, lib_id + ".npy"))
        except OSError:
            pass

    def _load_library(self) -> None:
        """Rehydrate previously-saved effects into ready pads. Missing or corrupt
        files are skipped silently, so a stale index never blocks startup."""
        try:
            with open(os.path.join(self._lib_dir, "index.json")) as fh:
                entries = json.load(fh)
        except (OSError, json.JSONDecodeError, ValueError):
            return
        restored: list[_Slot] = []
        for e in entries:
            lib_id = (e or {}).get("id", "")
            if not lib_id:
                continue
            try:
                arr = np.load(os.path.join(self._lib_dir, lib_id + ".npy"))
            except (OSError, ValueError):
                continue
            samples = arr.astype(np.float32) / 32767.0
            if samples.ndim == 1:
                samples = samples[:, None]
            sl = _Slot(e.get("name", "clip"))
            sl.samples = np.ascontiguousarray(samples)
            sl.status = "ready"
            sl.lib_id = lib_id
            restored.append(sl)
            if len(restored) >= MAX_SLOTS:
                break
        if restored:
            with self._lock:
                self._slots = restored

    # ---- internals -------------------------------------------------------
    def _append(self, slot: _Slot) -> bool:
        evicted = ""
        with self._lock:
            if len(self._slots) >= MAX_SLOTS:
                # Evict the oldest idle, fully-loaded pad to make room.
                for i, sl in enumerate(self._slots):
                    if not sl.active and sl.status != "loading":
                        evicted = sl.lib_id
                        self._slots.pop(i)
                        break
                else:
                    return False
            self._slots.append(slot)
        if evicted:
            self._delete_lib_file(evicted)      # drop the evicted clip from disk
            self._write_index()
        return True

    def _fail(self, slot: _Slot, log, why: str) -> None:
        slot.status = "error"
        with self._lock:
            try:
                self._slots.remove(slot)
            except ValueError:
                pass
        log(f"[fx]    {why}")

    @staticmethod
    def _short_name(url: str) -> str:
        try:
            p = urlparse(url)
            base = p.path.rstrip("/").split("/")[-1] or p.netloc or "clip"
        except Exception:  # noqa: BLE001
            base = "clip"
        return (base[:24] or "clip")
