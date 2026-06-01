"""Sound-FX rack: paste a link, get a triggerable one-shot pad.

Download a short clip from any URL yt-dlp understands (Instagram reels, YouTube,
TikTok, a direct media link), decode it with ffmpeg, trim it to a punchy
one-shot and keep it in RAM as a pad the DJ fires over the live mix. The
downloaded source file is deleted straight after decode — disk stays clean, only
the trimmed in-memory sample survives.

No new deps: yt-dlp (already the track source) + the ffmpeg decode path. render()
is allocation-light and wrapped so it can never crash the audio callback.

Note on Instagram: yt-dlp can pull *public* reels/posts, but private or
login-walled media needs cookies and will simply fail here (surfaced in the log,
never by asking for credentials). Public reels and ordinary YouTube/TikTok links
are the happy path.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from urllib.parse import urlparse

import numpy as np

from . import CHANNELS, SAMPLE_RATE
from .audio_io import decode

FX_MAX_SEC = 6.0       # hard cap on a one-shot's length
MAX_SLOTS = 8          # pads kept in the rack at once
_SILENCE = 0.02        # lead-in trim threshold (fraction of peak)
_FADE_SEC = 0.006      # click-guard fade at both edges


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
    __slots__ = ("name", "status", "samples", "pos", "active", "gain")

    def __init__(self, name: str) -> None:
        self.name = name
        self.status = "loading"        # loading | ready | error
        self.samples: np.ndarray | None = None
        self.pos = 0
        self.active = False
        self.gain = 1.0


class FXRack:
    """A handful of one-shot pads + a master FX level. Thread-safe: the small
    slot list is mutated only under a lock, and render() works on a snapshot."""

    def __init__(self) -> None:
        self.level = 0.8
        self._slots: list[_Slot] = []
        self._lock = threading.Lock()

    # ---- controls --------------------------------------------------------
    def set_level(self, v: float) -> None:
        self.level = float(min(1.0, max(0.0, v)))

    def trigger(self, idx: int) -> None:
        """(Re)start a pad from its head — retriggering chokes the prior hit."""
        with self._lock:
            if 0 <= idx < len(self._slots):
                sl = self._slots[idx]
                if sl.status == "ready" and sl.samples is not None:
                    sl.pos = 0
                    sl.active = True

    def clear(self, idx: int) -> None:
        with self._lock:
            if 0 <= idx < len(self._slots):
                self._slots.pop(idx)

    def clear_all(self) -> None:
        with self._lock:
            self._slots = []

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
        """Sum every playing pad's next block. Silence when idle."""
        try:
            with self._lock:
                slots = list(self._slots)
            out = None
            for sl in slots:
                if not sl.active or sl.samples is None:
                    continue
                s = sl.samples
                p = sl.pos
                L = len(s)
                if p >= L:
                    sl.active = False
                    continue
                k = min(n, L - p)
                if out is None:
                    out = np.zeros((n, CHANNELS), dtype=np.float32)
                out[:k] += s[p:p + k] * sl.gain
                sl.pos = p + k
                if sl.pos >= L:
                    sl.active = False
            if out is None:
                return np.zeros((n, CHANNELS), dtype=np.float32)
            return out * self.level
        except Exception:  # noqa: BLE001 - never break the audio callback
            return np.zeros((n, CHANNELS), dtype=np.float32)

    # ---- internals -------------------------------------------------------
    def _append(self, slot: _Slot) -> bool:
        with self._lock:
            if len(self._slots) >= MAX_SLOTS:
                # Evict the oldest idle, fully-loaded pad to make room.
                for i, sl in enumerate(self._slots):
                    if not sl.active and sl.status != "loading":
                        self._slots.pop(i)
                        break
                else:
                    return False
            self._slots.append(slot)
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
