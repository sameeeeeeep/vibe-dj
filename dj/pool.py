"""A streaming track pool for YouTube sources.

Keeps a rolling window of `buffer` tracks downloaded + analyzed on disk (1 that
is playing plus a few lookahead), so the controller always has real choices for
"what's best next" and the next tracks are pre-fetched (no buffering at the
transition). Tracks are deleted once they finish, so disk stays bounded; the
playlist loops so the set never runs out.

Duck-types Library (`tracks`, `load_audio`, `release`) so the Controller uses
it unchanged.
"""

from __future__ import annotations

import collections
import os
import threading
import time
from typing import Callable, Optional

from . import SAMPLE_RATE
from .analysis import analyze
from .audio_io import decode
from .library import Track, score_energy
from . import youtube_source as yt


class TrackPool:
    def __init__(self, urls: list[str], cache_dir: str = "yt_cache",
                 buffer: int = 5, limit: int = 0, ephemeral: bool = True,
                 log: Optional[Callable[[str], None]] = None):
        self.cache_dir = cache_dir
        self.buffer = max(2, buffer)
        self.limit = limit
        self.ephemeral = ephemeral
        self.log = log or (lambda m: None)
        self._urls = urls

        self._lock = threading.Lock()
        self._tracks: list[Track] = []
        self._queue: collections.deque[dict] = collections.deque()
        self._candidates: list[dict] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---- Library-compatible surface -------------------------------------
    @property
    def tracks(self) -> list[Track]:
        with self._lock:
            return list(self._tracks)        # snapshot: filler may mutate concurrently

    def load_audio(self, track: Track) -> np.ndarray:  # noqa: F821 - numpy via decode
        return decode(track.path, sr=SAMPLE_RATE)

    def release(self, track: Track) -> None:
        with self._lock:
            if track in self._tracks:
                self._tracks.remove(track)
                score_energy(self._tracks)
        if self.ephemeral:
            try:
                os.remove(track.path)
            except OSError:
                pass
        self.log(f"[drop]  {track.name[:38]}  deleted  (pool {len(self._tracks)})")

    # ---- lifecycle -------------------------------------------------------
    def prime(self) -> int:
        """List the playlist and download the first track synchronously."""
        self._candidates = yt.list_entries(self._urls, limit=self.limit, log=self.log)
        self._queue.extend(self._candidates)
        self.log(f"queued {len(self._candidates)} candidate tracks")
        self._fill_once()
        return len(self._tracks)

    def start(self) -> "TrackPool":
        self._thread = threading.Thread(target=self._filler, daemon=True)
        self._thread.start()
        return self

    def _filler(self) -> None:
        while not self._stop.is_set():
            if len(self._tracks) < self.buffer and self._queue:
                self._fill_once()
            elif not self._queue and self._candidates:
                self._requeue()                  # loop the playlist
                time.sleep(0.3)
            else:
                time.sleep(0.4)

    def _requeue(self) -> None:
        with self._lock:
            have = {t.path for t in self._tracks}
        for c in self._candidates:
            self._queue.append(c)

    def _fill_once(self) -> None:
        if not self._queue:
            return
        entry = self._queue.popleft()
        path = yt.download_one(entry, self.cache_dir, log=self.log)
        if not path:
            return
        if any(t.path == path for t in self.tracks):
            return                               # already buffered
        try:
            audio = decode(path, sr=SAMPLE_RATE)
            an = analyze(audio, sr=SAMPLE_RATE)
        except Exception as exc:  # noqa: BLE001
            self.log(f"  analyze failed {entry.get('title')}: {exc}")
            return
        track = Track(path=path, name=entry.get("title") or os.path.basename(path), analysis=an)
        with self._lock:
            self._tracks.append(track)
            score_energy(self._tracks)
        self.log(f"[buf]   {track.name[:38]:38} {track.bpm:5.0f} BPM  "
                 f"energy {track.energy:.2f}  (pool {len(self._tracks)})")

    def stop(self) -> None:
        self._stop.set()
