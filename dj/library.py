"""Scan a folder, analyse each track (cached), and score energy across the set."""

from __future__ import annotations

import dataclasses
import json
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import SAMPLE_RATE
from .analysis import Analysis, analyze
from .audio_io import decode

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".oga",
              ".opus", ".webm", ".mp4", ".aiff", ".aif"}
CACHE_NAME = ".dj_cache.json"
# yt-dlp names cached files "<title> [<11-char-id>].<ext>"; pull the id back out
# so a runtime download can be recorded in the manifest and re-fetched later.
MANIFEST_NAME = "yt_library_manifest.json"
_VID_RE = re.compile(r"\[([A-Za-z0-9_-]{11})\]")


def score_energy(tracks: list["Track"]) -> None:
    """Rank-normalise each track's energy to 0..1 across the given set.

    Used for both the static library and the streaming pool, so a pool can
    re-score itself whenever a track is added or released.
    """
    if not tracks:
        return
    rms = np.array([t.analysis.rms for t in tracks])
    cen = np.array([t.analysis.centroid for t in tracks])
    ons = np.array([t.analysis.onset_rate for t in tracks])

    def rank(x: np.ndarray) -> np.ndarray:
        if len(x) == 1:
            return np.array([0.5])
        order = x.argsort()
        r = np.empty_like(order, dtype=float)
        r[order] = np.linspace(0.0, 1.0, len(x))
        return r

    score = 0.4 * rank(rms) + 0.4 * rank(ons) + 0.2 * rank(cen)
    for t, s in zip(tracks, score):
        t.energy = float(s)


@dataclass
class Track:
    path: str
    name: str
    analysis: Analysis
    energy: float = 0.5         # 0..1, normalised across the library
    play_count: int = 0
    last_played_at: float = 0.0  # monotonic seconds; 0 = never
    # Runtime YouTube download (jukebox/auto-dig) recorded in the manifest, so it
    # can be deleted after play and re-fetched later. User's own files stay False.
    ephemeral: bool = False

    @property
    def bpm(self) -> float:
        return self.analysis.bpm


def _iter_audio_files(folder: str):
    for root, _dirs, files in os.walk(folder):
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() in AUDIO_EXTS:
                yield os.path.join(root, fn)


def _load_cache(folder: str) -> dict:
    path = os.path.join(folder, CACHE_NAME)
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(folder: str, cache: dict) -> None:
    path = os.path.join(folder, CACHE_NAME)
    try:
        with open(path, "w") as fh:
            json.dump(cache, fh)
    except OSError:
        pass


@dataclass
class Library:
    folder: str
    tracks: list[Track] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Guards concurrent runtime additions (add_url) against the controller /
        # dashboard threads reading self.tracks. Not a dataclass field.
        self._add_lock = threading.Lock()
        # Manifest lives beside the cache folder (repo root for yt_cache), matching
        # tools/redownload_library.py, so a trimmed cache can be restored later.
        self._manifest_path = os.path.join(
            os.path.dirname(os.path.abspath(self.folder)) or ".", MANIFEST_NAME)

    def scan(self, progress: Optional[Callable[[int, int, str], None]] = None) -> "Library":
        files = list(_iter_audio_files(self.folder))
        cache = _load_cache(self.folder)
        new_cache: dict = {}
        self.tracks = []

        for i, path in enumerate(files, 1):
            name = os.path.splitext(os.path.basename(path))[0]
            if progress:
                progress(i, len(files), name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            key = path
            cached = cache.get(key)
            sig = {"mtime": st.st_mtime, "size": st.st_size}
            if (cached and cached.get("mtime") == sig["mtime"]
                    and cached.get("size") == sig["size"]
                    and "key_root" in (cached.get("analysis") or {})):
                # "key_root" gate forces re-analysis of entries cached before key
                # detection existed, so autotune has a key to snap to.
                an = Analysis(**cached["analysis"])
            else:
                try:
                    audio = decode(path, sr=SAMPLE_RATE)
                except Exception:
                    continue
                an = analyze(audio, sr=SAMPLE_RATE)
            new_cache[key] = {**sig, "analysis": dataclasses.asdict(an)}
            self.tracks.append(Track(path=path, name=name, analysis=an))

        _save_cache(self.folder, new_cache)
        self._score_energy()
        return self

    def _score_energy(self) -> None:
        score_energy(self.tracks)

    def load_audio(self, track: Track) -> np.ndarray:
        return decode(track.path, sr=SAMPLE_RATE)

    def release(self, track: Track) -> None:
        """Free a *runtime-downloaded* track once it has actually played: drop it
        from the live set and delete the on-disk file (it's recorded in the
        manifest, so it can be re-fetched). The user's own folder files (not
        ephemeral) are never touched, keeping the static-folder contract.

        The ``play_count >= 1`` gate matters: the controller also calls release()
        when *unstaging* a cued-but-never-played track (it decrements play_count to
        0 first), and we must keep those — a guest's queued request bumped off the
        cue shouldn't be deleted before it ever plays."""
        if not getattr(track, "ephemeral", False) or track.play_count < 1:
            return
        with self._add_lock:
            self.tracks = [t for t in self.tracks if t is not track]
            self._score_energy()
        try:
            os.remove(track.path)
        except OSError:
            pass

    def add_analyzed(self, track: Track) -> Optional[Track]:
        """Splice an already-analysed Track into the live set. Thread-safe: the
        list is rebuilt and atomically rebound so the controller/dashboard never
        see a torn list, and energies are re-ranked across the set. Returns the
        Track, or None if one with the same path is already present.

        Any track whose filename carries a YouTube id (every runtime download —
        jukebox or auto-dig) is marked ephemeral and recorded in the manifest, so
        release() can delete the file after play and it stays re-fetchable."""
        vid_match = _VID_RE.search(os.path.basename(track.path))
        with self._add_lock:
            if any(t.path == track.path for t in self.tracks):
                return None
            if vid_match:
                track.ephemeral = True
            self.tracks = list(self.tracks) + [track]
            self._score_energy()
            if vid_match:
                self._manifest_add(vid_match.group(1), track.name, track.path)
        return track

    def add_track_file(self, path: str,
                       log: Optional[Callable[[str], None]] = None) -> Optional[Track]:
        """Analyse a single already-downloaded audio file and splice it in.
        Returns the Track, or None if it can't be analysed / is already present.
        This is the shared ingest path for both add_url and the auto crate-digger."""
        log = log or (lambda m: None)
        if any(t.path == path for t in self.tracks):
            return None
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            audio = decode(path, sr=SAMPLE_RATE)
            an = analyze(audio, sr=SAMPLE_RATE)
        except Exception as exc:  # noqa: BLE001 - one bad file shouldn't abort
            log(f"  analyse failed {name}: {exc}")
            return None
        return self.add_analyzed(Track(path=path, name=name, analysis=an))

    def add_url(self, url: str, log: Optional[Callable[[str], None]] = None) -> list[Track]:
        """Download a YouTube URL into this library's folder, analyse it, and
        splice the new track(s) into the live set. Returns the Track(s) added
        (a playlist URL may add several). Safe to call from a worker thread:
        the slow download/analysis happens lock-free, and only the brief list
        splice + energy re-score is serialised. The autopilot picks the new
        track up on its next selection with no other wiring.
        """
        from . import youtube_source as yt

        log = log or (lambda m: None)
        existing = {t.path for t in self.tracks}          # snapshot before fetch
        yt.fetch([url], cache_dir=self.folder, log=log)   # network — slow, lock-free

        added: list[Track] = []
        for path in _iter_audio_files(self.folder):
            if path in existing:
                continue
            t = self.add_track_file(path, log=log)
            if t is not None:
                added.append(t)
        return added

    def _manifest_add(self, vid: str, title: str, path: str) -> None:
        """Record a runtime download in the manifest so a trimmed cache can be
        restored by tools/redownload_library.py. Merge-by-id (idempotent) and
        preserve the file's existing note/cache_dir header. Call holding _add_lock.
        Best-effort: a manifest write failure must not break ingest."""
        try:
            try:
                with open(self._manifest_path) as fh:
                    data = json.load(fh)
            except (OSError, json.JSONDecodeError):
                data = {}
            tracks = data.get("tracks")
            if not isinstance(tracks, list):
                tracks = []
            if not any(isinstance(e, dict) and e.get("id") == vid for e in tracks):
                tracks.append({
                    "id": vid,
                    "title": title,
                    "filename": os.path.basename(path),
                    "url": f"https://www.youtube.com/watch?v={vid}",
                })
            data["tracks"] = tracks
            data["count"] = len(tracks)
            data.setdefault("cache_dir", os.path.basename(os.path.abspath(self.folder)))
            data.setdefault("note", "Run tools/redownload_library.py to restore audio "
                                    "into cache_dir from these URLs.")
            with open(self._manifest_path, "w") as fh:
                json.dump(data, fh, indent=2)
        except OSError:
            pass

    def stop(self) -> None:
        """Shutdown sweep: delete any ephemeral downloads that never played, so a
        run that fetched guest/auto-dig tracks doesn't leave them on the tight
        disk. They remain in the manifest, so they're re-fetchable."""
        with self._add_lock:
            leftover = [t for t in self.tracks if getattr(t, "ephemeral", False)]
        for t in leftover:
            try:
                os.remove(t.path)
            except OSError:
                pass
