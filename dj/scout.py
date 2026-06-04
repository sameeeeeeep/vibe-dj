"""Auto crate-digger: keep the queue stocked with tracks that fit the vibe.

When the pool of fresh (unplayed) tracks runs low, the scout pulls more from
YouTube's own "Mix" — the ``RD<id>`` autoplaylist seeded by the current live
track. YouTube's recommender already clusters by vibe, so the candidates come
in on-theme; the scout then keeps only the ones whose analysed tempo sits near
the live track (so the beatmatched blends stay clean) and deletes the rest, so
disk stays bounded. Keepers are spliced in via ``Library.add_track_file`` /
``add_analyzed``, so the autopilot picks them up — by energy — with no other
wiring.

Energy is rank-normalised across the whole set, so it isn't a meaningful filter
on a single fresh download; tempo is the hard gate here, and the autopilot's
own energy scoring decides *which* of the vibe-seeded crate actually plays next.

Only runs against the static folder ``Library`` (which exposes ``add_analyzed``);
the streaming ``TrackPool`` already fills itself, so the scout no-ops there.
"""

from __future__ import annotations

import os
import re
import threading
from typing import Callable, Optional

from . import SAMPLE_RATE
from .analysis import analyze
from .audio_io import decode
from .library import Track
from . import youtube_source as yt

_VID_RE = re.compile(r"\[([A-Za-z0-9_-]{11})\]")


class Scout:
    def __init__(
        self,
        library,
        controller,
        cache_dir: str,
        min_fresh: int = 4,
        batch: int = 6,
        bpm_window: float = 0.18,
        interval: float = 20.0,
        warmup: float = 8.0,
        log: Optional[Callable[[str], None]] = None,
    ):
        self.library = library
        self.controller = controller
        self.cache_dir = cache_dir
        self.min_fresh = max(1, min_fresh)
        self.batch = max(1, batch)
        self.bpm_window = bpm_window
        self.interval = interval
        self.warmup = warmup
        # Prefer the controller's teed event log so digs show in the dashboard.
        self.log = log or getattr(controller, "log", None) or (lambda m: None)

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---- lifecycle -------------------------------------------------------
    def start(self) -> "Scout":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def dig_now(self, count: int = 3) -> Optional[str]:
        """Fire a single on-demand dig for `count` vibe-matched tracks seeded by
        the live song, off the background cadence — the dashboard's "fetch
        similar" button. Runs on a worker thread so the caller returns at once.
        Returns a short status to echo, or None if nothing's playing to seed
        from. Works even when the background auto-dig loop was never started."""
        live = self.controller.deck_tracks.get(self.controller.mixer.current)
        if live is None:
            self.log("[scout] nothing playing to seed a dig")
            return None
        n = max(1, int(count))

        def work() -> None:
            try:
                self._dig(n)
            except Exception as exc:  # noqa: BLE001 - surface, don't crash the server
                self.log(f"[scout] dig failed: {exc}")

        threading.Thread(target=work, daemon=True).start()
        return f"digging {n} similar to {live.name[:40]}"

    def _loop(self) -> None:
        self._stop.wait(self.warmup)        # let the set get going first
        while not self._stop.is_set():
            try:
                if self._fresh_count() < self.min_fresh:
                    self._dig()
            except Exception as exc:  # noqa: BLE001 - a bad dig shouldn't kill the loop
                self.log(f"[scout] error: {exc}")
            self._stop.wait(self.interval)

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def _vid(path: str) -> Optional[str]:
        m = _VID_RE.search(os.path.basename(path or ""))
        return m.group(1) if m else None

    def _fresh_count(self) -> int:
        """Unplayed tracks not currently on a deck — the 'crate' the autopilot
        still has to choose from. We top up only when this dips below min_fresh,
        which bounds how much the scout ever downloads."""
        loaded = {id(t) for t in self.controller.deck_tracks.values() if t is not None}
        return sum(1 for t in self.library.tracks
                   if t.play_count == 0 and id(t) not in loaded)

    def _discard(self, path: str) -> None:
        try:
            os.remove(path)
        except OSError:
            pass

    # ---- the dig ---------------------------------------------------------
    def _dig(self, count: Optional[int] = None) -> None:
        live = self.controller.deck_tracks.get(self.controller.mixer.current)
        if live is None:
            return
        keep_cap = max(1, count if count is not None else self.batch)
        target = self.controller.current_target()
        live_bpm = self.controller.mixer.live_deck.effective_bpm

        have_ids = {self._vid(t.path) for t in self.library.tracks}
        have_ids.discard(None)
        seed_id = self._vid(live.path)
        if seed_id:
            url = f"https://www.youtube.com/watch?v={seed_id}&list=RD{seed_id}"
            how = f"mix of {live.name[:30]}"
        else:
            q = re.sub(r"\s*\[[^\]]*\]\s*", " ", live.name).strip() or live.name
            url = f"ytsearch{keep_cap * 2}:{q}"
            how = f"search '{q[:30]}'"
        self.log(f"[scout] digging {keep_cap} — {how}  (target {target:.2f}, ~{live_bpm:.0f} BPM)")

        try:
            entries = yt.list_entries([url], limit=max(self.batch, keep_cap) * 3, log=self.log)
        except Exception as exc:  # noqa: BLE001
            self.log(f"[scout] list failed: {exc}")
            return

        kept = 0
        for e in entries:
            if self._stop.is_set() or kept >= keep_cap:
                break
            eid = e.get("id")
            if not eid or eid == seed_id or eid in have_ids:
                continue
            path = yt.download_one(e, self.cache_dir, log=self.log)
            if not path:
                continue
            try:
                an = analyze(decode(path, sr=SAMPLE_RATE), sr=SAMPLE_RATE)
            except Exception as exc:  # noqa: BLE001
                self.log(f"[scout] analyse failed: {exc}")
                self._discard(path)
                continue
            # Tempo gate: keep only what blends cleanly with the live track.
            if live_bpm > 0 and abs(an.bpm - live_bpm) / live_bpm > self.bpm_window:
                self.log(f"[scout] skip {an.bpm:.0f} BPM off-tempo  "
                         f"{os.path.basename(path)[:28]}")
                self._discard(path)
                continue
            track = Track(path=path, name=e.get("title") or os.path.basename(path), analysis=an)
            added = self.library.add_analyzed(track)
            if added is None:                 # already present — drop the dup file
                self._discard(path)
                continue
            have_ids.add(eid)
            kept += 1
            self.log(f"[scout] +{added.name[:32]}  {added.bpm:.0f} BPM  "
                     f"energy {added.energy:.2f}")

        if kept == 0:
            self.log("[scout] nothing new fit the vibe this pass")
