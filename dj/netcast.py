"""Broadcast the live master mix to other machines on the same Wi-Fi. Open
``http://<host-ip>:<port>/listen`` on any Mac, phone or tablet on the network and
it plays — no extra software on the listener (unlike the Loopback/Soundflower
route, which needs an install on every box).

Two delivery paths share the same realtime ``feed()``:

* **WebSocket raw PCM (low-latency, preferred).** float32 blocks are converted
  to little-endian int16 and fanned straight to each WS listener. The browser
  runs a tight Web Audio jitter buffer (~120ms), so end-to-end latency is
  ~150-250ms — close enough to ride alongside the host speakers. ~1.4 Mbps per
  listener, trivial on a LAN. No encoder, so these listeners cost nothing but a
  memcpy.
* **MP3 (compatibility fallback).** A single persistent ffmpeg encodes the same
  PCM to MP3; each MP3 listener gets a fan-out queue of the encoded bytes. The
  encoder only spins up while at least one MP3 listener is connected. Browser
  ``<audio>`` buffering puts these listeners ~1-3s behind and not phase-aligned —
  fine for filling another room, not for side-by-side speakers.

Realtime-safe: ``feed()`` (called from the audio callback) is a non-blocking
enqueue on both paths that drops a block if a consumer falls behind, so a slow
network can never stall the audio thread. All the encoding/network work happens
on worker threads. Tight sample-locked multi-room sync would still need something
like Snapcast on each machine.
"""

from __future__ import annotations

import queue
import subprocess
import threading
from typing import Optional

import numpy as np

from . import CHANNELS, SAMPLE_RATE

_PCM_BACKLOG = 64        # ~3s of audio buffered toward the encoder before dropping
_CLIENT_BACKLOG = 512    # encoded-byte chunks queued per listener before dropping
_READ_SIZE = 2048        # bytes pulled from ffmpeg stdout per fan-out tick
# Raw-PCM (WebSocket) listeners get a much shorter queue: this is the
# low-latency path, so a backed-up client should drop blocks and snap back to
# live rather than accumulate seconds of delay. 16 blocks ≈ 0.75s ceiling.
_WS_BACKLOG = 16


class NetCast:
    def __init__(self, bitrate: str = "192k"):
        self.bitrate = bitrate
        self._lock = threading.Lock()
        self._clients: set[queue.Queue] = set()
        self._proc: Optional[subprocess.Popen] = None
        self._pcm: Optional[queue.Queue] = None
        self._running = False
        # Low-latency path: raw int16 PCM fanned straight out over WebSockets to
        # browsers running a Web Audio jitter buffer. Independent of the MP3
        # encoder — these clients pay no ffmpeg cost. The snapshot tuple lets the
        # audio callback iterate listeners without taking the lock (registration
        # is rare; the realtime feed must never block).
        self._ws_clients: set[queue.Queue] = set()
        self._ws_snapshot: tuple = ()

    # ---- realtime feed (called from the audio callback) ------------------
    def feed(self, block) -> None:
        """Push one master block (float32 (frames, CHANNELS)) to both paths:
        the WebSocket low-latency listeners (raw int16 PCM) and, if the MP3
        encoder is up, its stdin. No-op when nobody's listening on a path; never
        blocks the audio callback."""
        ws = self._ws_snapshot
        if ws:
            # float32 [-1,1] → little-endian int16; one conversion shared by all
            # WS listeners. A backed-up client drops its oldest block (snap-to-
            # live) rather than stalling the others or the audio thread.
            pcm16 = (np.clip(block, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            for q in ws:
                try:
                    q.put_nowait(pcm16)
                except queue.Full:
                    try:
                        q.get_nowait()
                        q.put_nowait(pcm16)
                    except (queue.Empty, queue.Full):
                        pass
        if not self._running:
            return
        pcm = self._pcm
        if pcm is None:
            return
        try:
            pcm.put_nowait(block.tobytes())
        except queue.Full:
            pass  # encoder fell behind — drop a block rather than stall audio

    # ---- listeners -------------------------------------------------------
    def add_client(self) -> queue.Queue:
        """Register a listener; spins the encoder up on the first one."""
        with self._lock:
            self._start_locked()
            q: queue.Queue = queue.Queue(maxsize=_CLIENT_BACKLOG)
            self._clients.add(q)
            return q

    def remove_client(self, q: queue.Queue) -> None:
        """Drop a listener; tears the encoder down once the last one leaves."""
        with self._lock:
            self._clients.discard(q)
            if not self._clients:
                self._stop_locked()

    def listeners(self) -> int:
        with self._lock:
            return len(self._clients)

    # ---- low-latency WebSocket listeners ---------------------------------
    def add_ws_client(self) -> queue.Queue:
        """Register a raw-PCM listener. No encoder is involved, so this is just a
        queue the audio callback fans int16 blocks into."""
        with self._lock:
            q: queue.Queue = queue.Queue(maxsize=_WS_BACKLOG)
            self._ws_clients.add(q)
            self._ws_snapshot = tuple(self._ws_clients)
            return q

    def remove_ws_client(self, q: queue.Queue) -> None:
        with self._lock:
            self._ws_clients.discard(q)
            self._ws_snapshot = tuple(self._ws_clients)

    def ws_listeners(self) -> int:
        return len(self._ws_snapshot)

    def stop(self) -> None:
        with self._lock:
            self._clients.clear()
            self._ws_clients.clear()
            self._ws_snapshot = ()
            self._stop_locked()

    # ---- encoder lifecycle (call holding self._lock) ---------------------
    def _start_locked(self) -> None:
        if self._running:
            return
        self._proc = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS), "-i", "pipe:0",
             "-f", "mp3", "-b:a", self.bitrate, "pipe:1"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, bufsize=0,
        )
        self._pcm = queue.Queue(maxsize=_PCM_BACKLOG)
        self._running = True
        proc, pcm = self._proc, self._pcm
        threading.Thread(target=self._feed_loop, args=(proc, pcm), daemon=True).start()
        threading.Thread(target=self._pump_loop, args=(proc,), daemon=True).start()

    def _stop_locked(self) -> None:
        self._running = False
        proc, self._proc = self._proc, None
        self._pcm = None
        if proc is not None:
            try:
                proc.stdin.close()
            except (OSError, ValueError):
                pass
            try:
                proc.terminate()
            except OSError:
                pass

    # ---- worker threads --------------------------------------------------
    def _feed_loop(self, proc, pcm) -> None:
        """Drain queued PCM into ffmpeg's stdin."""
        while self._running and self._proc is proc:
            try:
                data = pcm.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                proc.stdin.write(data)
            except (BrokenPipeError, ValueError, OSError):
                break

    def _pump_loop(self, proc) -> None:
        """Read encoded MP3 off ffmpeg's stdout and fan it out to every listener.
        A slow listener drops its oldest chunk rather than stalling the others."""
        out = proc.stdout
        while self._running and self._proc is proc:
            chunk = out.read(_READ_SIZE)
            if not chunk:
                break
            with self._lock:
                for q in self._clients:
                    try:
                        q.put_nowait(chunk)
                    except queue.Full:
                        try:
                            q.get_nowait()
                            q.put_nowait(chunk)
                        except (queue.Empty, queue.Full):
                            pass
