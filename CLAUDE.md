# CLAUDE.md — vibe-dj

Project context for resuming work in a new thread.

## What this is
An **AI DJ that beatmatches a music library and steers the set by reading the crowd**.
A webcam measures room motion → a `0..1` "energy" signal → the autopilot picks the
next track to match where the room is heading, beatmatches it, and crossfades.

Repo: https://github.com/sameeeeeeep/vibe-dj (public, MIT)

## ▶️ NEXT STEPS (agreed plan — start here)
1. **Web dashboard UI** (chosen over a terminal UI). Show: deck A/B (now-playing + cued,
   BPM/energy/playhead), live **crowd-cam thumbnail + energy meter**, the **upcoming buffer**
   (the lookahead tracks), crossfader state, and controls (skip, force-transition, nudge energy).
   Engine already runs in-process, so the UI is a thin state-broadcast layer
   (FastAPI + WebSocket, or stdlib + SSE to stay dep-light).
2. **Real webcam path** — fold into the UI work (the dashboard surfaces it anyway).
   `dj/crowd.py::_camera_loop` is written but UNTESTED. Verify `cv2.VideoCapture(0)` opens,
   tune the motion normalization, and show the frame in the dashboard.
3. **Mixing upgrades — BOTH are wanted:**
   - **Bass-swap EQ crossfade** (do first, fastest big win): 3-band EQ per deck (scipy biquads)
     applied in `dj/mixer.py` (`mix`/`_advance_crossfade`). During a transition: fade out
     track-1 lows → fade in track-2 lows → fade out track-1 highs (from `artificial_dj`).
   - **Section-aware transitions**: detect intro/buildup/outro so we mix at the right point,
     not just at track-end. Add section/cue fields to `dj/analysis.py::Analysis`; have
     `dj/controller.py` trigger on cue points instead of only `remaining_sec`.

## Dev setup
- **Python 3.13** (NOT 3.14 — DSP wheels missing there). venv at `./.venv`.
- System deps: `ffmpeg`, `portaudio` (`brew install ffmpeg portaudio`).
- Install: `./.venv/bin/python -m pip install -r requirements.txt`
- Run (local): `./.venv/bin/python main.py ~/Music/folder`
- Run (YouTube): `./.venv/bin/python main.py --youtube "PLAYLIST_URL" --buffer 5`
- Headless test (no audio/cam): `./.venv/bin/python main.py demo_tracks --dry-run --simulate-crowd --duration 60`
- Make test tracks: `./.venv/bin/python tools/gen_demo_tracks.py demo_tracks`

## Architecture
```
main.py            CLI entrypoint + live status line
dj/
  audio_io.py      decode any format via ffmpeg subprocess → numpy (no libsndfile/librosa)
  analysis.py      BPM + beat-grid + energy, numpy/scipy only (spectral-flux onset →
                   autocorr w/ log-Gaussian tempo prior → comb-filter phase)
  library.py       static folder scan + analysis cache (.dj_cache.json) + energy ranking.
                   score_energy() is shared with the pool. release() = no-op (don't delete user files)
  deck.py          one deck: in-RAM samples + fractional playhead + varispeed read (beatmatch)
  mixer.py         two decks, equal-power beat-aligned crossfade, sounddevice OR dummy backend
  crowd.py         webcam motion-energy → 0..1 (EMA + adaptive scale); simulator fallback
  controller.py    autopilot: read crowd → target_energy (slew-limited) → pick next → transition
  youtube_source.py yt-dlp wrapper: list_entries / download_one / fetch
  pool.py          TrackPool: streaming buffer for YouTube. Duck-types Library
                   (tracks/load_audio/release). Keeps `buffer` tracks on disk, deletes after
                   play, loops the playlist. Background filler thread.
tools/gen_demo_tracks.py  synthetic four-on-the-floor tracks at known BPMs
```

## Key design decisions (don't undo without reason)
- **No librosa/numba.** Analysis is numpy/scipy; decode is ffmpeg subprocess. This is why it
  installs on new Python. Keep new DSP in that style.
- **Beatmatch = varispeed** (pitch shifts slightly, capped ±8% via `RATE_LIMIT` in mixer.py).
  The keylock time-stretch (Rubber Band) is a future upgrade, not done.
- **Library keeps only metadata in RAM**; audio is decoded on demand when a track loads onto a
  deck. Two decks decoded at a time (~85MB each); pool tracks are just files + metadata.
- **Controller is backend-agnostic**: it talks to `library` via `tracks` / `load_audio` /
  `release`, so `Library` (folder) and `TrackPool` (YouTube) are interchangeable.
- **Energy policy** (`controller.target_energy`): move toward the crowd, capped per transition
  (`max_step=0.25`) so the set ramps instead of whiplashing. (Earlier +0.12 nudge was a bug —
  a hot crowd didn't escalate; fixed.)

## Status — verified vs. not
Verified: BPM within ~1 BPM (synthetic + real YouTube audio); energy ranking; crowd→selection;
beat-aligned crossfade (headless); YouTube streaming buffer keeps disk pinned at `buffer` files
while the playlist loops.
NOT tested: real webcam; live audio output through real speakers (only dry-run validated).

## Constraints / preferences
- **Disk is tight on this machine.** Delete reproducible artifacts (yt-dlp caches, demo tracks,
  scratch) once they've served their purpose. `.gitignore` already excludes them. Prefer
  self-cleaning designs (the pool already deletes played tracks).
- Don't commit unless asked.

## References for the mixing work
- `pnlong/artificial_dj` — bass-swap transition (lows then highs), section detection
  ("where to mix > how to mix"), key detection. Sub-repos: determine_tempo/key/sections.
- `mixxxdj/mixxx` — keylock time-stretch (Rubber Band/SoundTouch), beat grids, EQ kill,
  crossfader curves. Reference for the DJ UI layout. (C++/Qt — borrow concepts, don't port.)
