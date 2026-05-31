# CLAUDE.md — vibe-dj

Project context for resuming work in a new thread.

## What this is
An **AI DJ that beatmatches a music library and steers the set by reading the crowd**.
A webcam measures room motion → a `0..1` "energy" signal → the autopilot picks the
next track to match where the room is heading, beatmatches it, and crossfades.

Repo: https://github.com/sameeeeeeep/vibe-dj (public, MIT)

## ✅ DONE (recent)
- **Web dashboard UI** — `dj/dashboard.py` (stdlib `http.server` + SSE, no new deps). Run with
  `--dashboard` (default `--port 8765`). Deck A/B (now-playing + cued, BPM/energy/playhead),
  crowd-cam thumbnail + energy/target meters, upcoming buffer, crossfader, and controls (skip,
  force-transition, nudge energy ±0.1, AUTO/MANUAL crowd toggle + drag-to-set vibe fader).
- **Manual crowd override** — DJ can dictate the room vibe instead of the sensor.
  `Controller._crowd_override` / `effective_crowd()` / `set_crowd_manual()` / `set_crowd_energy()`;
  dashboard AUTO|MANUAL toggle + vibe fader (`crowd_manual` / `crowd_set` control cmds).
- **Bass-swap EQ crossfade** — `dj/eq.py` `ThreeBandEQ` (scipy biquads; bands by subtraction so
  unity gain is bit-exact transparent). Wired into `dj/mixer.py`: per-deck `eqs` + `bands`;
  `_advance_crossfade` trades the lows over `[BASS_SWAP_START, BASS_SWAP_END]` so only one
  bassline plays at a time; mids/highs ride the equal-power fade. Verified headless.
- **Section-aware transitions** — `dj/analysis.py::Analysis` now has `intro_end`/`outro_start`
  (from a smoothed loudness envelope, reusing the STFT) + a `phrase_period` grid, exposed as
  phrase-aligned `mix_in_sec()` / `mix_out_sec()`. `mixer.start_transition` brings the incoming
  track in at its mix-in cue (skips the intro); `controller.tick` fires the crossfade at the
  live track's outro cue (with the end-of-track trigger kept as fallback). Verified headless.

## ▶️ NEXT STEPS (pick one)
1. **In-track EQ kills + dashboard controls** (smallest next win — EQ infra already exists):
   expose per-deck low/mid/high cut that the DJ (dashboard buttons) and/or the autopilot can
   fire mid-track, by driving the existing `Mixer.bands`. Add `Controller` methods + `/control`
   cmds + UI. Bass/mid/high "kill" buttons per deck.
2. **Filter sweep (LP/HP)** — one moving-cutoff biquad per deck for buildups/transitions; big
   "live FX" feel, cheap. Add to `dj/eq.py` (or a sibling), drive from `mixer`/dashboard.
3. **Looping / loop-rolls** — sample-index math on `dj/deck.py` (loop in/out over N beats). Lets
   the autopilot *stretch a breakdown* when the room cools — crowd-driven arrangement, not just
   selection. Medium.
4. **Real webcam path** — thumbnail plumbing DONE (`CrowdSensor.last_jpeg` + `/frame.jpg`), still
   UNTESTED on hardware: verify `cv2.VideoCapture(0)` opens, tune motion normalization in
   `dj/crowd.py::_camera_loop`. Simulated mode shows a "no camera" placeholder, as expected.
5. Stretch: hot cues + live re-arrangement; echo/delay FX throw; downbeat detection to sharpen
   the phrase grid (today it assumes `beat_offset` is a downbeat); keylock time-stretch (hard).

## Dev setup
- **Python 3.13** (NOT 3.14 — DSP wheels missing there). venv at `./.venv`.
- System deps: `ffmpeg`, `portaudio` (`brew install ffmpeg portaudio`).
- Install: `./.venv/bin/python -m pip install -r requirements.txt`
- Run (local): `./.venv/bin/python main.py ~/Music/folder`
- Run + dashboard: `./.venv/bin/python main.py ~/Music/folder --dashboard` → http://127.0.0.1:8765
- Run (YouTube): `./.venv/bin/python main.py --youtube "PLAYLIST_URL" --buffer 5`
- Headless test (no audio/cam): `./.venv/bin/python main.py demo_tracks --dry-run --simulate-crowd --duration 60`
- Make test tracks: `./.venv/bin/python tools/gen_demo_tracks.py demo_tracks`

## Architecture
```
main.py            CLI entrypoint + live status line (--dashboard starts the web UI)
dj/
  dashboard.py     stdlib HTTP + SSE state-broadcast & control layer over the live engine
                   (GET / page, GET /events snapshots, GET /frame.jpg cam, POST /control)
  audio_io.py      decode any format via ffmpeg subprocess → numpy (no libsndfile/librosa)
  analysis.py      BPM + beat-grid + energy, numpy/scipy only (spectral-flux onset →
                   autocorr w/ log-Gaussian tempo prior → comb-filter phase). Also section
                   detection (intro_end/outro_start from a smoothed loudness envelope) + a
                   phrase grid → mix_in_sec()/mix_out_sec() phrase-aligned cue points.
  eq.py            ThreeBandEQ: scipy-biquad low/mid/high, bands by subtraction (unity =
                   transparent), stateful across blocks. Used for bass-swap + (future) kills.
  library.py       static folder scan + analysis cache (.dj_cache.json) + energy ranking.
                   score_energy() is shared with the pool. release() = no-op (don't delete user files)
  deck.py          one deck: in-RAM samples + fractional playhead + varispeed read (beatmatch)
  mixer.py         two decks, equal-power beat-aligned crossfade + per-deck 3-band EQ; bass-swap
                   during transitions; mixes incoming in at its mix-in cue. sounddevice/dummy backend
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
- **EQ bands by subtraction** (`dj/eq.py`): `mid = input - low - high`, so the three bands sum
  back to the input exactly and unity gain is transparent — the EQ only colors when a band is
  actually cut. Filter state is carried across blocks (no clicks). Keep this property.
- **Mix cues are phrase-aligned** (`analysis.mix_in_sec`/`mix_out_sec`): structure detection is
  coarse + heuristic; snapping to the phrase grid is what makes it musical. No-structure tracks
  return intro=0 / outro=duration so the controller cleanly falls back to start/end behavior.

## Status — verified vs. not
Verified: BPM within ~1 BPM (synthetic + real YouTube audio); energy ranking; crowd→selection;
beat-aligned crossfade (headless); YouTube streaming buffer keeps disk pinned at `buffer` files
while the playlist loops; **web dashboard** (headless + real browser: SSE snapshots, controls
mutate engine state, transition/beatmatch/buffer all render); **live audio output** (running a
real Lane 8/Elderbrook YouTube set through speakers); **bass-swap EQ** (headless: lows trade,
unity transparent, no clipping); **section-aware mixing** (headless: intro/outro detected,
phrase-snapped cues, controller fires at outro & brings incoming in past its intro).
NOT tested: real webcam; EQ bass-swap + section cues *by ear* on a real set (logic verified, not
yet auditioned through speakers).

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
