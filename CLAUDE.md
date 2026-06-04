# CLAUDE.md — vibe-dj

Project context for resuming work in a new thread.

## What this is
An **AI DJ that beatmatches a music library and steers the set by reading the crowd**.
A webcam measures room motion → a `0..1` "energy" signal → the autopilot picks the
next track to match where the room is heading, beatmatches it, and crossfades.

Repo: https://github.com/sameeeeeeep/vibe-dj (public, MIT)

## ✅ DONE (recent)
- **LAN audio broadcast (listen on other Macs/phones over Wi-Fi)** — `dj/netcast.py` `NetCast`,
  two delivery paths sharing one realtime-safe `feed()` (called from the audio callback; drops a
  block rather than ever stalling audio). Tapped in `mixer.mix()` (`self.netcast`). `main.py --lan`
  binds `0.0.0.0` + prints the LAN URL (also exposes the control UI — warned at startup; trusted
  Wi-Fi only). `_lan_ip()` finds the route-out address via a UDP connect (no packets). Topbar/
  listen-page ON-AIR badge sums both paths' listener counts (`snapshot()['broadcast']`).
  `.claude/launch.json` dj-youtube adds `--lan`.
  - **Low-latency (preferred): raw int16 PCM over a WebSocket.** `feed()` converts the float32
    block to little-endian int16 and fans it to each WS listener (lock-free `_ws_snapshot`, short
    `_WS_BACKLOG=16` drop-oldest queue → snap-to-live, no encoder cost). Dashboard `GET /ws-audio`
    does the RFC6455 handshake by hand (stdlib has no WS server; SHA1+base64 accept-key, `_WS_GUID`)
    then writes each block as one unmasked binary frame (`_ws_frame`, opcode 0x82; we only ever
    send). The `/listen` page leads with a **"TAP TO GO LIVE"** Web Audio player (`goLive`/`onPCM`):
    decodes int16→float32 into `AudioBuffer`s scheduled against `AudioContext.currentTime` with a
    tight jitter buffer (`TARGET=0.12s`, `CEIL=0.40s`); underrun rebuilds the cushion, overbuffer
    drops a block to snap back. Auto-reconnect on WS close; live "≈ N ms behind live" readout.
    Verified live (headless Chromium): handshake → 101, PCM flows continuously, ~190-227 ms behind,
    no clicks, no console errors, listener count = 1.
  - **Compatibility fallback: MP3.** A persistent ffmpeg (`f32le → mp3 192k`) encodes the same PCM;
    icecast-style HTTP fan-out, one drop-oldest queue per listener; encoder spins up only while ≥1
    MP3 listener is connected, tears down on the last disconnect. `GET /stream.mp3` (Connection:
    close, no Content-Length, body delimited by hangup). Demoted on `/listen` to a collapsed
    "compatibility mode" `<details>` (`preload="none"` so ffmpeg only starts if actually used).
    ~1-3 s behind, not phase-locked — fine for filling another room. Verified live: emits real MP3
    (ID3 + MPEG frames, ~192k). Tight sample-locked multi-room sync would still need Snapcast.
- **Jukebox (LAN guests request tracks into the queue)** — the `/listen` page carries a REQUEST
  panel: a YouTube **search** box (`GET /jukebox/search?q=` → `youtube_source.list_entries`
  `ytsearchN:`, metadata only, no download) AND a browsable **loaded-crate** list, plus an UP NEXT
  view, all polling `GET /jukebox` (`Dashboard.jukebox_state()`). A pick POSTs `/jukebox/request`
  (`Dashboard.jukebox_request`): a crate `tid` calls `controller.queue_add(id(t))` instantly
  (disk-safe, no download); a YouTube `vid` worker-threads `library.add_url(url)` then
  `queue_add` each result. **Auto-queue, no DJ approval gate** (user's choice). Guardrails honoring
  the tight disk: `_JUKEBOX_MAX_INFLIGHT=4` concurrent downloads + dedupe by video id (returns
  "already fetching"/"DJ's busy"). Verified live: crate request lands in up-next, search returns 6
  hits, vid request downloads+queues, dedupe fires, toast + ✓ states render on the page.
- **Self-cleaning runtime downloads (folder mode)** — guest/auto-dig YouTube tracks no longer pile
  up in `yt_cache`. `dj/library.py`: any track whose filename carries an 11-char YouTube id
  (`_VID_RE`) is added through `add_analyzed()` as `Track.ephemeral=True` and recorded in
  `yt_library_manifest.json` (merge-by-id, beside the cache folder = repo root, matching
  `tools/redownload_library.py`). `release()` (was a no-op) now deletes the on-disk file + drops the
  track once it has actually played — gated on `play_count >= 1` so a cued-but-unplayed request the
  controller *unstages* (which decrements play_count to 0) is preserved, not deleted before it
  plays. `stop()` sweeps any ephemeral downloads still in the set at shutdown. The user's own scanned
  folder files (no id, created directly in `scan()`, not via `add_analyzed`) stay `ephemeral=False`
  and are never touched. Everything stays re-fetchable via the manifest. Verified headless: ephemeral
  flagging + manifest record, unplayed-preserve, played-delete, user-file-untouched, shutdown sweep.
- **Transition styles (6)** — beyond the single equal-power+bass-swap blend. `dj/mixer.py`
  `TRANSITIONS = (smooth, bass_swap, filter, cut, echo, brake)`; `start_transition(dur, kind)` +
  a per-kind dispatch in `_advance_crossfade`. `smooth`=equal-power+bass swap; `bass_swap`=both
  grooves coexist at volume, basslines trade, outgoing drops late; `filter`=HP-sweep the outgoing
  / open the incoming (drives the new 3-band `eq_auto` automation that replaced low-only
  `bass_auto`); `cut`=~35ms downbeat swap; `echo`=feedback-delay throw (`_FeedbackDelay`, one beat,
  vectorised block-wise so no per-sample loop in the callback) rung out in `mix()`; `brake`=tape-
  stop (ramp outgoing `rate`→0, incoming held frozen at its cue then slams in). Controller
  `transition_kind` ("auto" or forced) + `_choose_kind()` picks by energy delta (lift→cut/echo,
  drop→filter/brake, cruise→smooth/bass_swap/filter) with a rotor so it varies. Dashboard
  TRANSITION-STYLE picker (`transition_kind` cmd, `renderTx`, live readout). Verified headless
  (all 6 render finite, complete, swap, no clip, eq_auto resets, brake releases rate) + live.
- **Drum/beat loop layer ("beats" pad)** — `dj/beats.py` `BeatMachine`: numpy-synth voices
  (kick/clap/hats/snare/perc, no samples on disk), 16-step pattern, bar sized to the live deck's
  heard tempo so it's tempo-matched + beat-locked (downbeat lock for free), intensity follows the
  song vibe (AUTO) or manual. Summed in `mixer.mix()`; controller pushes `set_vibe`; dashboard
  BEATS panel (toggle / AUTO·MANUAL / VIBE / MIX). Verified headless + live.
- **Selectable beat styles + AUTO** — `dj/beats.py` now carries 6 named drum patterns
  `STYLES = (four_floor, house, techno, breakbeat, trap, afro)` (each a `_p_*` builder; steps may
  be fractional, e.g. 32nd-note trap rolls) plus `style="auto"` where `_auto_style()` maps the
  song vibe to a fitting groove (afro→house→four_floor→techno; breakbeat/trap are manual-only).
  Manual pick always wins. `set_style()` / `effective_style()`; `state()` exposes `style` +
  `style_eff`; render cache rebuilds on style change. Dashboard BEATS **style picker** (7 buttons,
  `beats_style` cmd; active = chosen, outline = AUTO's current pick). Verified headless + live.
- **Sound-FX rack (paste a link → triggerable pad)** — `dj/fx.py` `FXRack`: paste any URL yt-dlp
  understands (public Instagram reel, YouTube, TikTok, direct media) → download to a tempdir →
  ffmpeg `decode()` → `_trim_oneshot` (skip lead-in silence, cap `FX_MAX_SEC=6s`, normalize, 6ms
  click-guard fades) → an in-RAM one-shot pad; the source file is deleted right after decode
  (self-cleaning, disk-tight). Up to `MAX_SLOTS=8` pads, oldest idle evicted at cap; `trigger(idx)`
  (retrigger chokes the prior hit), master `level`. Summed in `mixer.mix()`. Worker-thread load so
  the POST returns instantly. Dashboard FX panel (url input + LOAD, pad grid with fire-flash, LVL).
  Private/login-walled media fails to the log — never prompts for credentials. Verified headless +
  live (stubbed download).
- **MIDI melody layer (.mid → synth voices over the track)** — `dj/midi.py`: a tiny Standard MIDI
  File reader (`parse_midi`, stdlib bytes only, NO mido/pretty_midi) that positions notes in
  *beats* (tick/PPQ), independent of the file's own tempo, so the layer slaves to the live deck;
  GM drum channel 9 skipped. `MelodyLayer` synthesises the notes into one beat-locked loop
  (poly osc: sin + 0.25·2nd + 0.12·3rd harmonic, exp-decay pluck env), tempo-matched the same way
  the beats layer is; `transpose` ±24 semis just rebuilds the loop. Summed in `mixer.mix()`.
  Dashboard MELODY panel (.mid discovery dropdown + path load, toggle, SEMI ± transpose, MIX).
  Verified headless + live.
- **Headphone cue / second output + custom cue points + master-pin** — monitor feed on a second
  device (`--cue-device` or dashboard MONITOR panel) reads decks via `read_preview()` on its own
  playheads (never disturbs the master); CUED/LIVE/MASTER source + AUDITION (renders the real
  blend to the monitor only). Draggable per-deck mix-in/out markers (`Deck.mix_*_override`,
  `set_mix_in/out`). `--master-device` pins the room output so making AirPods the default doesn't
  drag the master onto them. NOTE: macOS only exposes AirPods to CoreAudio once they're actively
  routed; PortAudio caches devices at process init, so newly-activated devices need a restart.
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
1. **Audition the new layers + transition styles by ear** on a real set through speakers. Logic +
   audio are verified headless and the engine runs live clean, but none of these have been
   critically tuned by listening yet: the 6 transition styles (esp. `echo` tail level/feedback,
   `brake` halt timing, `filter` sweep shape), the beats pad mix + the 6 drum styles (do they
   read as house/techno/afro/etc?), the MIDI melody synth tone/level, and FX one-shot loudness.
   Tune constants in `dj/mixer._advance_crossfade` / `_KIND_SECONDS` / `_FeedbackDelay`,
   `dj/beats.py` (per-voice levels + the `_p_*` builders), `dj/midi.py` (`_synth_loop` harmonics
   / envelope / `0.18` vel scale), and `dj/fx.py` (`_trim_oneshot` normalize target).
2. **Looping / loop-rolls** — sample-index math on `dj/deck.py` (loop in/out over N beats). Lets
   the autopilot *stretch a breakdown* when the room cools — crowd-driven arrangement, not just
   selection. Medium.
3. **Filter sweep as a standalone live-FX knob** — the `filter` *transition* exists, but a
   per-deck moving-cutoff sweep the DJ can ride mid-track (buildups) is still open. A real
   LP/HP biquad in `dj/eq.py` would be crisper than the current EQ-band approximation.
4. **Real webcam path** — thumbnail plumbing DONE (`CrowdSensor.last_jpeg` + `/frame.jpg`), still
   UNTESTED on hardware: verify `cv2.VideoCapture(0)` opens, tune motion normalization in
   `dj/crowd.py::_camera_loop`. Simulated mode shows a "no camera" placeholder, as expected.
5. Stretch: hot cues + live re-arrangement; downbeat detection to sharpen the phrase grid (today
   it assumes `beat_offset` is a downbeat); keylock time-stretch (hard).

DONE already (were on this list): in-track EQ kills (per-deck low/mid/high kill buttons +
`Mixer.set_eq`/`toggle_kill`), filter sweep (now a `filter` transition style), echo/delay throw
(now an `echo` transition style).

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
                   (GET / DJ page, GET /events snapshots, GET /frame.jpg cam, POST /control;
                   GET /listen guest player, GET /ws-audio low-latency PCM WebSocket (hand-rolled
                   RFC6455 handshake + _ws_frame), GET /stream.mp3 MP3 fallback; jukebox: GET
                   /jukebox state, GET /jukebox/search, POST /jukebox/request → auto-queue)
  netcast.py       NetCast: broadcasts the master mix to LAN listeners on two paths off one
                   realtime-safe feed() (fed from mixer.mix()): WebSocket raw int16 PCM (low-
                   latency, no encoder, lock-free fan-out + snap-to-live) and a persistent-ffmpeg
                   PCM→MP3 icecast-style stream (compat fallback, encoder runs only while ≥1
                   MP3 listener connected).
  audio_io.py      decode any format via ffmpeg subprocess → numpy (no libsndfile/librosa)
  analysis.py      BPM + beat-grid + energy, numpy/scipy only (spectral-flux onset →
                   autocorr w/ log-Gaussian tempo prior → comb-filter phase). Also section
                   detection (intro_end/outro_start from a smoothed loudness envelope) + a
                   phrase grid → mix_in_sec()/mix_out_sec() phrase-aligned cue points.
  eq.py            ThreeBandEQ: scipy-biquad low/mid/high, bands by subtraction (unity =
                   transparent), stateful across blocks. Used for bass-swap, in-track EQ kills,
                   and the filter-transition sweep (via Mixer.eq_auto).
  beats.py         BeatMachine: procedural numpy drum-loop layer ("beats" pad), tempo-matched +
                   beat-locked to the live deck, intensity follows song vibe. 6 selectable styles
                   (four_floor/house/techno/breakbeat/trap/afro) + AUTO (_auto_style by vibe);
                   manual pick wins. Summed in mix().
  midi.py          parse_midi(): minimal Standard MIDI File reader (stdlib bytes, no mido), notes
                   positioned in beats so the layer slaves to the deck's tempo. MelodyLayer: poly
                   synth, one beat-locked loop, transpose ±24. Summed in mix().
  fx.py            FXRack: paste a URL → yt-dlp download → ffmpeg decode → trimmed one-shot pad
                   (source deleted after decode, self-cleaning). Up to 8 triggerable pads + master
                   level. Summed in mix().
  library.py       static folder scan + analysis cache (.dj_cache.json) + energy ranking.
                   score_energy() is shared with the pool. release() = no-op (don't delete user files)
  deck.py          one deck: in-RAM samples + fractional playhead + varispeed read (beatmatch)
  mixer.py         two decks + per-deck 3-band EQ (eq_manual × eq_auto). 6 transition styles
                   (TRANSITIONS / start_transition(dur,kind) / _advance_crossfade dispatch):
                   smooth, bass_swap, filter, cut, echo (_FeedbackDelay), brake. Beat-aligned,
                   mixes incoming in at its mix-in cue. Second monitor output + master-pin.
                   Drum/melody/fx layers summed on top. sounddevice/dummy backend
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
phrase-snapped cues, controller fires at outro & brings incoming in past its intro);
**selectable beat styles** (headless: 6 patterns render, AUTO picks afro/house/four_floor/techno
by vibe, manual override wins; live dashboard switched style→techno); **FX rack** (headless:
trim/trigger/retrigger, capacity eviction, add_from_url via stubbed download, summed in mix;
live: bad link rejected to log); **MIDI melody** (headless: SMF parse of a 4-note arpeggio at
beats 0/1/2/3, beat-locked render, transpose ±, summed in mix; live: `[midi] loaded` clean).
Deployed: the live YouTube set on `--dashboard --port 8765` runs the bass-swap EQ +
section-aware mixing code through real speakers (crowd simulated). It comes up clean — buffers
the playlist, starts on track 1, dashboard serves the manual-crowd UI, no tracebacks.
NOT tested: real webcam; everything *by ear* on a real set — EQ bass-swap, section cues, the 6
transition styles, the 6 beat styles, the MIDI melody synth, and FX one-shots are all logic-
verified (and the engine runs live clean) but not yet critically auditioned through speakers.
The port-8765 background set still runs the OLD transition-only code — restart it to pick up the
beat-styles / FX / melody layers (left as-is; it's the user's live set).

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
