# vibe-dj

An AI DJ that beatmatches a music library and **steers the set by reading the crowd**. A webcam measures how much the room is moving; when energy rises the DJ builds, when it cools it eases off — picking the next track, beatmatching it, and crossfading automatically.

> Status: working prototype. Beatmatching, energy-based selection, the crowd→set feedback loop, and a self-cleaning YouTube source are all built and tested. See [Status](#status) for what's verified vs. still on the bench.

## How it works

Three subsystems wired together by an autopilot:

1. **Audio / mixing engine** — analyzes each track (BPM, beat grid, energy), runs two decks, beatmatches via varispeed, and does an equal-power, beat-aligned crossfade.
2. **Crowd sensing** — a webcam loop turns inter-frame motion ("how much is the room moving") into a self-calibrating `0..1` energy signal. Falls back to a simulator with no camera.
3. **The feedback loop** — the controller reads crowd energy, picks the next track whose energy matches where the room is heading (with a slew limit so the set ramps instead of whiplashing), and fires the transition.

```
webcam ── motion energy ──┐
                          ▼
library/pool ─► analysis ─► controller ─► mixer (deck A ⇄ deck B) ─► speakers
   (local or YouTube)        (BPM/energy)   (pick + beatmatch + crossfade)
```

## Install

Requires **Python 3.13**, plus `ffmpeg` and `portaudio` (macOS shown):

```bash
brew install ffmpeg portaudio
python3.13 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

## Usage

**Local folder of music:**
```bash
./.venv/bin/python main.py ~/Music/yourset
```

**YouTube playlist** (streaming buffer — downloads a few tracks ahead, deletes them after they play, loops the playlist; personal use):
```bash
./.venv/bin/python main.py --youtube "https://www.youtube.com/playlist?list=..." --buffer 5
```

**No music handy?** Generate synthetic test tracks at known BPMs:
```bash
./.venv/bin/python tools/gen_demo_tracks.py demo_tracks
./.venv/bin/python main.py demo_tracks --simulate-crowd
```

**Run it headless** (no audio device / no webcam — good for testing the logic):
```bash
./.venv/bin/python main.py demo_tracks --dry-run --simulate-crowd --duration 60
```

### Useful flags
| Flag | Meaning |
|------|---------|
| `--simulate-crowd` | fake the crowd signal instead of using the webcam |
| `--camera N` | webcam index (default 0) |
| `--dry-run` | no audio device; just run the loop |
| `--crossfade SEC` | crossfade length (default 12) |
| `--cue-lead SEC` | cue the next track this long before the end (default 25) |
| `--buffer N` | YouTube: tracks kept downloaded at once — 1 playing + lookahead (default 5) |
| `--limit N` | YouTube: max distinct tracks to pull from a playlist (0 = all, loops) |
| `--keep` | YouTube: keep downloaded files instead of deleting after play |
| `--duration SEC` | auto-stop after N seconds (0 = run forever) |

## Project layout

```
main.py                  CLI entrypoint + live status line
dj/
  audio_io.py            decode any format via ffmpeg → numpy
  analysis.py            BPM + beat-grid + energy (numpy/scipy, no librosa)
  library.py             scan a folder, cache analysis, rank-normalize energy
  deck.py                one deck: playhead + varispeed read (beatmatch)
  mixer.py               two decks, equal-power beat-aligned crossfade
  crowd.py               webcam motion-energy → 0..1 vibe (sim fallback)
  controller.py          autopilot: read crowd, pick next, fire transitions
  youtube_source.py      yt-dlp wrapper: list / download / cache
  pool.py                streaming buffer for YouTube (download-ahead + cleanup)
tools/gen_demo_tracks.py synthetic test tracks at known BPMs
```

A deliberate choice: analysis uses only **numpy/scipy** (no `librosa`/`numba`), and decoding shells out to **ffmpeg** — so it installs cleanly even on bleeding-edge Python where the DSP wheels don't exist yet.

## Status

Verified:
- BPM detection within ~1 BPM on synthetic and real tracks.
- Energy ranking, crowd→target mapping, and track selection.
- Beat-aligned equal-power crossfade (validated headless).
- YouTube ingest + streaming buffer: disk stays pinned at the buffer size while the playlist loops.

Not yet tested / known limits:
- **Real webcam** crowd path (only exercised in simulation so far).
- **Live audio output** (validated headless; needs real speakers to confirm latency).
- Beatmatch is **varispeed** (slight pitch shift, capped at ±8%) — tracks far apart in BPM won't lock perfectly.

## Roadmap

Inspired by [pnlong/artificial_dj](https://github.com/pnlong/artificial_dj) and [mixxxdj/mixxx](https://github.com/mixxxdj/mixxx):

- **Bass-swap EQ transition** — 3-band EQ crossfade (swap lows first) instead of a plain fade.
- **Section-aware transitions** — detect intro/buildup/outro and mix at the right point, not just track-end.
- **Keylock time-stretch** (Rubber Band) — wider BPM range without the pitch shift.
- **Harmonic mixing** — key detection + Camelot-wheel compatibility in selection.
- **Web dashboard** — decks, crowd-cam + energy meter, the upcoming buffer, and live controls.

## A note on YouTube

The `--youtube` source uses `yt-dlp`. Downloading from YouTube is against YouTube's Terms of Service unless the content is your own, Creative Commons, or public domain. Intended for personal experimentation; point it at CC/royalty-free channels if you want to stay clean.

## License

MIT — see [LICENSE](LICENSE).
