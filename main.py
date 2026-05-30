"""AI DJ — beatmatch a local library and steer the set by the crowd's energy.

Examples:
    python main.py ~/Music/sets
    python main.py ~/Music/sets --simulate-crowd
    python main.py ./demo_tracks --dry-run --simulate-crowd --duration 60
    python main.py --youtube "https://youtube.com/playlist?list=..." --simulate-crowd
"""

from __future__ import annotations

import argparse
import sys
import time

from dj.audio_io import have_ffmpeg
from dj.controller import Controller
from dj.crowd import CrowdSensor
from dj.library import Library
from dj.mixer import Mixer
from dj.pool import TrackPool


def _progress(i: int, n: int, name: str) -> None:
    print(f"\r  analysing {i}/{n}: {name[:48]:48}", end="", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="AI DJ")
    ap.add_argument("folder", nargs="?", help="folder of audio files to play")
    ap.add_argument("--youtube", nargs="+", metavar="URL",
                    help="YouTube video/playlist URLs to download and play (personal use)")
    ap.add_argument("--cache-dir", default="yt_cache", help="where YouTube audio is cached")
    ap.add_argument("--limit", type=int, default=0, help="max distinct tracks to pull from a playlist (0 = all, loops)")
    ap.add_argument("--buffer", type=int, default=5, help="tracks kept downloaded at once: 1 playing + lookahead")
    ap.add_argument("--keep", action="store_true", help="keep downloaded files instead of deleting after play")
    ap.add_argument("--simulate-crowd", action="store_true", help="fake the crowd signal (no webcam)")
    ap.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    ap.add_argument("--dry-run", action="store_true", help="no audio device; just run the loop")
    ap.add_argument("--crossfade", type=float, default=12.0, help="crossfade length, seconds")
    ap.add_argument("--cue-lead", type=float, default=25.0, help="cue the next track this long before the end")
    ap.add_argument("--duration", type=float, default=0.0, help="auto-stop after N seconds (0 = run forever)")
    args = ap.parse_args()

    if not have_ffmpeg():
        print("error: ffmpeg not found on PATH (brew install ffmpeg)", file=sys.stderr)
        return 1

    if args.youtube:
        print("Listing & buffering from YouTube ...")
        library = TrackPool(args.youtube, cache_dir=args.cache_dir, buffer=args.buffer,
                            limit=args.limit, ephemeral=not args.keep, log=print)
        if library.prime() == 0:
            print("error: could not fetch any tracks.", file=sys.stderr)
            return 1
        library.start()
    else:
        if not args.folder:
            print("error: give a music folder or --youtube URLs.", file=sys.stderr)
            return 1
        print(f"Scanning {args.folder} ...")
        library = Library(args.folder).scan(progress=_progress)
        print()
        if not library.tracks:
            print("error: no audio files found.", file=sys.stderr)
            return 1

    tracks = library.tracks
    print(f"Tracks ready: {len(tracks)}  "
          f"BPM {min(t.bpm for t in tracks):.0f}-{max(t.bpm for t in tracks):.0f}")

    mixer = Mixer(dry_run=args.dry_run)
    crowd = CrowdSensor(simulate=args.simulate_crowd, camera=args.camera).start()
    controller = Controller(
        library, mixer, crowd,
        crossfade_sec=args.crossfade, cue_lead_sec=args.cue_lead,
        log=print,
    )

    print(f"Crowd: {crowd.mode}   Audio: {'dry-run' if args.dry_run else 'live output'}")
    mixer.start()
    controller.start_set()

    start = time.monotonic()
    try:
        while True:
            controller.tick()
            if args.duration and (time.monotonic() - start) >= args.duration:
                break
            # Lightweight status line.
            live = controller.deck_tracks[mixer.current]
            print(f"\r  crowd {crowd.energy:.2f} | {('mixing' if mixer.is_transitioning() else 'playing'):7} "
                  f"| {(live.name[:30] if live else '-'):30} | {mixer.live_deck.remaining_sec:5.1f}s left "
                  f"| {mixer.live_deck.effective_bpm:5.1f} BPM   ", end="", flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        controller.stop()
        crowd.stop()
        mixer.stop()
        if hasattr(library, "stop"):
            library.stop()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
