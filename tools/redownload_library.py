"""Restore the vibe-dj working library from yt_library_manifest.json.

yt_cache is kept tiny on disk (a handful of songs). The manifest records every
track that was ever in the library (title + YouTube id), so the full set can be
re-downloaded on demand — before a gig, or to top the crate back up — without
keeping ~700MB of audio parked on disk year-round.

Usage:
    ./.venv/bin/python tools/redownload_library.py            # restore ALL missing
    ./.venv/bin/python tools/redownload_library.py --limit 20 # restore 20 missing
    ./.venv/bin/python tools/redownload_library.py --list     # show missing, download nothing

Already-present files are skipped (yt-dlp nooverwrites), so re-running is cheap
and safe. Disk is tight on this machine — restore only what you need.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dj import youtube_source as yt  # noqa: E402

MANIFEST = os.path.join(ROOT, "yt_library_manifest.json")


def load_manifest() -> dict:
    with open(MANIFEST) as fh:
        return json.load(fh)


def present_ids(cache_dir: str) -> set[str]:
    """Ids already on disk, read straight off the filenames (`... [id].ext`)."""
    ids: set[str] = set()
    if not os.path.isdir(cache_dir):
        return ids
    for fn in os.listdir(cache_dir):
        if "[" in fn and "]" in fn:
            ids.add(fn[fn.rfind("[") + 1:fn.rfind("]")])
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description="Restore the library from the manifest.")
    ap.add_argument("--limit", type=int, default=0, help="max tracks to download (0 = all missing)")
    ap.add_argument("--list", action="store_true", help="list missing tracks and exit")
    args = ap.parse_args()

    manifest = load_manifest()
    cache_dir = os.path.join(ROOT, manifest.get("cache_dir", "yt_cache"))
    have = present_ids(cache_dir)
    missing = [t for t in manifest["tracks"] if t["id"] not in have]

    print(f"{len(manifest['tracks'])} in manifest, {len(have)} on disk, {len(missing)} missing.")
    if args.list or not missing:
        for t in missing:
            print(f"  {t['id']}  {t['title']}")
        return 0

    todo = missing[: args.limit] if args.limit > 0 else missing
    print(f"downloading {len(todo)} track(s) into {cache_dir} ...")
    ok = 0
    for i, t in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {t['title']}")
        path = yt.download_one(t, cache_dir=cache_dir, log=lambda m: print("   ", m))
        if path:
            ok += 1
    print(f"done: {ok}/{len(todo)} restored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
