"""Pull audio off YouTube with yt-dlp into a local cache folder.

Once downloaded, a track is just a file on disk, so it feeds the same
beatmatched two-deck engine as any local library. We keep the native bestaudio
stream (no re-encode) since ffmpeg decodes whatever container it lands in.

Note: downloading from YouTube is against YouTube's ToS unless the content is
your own, Creative Commons, or public domain. Intended for personal use.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

DEFAULT_CACHE = "yt_cache"


def _opts(cache_dir: str, limit: int, log: Optional[Callable[[str], None]]):
    def hook(d: dict) -> None:
        if log and d.get("status") == "finished":
            log(f"  downloaded {os.path.basename(d.get('filename', ''))}")

    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(cache_dir, "%(title).80B [%(id)s].%(ext)s"),
        "restrictfilenames": True,
        "nooverwrites": True,          # skip files already in the cache
        "ignoreerrors": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "extract_flat": False,
        "progress_hooks": [hook],
    }
    if limit > 0:
        opts["playlistend"] = limit    # cap how many entries a playlist pulls
    return opts


def fetch(urls: list[str], cache_dir: str = DEFAULT_CACHE, limit: int = 0,
          log: Optional[Callable[[str], None]] = None) -> str:
    """Download every URL/playlist entry into cache_dir; return the folder.

    The folder is then handed to Library.scan() exactly like a local music dir.
    """
    import yt_dlp

    os.makedirs(cache_dir, exist_ok=True)
    log = log or (lambda m: None)

    with yt_dlp.YoutubeDL(_opts(cache_dir, limit, log)) as ydl:
        for url in urls:
            log(f"fetching {url}")
            try:
                ydl.extract_info(url, download=True)
            except Exception as exc:  # noqa: BLE001 - one bad URL shouldn't abort the set
                log(f"  skipped {url}: {exc}")
    return cache_dir


def list_entries(urls: list[str], limit: int = 0,
                 log: Optional[Callable[[str], None]] = None) -> list[dict]:
    """Flat-list playlist/video entries WITHOUT downloading audio.

    Returns [{"id", "title"}], cheap enough to call up front so the pool knows
    the full candidate queue before pulling any audio.
    """
    import yt_dlp

    log = log or (lambda m: None)
    opts = {"extract_flat": True, "quiet": True, "no_warnings": True, "ignoreerrors": True}
    out: list[dict] = []
    with yt_dlp.YoutubeDL(opts) as ydl:
        for url in urls:
            try:
                info = ydl.extract_info(url, download=False)
            except Exception as exc:  # noqa: BLE001
                log(f"  could not list {url}: {exc}")
                continue
            if not info:
                continue
            entries = info["entries"] if info.get("entries") else [info]
            for e in entries:
                if e and e.get("id"):
                    out.append({"id": e["id"], "title": e.get("title") or e["id"]})
    if limit > 0:
        out = out[:limit]
    return out


def download_one(entry: dict, cache_dir: str = DEFAULT_CACHE,
                 log: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """Download one entry's audio; return its file path (or None on failure).

    Skips the network download if the file is already cached.
    """
    import yt_dlp

    log = log or (lambda m: None)
    os.makedirs(cache_dir, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={entry['id']}"
    with yt_dlp.YoutubeDL(_opts(cache_dir, 0, log)) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
            path = ydl.prepare_filename(info) if info else None
        except Exception as exc:  # noqa: BLE001
            log(f"  download failed {entry.get('title')}: {exc}")
            return None
    return path if path and os.path.exists(path) else None
