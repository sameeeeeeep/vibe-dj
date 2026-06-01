"""Web dashboard: a thin state-broadcast + control layer over the live engine.

Stdlib only (http.server + Server-Sent Events) to keep the project dep-light.
The engine already runs in-process, so this just snapshots the
controller/mixer/crowd/library each tick and streams JSON to the browser, while
POSTs from the browser inject control commands (skip, force-transition, nudge).

Endpoints:
    GET  /            the single-page dashboard
    GET  /events      SSE stream of engine snapshots (~3 Hz)
    GET  /frame.jpg   latest crowd-cam frame (204 when simulated / no camera)
    POST /control     {"cmd": ..., "deck": "A"|"B", "band": "low"|"mid"|"high",
                       "value": <float>, "url": <str>}
                      cmds: skip | force | cue | pause | nudge | crowd_manual |
                            crowd_set | eq | eq_kill | trim | bend | xfade |
                            add_url
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _QuietHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        # Browsers (EventSource) and curl drop connections constantly; that's
        # expected, not a bug. Swallow the disconnect noise, surface real errors.
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


class Dashboard:
    def __init__(self, controller, mixer, crowd, library,
                 host: str = "127.0.0.1", port: int = 8765):
        self.controller = controller
        self.mixer = mixer
        self.crowd = crowd
        self.library = library
        self.host = host
        self.port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ---- state -----------------------------------------------------------
    def _deck_info(self, name: str) -> dict:
        d = self.mixer.decks[name]
        track = self.controller.deck_tracks.get(name)
        loaded = d.analysis is not None and track is not None
        if name == self.mixer.current:
            role = "live"
        elif loaded:
            role = "cued"
        else:
            role = "idle"
        an = d.analysis
        eqm = self.mixer.eq_manual[name]
        return {
            "role": role,
            # A freed deck keeps its old Deck.title; only surface a title when a
            # track is actually loaded so an idle deck reads clean, not stale.
            "title": (d.title or (track.name if track else "")) if loaded else "",
            "base_bpm": round(an.bpm, 1) if an else 0.0,
            "bpm": round(d.effective_bpm, 1),
            "energy": round(track.energy, 3) if track else None,
            "gain": round(d.gain, 3),
            "playing": d.playing,
            "position": round(d.position_sec, 2),
            "duration": round(an.duration, 1) if an else 0.0,
            "remaining": round(d.remaining_sec, 1),
            # manual deck controls (DJ-driven)
            "bend": round(d.bend, 4),
            "trim": round(self.mixer.trim[name], 3),
            "eq": {"low": round(eqm[0], 3), "mid": round(eqm[1], 3), "high": round(eqm[2], 3)},
            "bass_auto": round(self.mixer.bass_auto[name], 3),
            # beat grid + phrase cues, for the live pulse and cue markers
            "beat_offset": round(an.beat_offset, 4) if an else 0.0,
            "beat_period": round(an.beat_period, 4) if an else 0.0,
            "phrase_period": round(an.phrase_period, 4) if an else 0.0,
            "mix_in": round(an.mix_in_sec(), 2) if an else 0.0,
            "mix_out": round(an.mix_out_sec(), 2) if an else 0.0,
            "intro_end": round(an.intro_end, 2) if an else 0.0,
            "outro_start": round(an.outro_start, 2) if an else 0.0,
        }

    def _buffer_info(self) -> list[dict]:
        loaded_ids = {id(t) for t in self.controller.deck_tracks.values() if t is not None}
        out = []
        for t in self.library.tracks:
            out.append({
                "name": t.name,
                "bpm": round(t.bpm, 1),
                "energy": round(t.energy, 3),
                "play_count": t.play_count,
                "loaded": id(t) in loaded_ids,
            })
        return out

    def snapshot(self) -> dict:
        return {
            "live": self.mixer.current,
            "paused": self.mixer.paused,
            "transition": self.mixer.transition_state(),
            "decks": {"A": self._deck_info("A"), "B": self._deck_info("B")},
            "crowd": {
                "energy": round(self.controller.effective_crowd(), 3),
                "sensor": round(self.crowd.energy, 3),
                "manual": self.controller.crowd_manual,
                "mode": self.crowd.mode,
                "has_cam": self.crowd.last_jpeg is not None,
            },
            "target_energy": round(self.controller.current_target(), 3),
            "energy_bias": round(self.controller.energy_bias, 3),
            "buffer": self._buffer_info(),
            "log": [{"t": t, "m": m} for (t, m) in self.controller.recent_events(14)],
        }

    def handle_command(self, payload: dict) -> None:
        cmd = str(payload.get("cmd", ""))
        deck = str(payload.get("deck", ""))
        band = str(payload.get("band", ""))
        try:
            value = float(payload.get("value", payload.get("delta", 0.0)) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if cmd == "skip":
            self.controller.request_skip()
        elif cmd == "force":
            self.controller.request_transition()
        elif cmd == "cue":
            self.controller.request_cue()
        elif cmd == "pause":
            self.mixer.toggle_pause()
        elif cmd == "nudge":
            self.controller.nudge_energy(value)
        elif cmd == "crowd_manual":
            self.controller.set_crowd_manual(value >= 0.5)
        elif cmd == "crowd_set":
            self.controller.set_crowd_energy(value)
        elif cmd == "eq":
            self.mixer.set_eq(deck, band, value)
        elif cmd == "eq_kill":
            self.mixer.toggle_kill(deck, band)
        elif cmd == "trim":
            self.mixer.set_trim(deck, value)
        elif cmd == "bend":
            self.mixer.set_bend(deck, value)
        elif cmd == "xfade":
            self.mixer.scrub_crossfade(value)
        elif cmd == "add_url":
            url = str(payload.get("url", "")).strip()
            if url.startswith(("http://", "https://")):
                self._enqueue_url(url)
            else:
                self.controller.log("[add]   ignored (need an http(s) link)")

    def _enqueue_url(self, url: str) -> None:
        """Fetch + analyse a pasted URL on a worker thread so the POST returns
        at once and the engine keeps playing; the new track surfaces in the
        buffer/up-next and becomes a pick candidate when it's ready."""
        def work() -> None:
            log = self.controller.log
            if not hasattr(self.library, "add_url"):
                log("[add]   this source can't take URLs")
                return
            log(f"[add]   fetching {url} …")
            try:
                added = self.library.add_url(url, log=log)
            except Exception as exc:  # noqa: BLE001 - surface, don't crash the server
                log(f"[add]   failed: {exc}")
                return
            for t in (added or []):
                log(f"[add]   ready {t.name}  {t.bpm:.0f} BPM  energy {t.energy:.2f}")

        threading.Thread(target=work, daemon=True).start()

    # ---- server ----------------------------------------------------------
    def start(self) -> "Dashboard":
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):  # silence the default access log
                pass

            def _send(self, code, body=b"", ctype="text/plain", extra=None):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                if body:
                    self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/":
                    self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                elif path == "/events":
                    self._stream_events()
                elif path == "/frame.jpg":
                    jpeg = dashboard.crowd.last_jpeg
                    if jpeg is None:
                        self._send(204)
                    else:
                        self._send(200, jpeg, "image/jpeg",
                                   {"Cache-Control": "no-store"})
                else:
                    self._send(404, b"not found")

            def do_POST(self):
                if self.path.split("?", 1)[0] != "/control":
                    self._send(404, b"not found")
                    return
                length = int(self.headers.get("Content-Length", 0) or 0)
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    payload = json.loads(raw or b"{}")
                    if not isinstance(payload, dict):
                        raise ValueError("payload must be an object")
                    dashboard.handle_command(payload)
                    ok = True
                except (ValueError, TypeError):
                    ok = False
                self._send(200 if ok else 400,
                           json.dumps({"ok": ok}).encode(), "application/json")

            def _stream_events(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                try:
                    while not dashboard._stop.is_set():
                        data = json.dumps(dashboard.snapshot())
                        self.wfile.write(f"data: {data}\n\n".encode())
                        self.wfile.flush()
                        dashboard._stop.wait(0.33)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        self._httpd = _QuietHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        kwargs={"poll_interval": 0.3}, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>vibe-dj</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Saira+Semi+Condensed:wght@500;600;700&family=Barlow+Semi+Condensed:wght@500;600;700&family=Space+Mono&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{
    --bg-page:#0A0C10; --bg-panel:#12151C; --bg-raised:#1A1F29; --bg-inset:#070809;
    --border:#262C38; --border-soft:#1C212B;
    --deck-a:#27C3F2; --deck-a-dim:#0E5C73; --deck-b:#FF8A1E; --deck-b-dim:#7A4310;
    --accent:#7C5CFF; --ok:#3DD68C; --warn:#F59E0B; --danger:#FF2D6E;
    --tp:#E8EDF4; --td:#7A8699; --tf:#4A5568; --seg-off:#1C2230;
    --f-disp:"Saira Semi Condensed","Arial Narrow",system-ui,sans-serif;
    --f-num:"Barlow Semi Condensed","Arial Narrow",system-ui,sans-serif;
    --f-mono:"Space Mono",ui-monospace,Menlo,Consolas,monospace;
    --f-body:"Inter",system-ui,-apple-system,sans-serif;
  }
  *{box-sizing:border-box;}
  body{margin:0;background:var(--bg-page);color:var(--tp);font-family:var(--f-body);
    -webkit-font-smoothing:antialiased;}
  .app{max-width:1440px;margin:0 auto;padding:20px;display:flex;flex-direction:column;gap:16px;}

  /* top bar */
  .topbar{height:64px;background:var(--bg-panel);border:1px solid var(--border);border-radius:14px;
    display:flex;align-items:center;justify-content:space-between;padding:0 24px;}
  .brand{display:flex;align-items:center;gap:13px;}
  .logo{color:var(--deck-a);display:flex;}
  .brand-name{font-family:var(--f-disp);font-weight:700;font-size:20px;letter-spacing:.08em;}
  .brand-pill{font-family:var(--f-mono);font-size:10px;letter-spacing:.12em;color:var(--td);
    border:1px solid var(--border);border-radius:6px;padding:5px 9px;}
  .topstats{display:flex;align-items:center;gap:24px;}
  .tstat{display:flex;flex-direction:column;align-items:flex-end;gap:1px;}
  .tstat-l{font-family:var(--f-mono);font-size:9px;letter-spacing:.14em;color:var(--tf);}
  .tstat-v{font-family:var(--f-num);font-size:16px;font-weight:600;letter-spacing:.02em;}
  .live-pill{display:flex;align-items:center;gap:6px;border:1px solid var(--danger);border-radius:8px;
    padding:6px 11px;font-family:var(--f-mono);font-size:10px;letter-spacing:.14em;color:var(--danger);}
  .live-pill.live{border-color:var(--ok);color:var(--ok);}
  .dot{width:6px;height:6px;border-radius:50%;background:currentColor;display:inline-block;flex:none;}
  .dot-ok{background:var(--ok);} .dot-live{background:var(--danger);} .dot-faint{background:var(--tf);}

  /* decks row */
  .decks{display:flex;gap:16px;align-items:stretch;}
  .deck{flex:1;min-width:0;background:var(--bg-panel);border-radius:18px;padding:24px;
    display:flex;flex-direction:column;gap:16px;}
  .deck.a{border:1px solid var(--deck-a-dim);}
  .deck.b{border:1px solid var(--deck-b-dim);}
  .deck-head{display:flex;align-items:center;justify-content:space-between;}
  .chips{display:flex;align-items:center;gap:12px;}
  .chip-deck{font-family:var(--f-mono);font-size:10px;letter-spacing:.14em;padding:5px 9px;border-radius:6px;}
  .deck.a .chip-deck{color:var(--deck-a);background:#0C2630;border:1px solid var(--deck-a-dim);}
  .deck.b .chip-deck{color:var(--deck-b);background:#261A0C;border:1px solid var(--deck-b-dim);}
  .status{font-family:var(--f-mono);font-size:10px;letter-spacing:.14em;color:var(--td);}
  .deck-ch{font-family:var(--f-disp);font-size:13px;letter-spacing:.1em;color:var(--tf);}
  .deck-title{display:flex;flex-direction:column;gap:3px;}
  .t-name{font-family:var(--f-disp);font-weight:600;font-size:30px;line-height:1.05;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  .t-art{font-family:var(--f-body);font-size:13px;color:var(--td);min-height:16px;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  .bpm-row{display:flex;align-items:flex-end;justify-content:space-between;}
  .bpm{display:flex;align-items:baseline;gap:8px;}
  .bpm-v{font-family:var(--f-num);font-weight:700;font-size:64px;line-height:.9;}
  .deck.a .bpm-v{color:var(--deck-a);} .deck.b .bpm-v{color:var(--deck-b);}
  .bpm-u{font-family:var(--f-mono);font-size:12px;letter-spacing:.1em;color:var(--td);}
  .bm-side{display:flex;flex-direction:column;align-items:flex-end;gap:5px;}
  .bm-pill{font-family:var(--f-num);font-weight:600;font-size:14px;padding:4px 9px;border-radius:7px;}
  .deck.a .bm-pill{color:var(--deck-a);background:#0C2630;border:1px solid var(--deck-a-dim);}
  .deck.b .bm-pill{color:var(--deck-b);background:#261A0C;border:1px solid var(--deck-b-dim);}
  .bm-orig{font-family:var(--f-mono);font-size:9px;letter-spacing:.1em;color:var(--tf);}
  .wave{display:flex;align-items:flex-end;gap:2px;height:64px;}
  .wave span{flex:1;min-width:0;border-radius:2px;}
  .deck.a .wave span{background:var(--deck-a-dim);} .deck.a .wave span.lit{background:var(--deck-a);}
  .deck.b .wave span{background:var(--deck-b-dim);} .deck.b .wave span.lit{background:var(--deck-b);}
  .deck-foot{display:flex;align-items:center;justify-content:space-between;
    font-family:var(--f-mono);font-size:11px;color:var(--td);}
  .foot-mid{color:var(--tf);letter-spacing:.08em;}

  /* crossfader */
  .xfader{width:180px;flex:none;background:var(--bg-panel);border:1px solid var(--border);border-radius:18px;
    padding:18px 20px;display:flex;flex-direction:column;align-items:center;justify-content:space-between;}
  .xf-label{font-family:var(--f-mono);font-size:10px;letter-spacing:.14em;color:var(--td);}
  .xf-track{flex:1;display:flex;flex-direction:column;align-items:center;gap:10px;padding:14px 0;}
  .xf-end{font-family:var(--f-disp);font-weight:700;font-size:14px;}
  .xf-a{color:var(--deck-a);} .xf-b{color:var(--deck-b);}
  .xf-rail{position:relative;flex:1;width:6px;border-radius:3px;
    background:linear-gradient(180deg,var(--deck-a),#1A1F29 50%,var(--deck-b));}
  .xf-knob{position:absolute;left:50%;top:0%;width:22px;height:8px;border-radius:3px;
    background:var(--tp);transform:translate(-50%,-50%);box-shadow:0 0 8px #000a;transition:top .25s;}
  .xf-info{display:flex;flex-direction:column;align-items:center;gap:7px;width:100%;}
  .sync-pill{font-family:var(--f-num);font-weight:600;font-size:13px;color:var(--ok);
    background:#0F2018;border:1px solid #1C3A2B;border-radius:6px;padding:5px 10px;}
  .xfader.mixing .sync-pill{color:var(--accent);background:#16121F;border-color:#2E2547;}
  .xf-next{display:flex;flex-direction:column;align-items:center;gap:1px;}
  .xf-next-l{font-family:var(--f-mono);font-size:9px;letter-spacing:.12em;color:var(--tf);}
  .xf-next-v{font-family:var(--f-num);font-weight:600;font-size:18px;}
  .xf-sub{font-family:var(--f-mono);font-size:9px;letter-spacing:.1em;color:var(--td);}

  /* mid row */
  .midrow{display:flex;gap:16px;height:240px;}
  .crowd{flex:1;min-width:0;background:var(--bg-panel);border:1px solid var(--border);border-radius:18px;
    padding:20px;display:flex;gap:22px;}
  .cam{position:relative;width:236px;flex:none;border-radius:12px;overflow:hidden;
    background:#0A0D13;border:1px solid var(--border);}
  .cam-blobs{position:absolute;inset:0;filter:blur(34px);background:
    radial-gradient(120px 120px at 18% 82%, #FF2D6Ecc, transparent 70%),
    radial-gradient(130px 130px at 82% 14%, #27C3F2cc, transparent 70%),
    radial-gradient(120px 120px at 56% 66%, #7C5CFFcc, transparent 70%);}
  .cam img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;display:none;}
  .cam.has-cam img{display:block;} .cam.has-cam .cam-blobs{display:none;}
  .cam-tag{position:absolute;left:12px;top:12px;display:flex;align-items:center;gap:5px;
    background:#000000b0;border-radius:6px;padding:5px 8px;font-family:var(--f-mono);
    font-size:9px;letter-spacing:.1em;color:#B9C2D0;}
  .cam-foot{position:absolute;left:12px;bottom:12px;font-family:var(--f-mono);
    font-size:9px;letter-spacing:.1em;color:#B9C2D0;}
  .energy{flex:1;min-width:0;display:flex;flex-direction:column;justify-content:space-between;}
  .en-head{display:flex;align-items:center;justify-content:space-between;}
  .en-title{font-family:var(--f-mono);font-size:11px;letter-spacing:.14em;color:var(--td);}
  .mode-pill{display:flex;align-items:center;gap:6px;font-family:var(--f-mono);font-size:9px;
    letter-spacing:.12em;color:var(--ok);background:#0F2018;border:1px solid #1C3A2B;
    border-radius:6px;padding:5px 9px;}
  .en-big{display:flex;align-items:flex-end;gap:14px;}
  .en-val{font-family:var(--f-num);font-weight:700;font-size:52px;line-height:.85;}
  .en-cap{display:flex;flex-direction:column;gap:2px;padding-bottom:4px;}
  .en-tier{font-family:var(--f-disp);font-weight:600;font-size:14px;letter-spacing:.06em;}
  .en-sub{font-family:var(--f-body);font-size:12px;color:var(--td);}
  .meter-block{display:flex;flex-direction:column;gap:7px;}
  .meter-lab{display:flex;justify-content:space-between;font-family:var(--f-mono);
    font-size:10px;letter-spacing:.1em;color:var(--td);}
  .meter-lab .bias{color:var(--accent);margin-left:6px;}
  .meter-lab .mval{font-family:var(--f-num);font-weight:600;color:var(--tp);}
  .segs{display:flex;gap:3px;height:14px;}
  .segs span{flex:1;border-radius:2px;background:var(--seg-off);transition:background .25s;}
  .crowd-toggle{display:flex;border:1px solid var(--border);border-radius:7px;overflow:hidden;}
  .seg-btn{font-family:var(--f-mono);font-size:9px;letter-spacing:.12em;color:var(--td);
    background:transparent;border:none;padding:6px 12px;cursor:pointer;transition:background .15s,color .15s;}
  .seg-btn:hover{color:var(--tp);}
  #cm-auto.active{background:#0F2018;color:var(--ok);}
  #cm-manual.active{background:#16121F;color:var(--accent);}
  .segs.fader{cursor:ew-resize;outline:1px solid #2E2547;outline-offset:4px;border-radius:2px;touch-action:none;}
  .segs.fader span{transition:none;}
  .meter-block.manual .meter-lab span:first-child{color:var(--accent);}

  /* controls */
  .controls{width:400px;flex:none;background:var(--bg-panel);border:1px solid var(--border);
    border-radius:18px;padding:20px;display:flex;flex-direction:column;justify-content:space-between;gap:14px;}
  .ctl-head{display:flex;align-items:center;justify-content:space-between;}
  .ctl-title{font-family:var(--f-mono);font-size:10px;letter-spacing:.14em;color:var(--td);}
  .ctl-hint{font-family:var(--f-body);font-size:11px;line-height:1.5;color:var(--tf);}
  .auto-pill{display:flex;align-items:center;gap:6px;font-family:var(--f-mono);font-size:9px;
    letter-spacing:.1em;color:var(--ok);background:#0F2018;border:1px solid #1C3A2B;
    border-radius:6px;padding:5px 9px;}
  .ctl-actions{display:flex;gap:10px;height:84px;}
  .big-btn{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:9px;
    border-radius:12px;background:var(--bg-raised);border:1px solid var(--border);color:var(--tp);
    font-family:var(--f-disp);font-weight:600;font-size:13px;letter-spacing:.08em;cursor:pointer;
    transition:border-color .15s,transform .05s;}
  .big-btn svg{color:var(--tp);}
  .big-btn:hover{border-color:var(--deck-a);}
  .big-btn:active{transform:translateY(1px);}
  .big-btn.force{background:#16121F;border-color:#2E2547;color:var(--accent);}
  .big-btn.force svg{color:var(--accent);}
  .big-btn.force:hover{border-color:var(--accent);}
  .nudge{display:flex;flex-direction:column;gap:9px;}
  .nudge-lab{font-family:var(--f-mono);font-size:10px;letter-spacing:.14em;color:var(--td);}
  .nudge-row{display:flex;align-items:center;gap:10px;height:56px;}
  .nudge-btn{flex:1;height:100%;display:flex;align-items:center;justify-content:center;gap:8px;
    border-radius:10px;font-family:var(--f-disp);font-weight:600;font-size:13px;letter-spacing:.08em;
    cursor:pointer;transition:border-color .15s,transform .05s;}
  .nudge-btn:active{transform:translateY(1px);}
  .nudge-btn.cool{background:#0C1E26;border:1px solid var(--deck-a-dim);color:var(--deck-a);}
  .nudge-btn.cool:hover{border-color:var(--deck-a);}
  .nudge-btn.hot{background:#26140C;border:1px solid var(--deck-b-dim);color:var(--deck-b);}
  .nudge-btn.hot:hover{border-color:var(--deck-b);}
  .bias-box{flex:none;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:0 6px;}
  .bias-v{font-family:var(--f-num);font-weight:600;font-size:22px;}
  .bias-l{font-family:var(--f-mono);font-size:8px;letter-spacing:.14em;color:var(--tf);}

  /* up next */
  .upnext{background:var(--bg-panel);border:1px solid var(--border);border-radius:18px;
    padding:20px;display:flex;flex-direction:column;gap:16px;}
  .up-head{display:flex;align-items:center;justify-content:space-between;}
  .up-head-l{display:flex;align-items:center;gap:10px;font-family:var(--f-disp);font-weight:600;
    font-size:14px;letter-spacing:.1em;}
  .up-head-l svg{color:var(--td);}
  .up-head-r{font-family:var(--f-mono);font-size:10px;letter-spacing:.1em;color:var(--tf);}
  .up-add{display:flex;align-items:center;gap:10px;}
  .add-input{width:248px;background:var(--bg-inset);border:1px solid var(--border);border-radius:8px;
    color:var(--tp);font-family:var(--f-mono);font-size:11px;letter-spacing:.03em;padding:8px 11px;outline:none;
    transition:border-color .15s;}
  .add-input:focus{border-color:var(--deck-a);}
  .add-input::placeholder{color:var(--tf);}
  .add-btn{font-family:var(--f-disp);font-weight:600;font-size:11px;letter-spacing:.08em;cursor:pointer;
    color:var(--deck-a);background:#0C2630;border:1px solid var(--deck-a-dim);border-radius:8px;padding:8px 13px;
    transition:border-color .15s,transform .05s;}
  .add-btn:hover{border-color:var(--deck-a);} .add-btn:active{transform:translateY(1px);}
  .cards{display:flex;gap:12px;}
  .card{flex:1;min-width:0;display:flex;flex-direction:column;gap:12px;padding:14px;border-radius:12px;
    background:var(--bg-raised);border:1px solid var(--border);}
  .card.on{background:#0B161C;border-color:var(--deck-a-dim);}
  .card-top{display:flex;align-items:center;justify-content:space-between;}
  .cnum{font-family:var(--f-mono);font-size:11px;letter-spacing:.1em;color:var(--tf);}
  .cbadge{display:flex;align-items:center;gap:5px;font-family:var(--f-mono);font-size:8px;
    letter-spacing:.12em;color:var(--td);}
  .cbadge.live{color:var(--deck-a);background:#0C2630;border:1px solid var(--deck-a-dim);
    border-radius:5px;padding:4px 7px;}
  .card-title{display:flex;flex-direction:column;gap:2px;min-width:0;}
  .cname{font-family:var(--f-disp);font-weight:600;font-size:15px;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  .cart{font-family:var(--f-body);font-size:11px;color:var(--td);min-height:14px;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  .card-bot{display:flex;align-items:flex-end;justify-content:space-between;}
  .cbpm{font-family:var(--f-num);font-weight:600;font-size:18px;display:flex;align-items:baseline;gap:3px;}
  .cbpm small{font-family:var(--f-mono);font-size:8px;letter-spacing:.1em;color:var(--tf);}
  .cen{display:flex;align-items:center;gap:5px;font-family:var(--f-mono);font-size:12px;}
  .cdot{width:8px;height:8px;border-radius:50%;}
  .card-empty{color:var(--td);font-family:var(--f-mono);font-size:12px;padding:8px;}

  /* master transport (topbar play/pause) */
  .transport{display:flex;align-items:center;gap:8px;background:var(--ok);color:#04140C;border:none;
    border-radius:9px;padding:9px 15px;font-family:var(--f-disp);font-weight:700;font-size:13px;
    letter-spacing:.08em;cursor:pointer;transition:transform .05s,filter .15s;}
  .transport:hover{filter:brightness(1.08);} .transport:active{transform:translateY(1px);}
  .transport.paused{background:var(--warn);color:#1A0F00;}
  .transport .pp-ic{font-size:13px;line-height:1;}

  /* live beat + phrase pulse */
  .beatline{display:flex;align-items:center;justify-content:space-between;}
  .beats{display:flex;gap:7px;}
  .beats span{width:13px;height:13px;border-radius:50%;background:currentColor;opacity:.12;
    transition:opacity .04s linear;}
  .deck.a .beats{color:var(--deck-a);} .deck.b .beats{color:var(--deck-b);}
  .beats span.hit{box-shadow:0 0 11px currentColor;}
  .phrasetxt{font-family:var(--f-mono);font-size:10px;letter-spacing:.12em;color:var(--td);}

  /* waveform overlay: phrase cue markers + moving playhead */
  .wavewrap{position:relative;}
  .cue{position:absolute;top:-5px;bottom:-5px;width:2px;z-index:2;pointer-events:none;display:none;}
  .cue-in{background:var(--ok);box-shadow:0 0 6px var(--ok);}
  .cue-out{background:var(--danger);box-shadow:0 0 6px var(--danger);}
  .cue::before{content:"";position:absolute;top:-4px;left:-2px;width:6px;height:6px;border-radius:1px;background:inherit;}
  .playhead{position:absolute;top:-5px;bottom:-5px;left:0;width:2px;background:var(--tp);z-index:3;
    pointer-events:none;box-shadow:0 0 7px #fff9;}

  /* per-deck control strip: 3-band EQ + volume + pitch */
  .strip{display:flex;gap:16px;align-items:flex-start;justify-content:space-between;
    padding-top:14px;margin-top:4px;border-top:1px solid var(--border-soft);}
  .eq{display:flex;gap:13px;}
  .eqband,.chan{display:flex;flex-direction:column;align-items:center;gap:9px;}
  .strip-lab{font-family:var(--f-mono);font-size:9px;letter-spacing:.12em;color:var(--td);}
  .killbtn{font-family:var(--f-mono);font-size:9px;letter-spacing:.1em;color:var(--td);background:var(--bg-inset);
    border:1px solid var(--border);border-radius:5px;padding:5px 0;width:46px;cursor:pointer;transition:.15s;}
  .killbtn:hover{color:var(--tp);border-color:var(--td);}
  .killbtn.killed{color:var(--danger);border-color:var(--danger);background:#2A0E18;}
  .pitch-v{font-family:var(--f-num);font-size:11px;font-weight:600;color:var(--tp);min-height:14px;}
  .vrange{-webkit-appearance:none;appearance:none;writing-mode:vertical-lr;direction:rtl;
    width:10px;height:100px;background:var(--bg-inset);border-radius:6px;border:1px solid var(--border);
    cursor:pointer;margin:0;}
  .vrange::-webkit-slider-thumb{-webkit-appearance:none;width:26px;height:11px;border-radius:3px;
    background:var(--tp);box-shadow:0 0 6px #000a;border:1px solid #0008;}
  .vrange::-moz-range-thumb{width:26px;height:11px;border:none;border-radius:3px;background:var(--tp);}
  .deck.a .vrange::-webkit-slider-thumb{background:var(--deck-a);}
  .deck.b .vrange::-webkit-slider-thumb{background:var(--deck-b);}
  .deck.a .vrange::-moz-range-thumb{background:var(--deck-a);}
  .deck.b .vrange::-moz-range-thumb{background:var(--deck-b);}
  .vrange.pitch::-webkit-slider-thumb{background:var(--accent);}
  .vrange.pitch::-moz-range-thumb{background:var(--accent);}

  /* crossfader scrub + bass-swap meter (in the center column) */
  .xf-rail{cursor:ns-resize;}
  .bass-swap{width:100%;display:flex;flex-direction:column;gap:6px;}
  .bs-lab{font-family:var(--f-mono);font-size:9px;letter-spacing:.12em;color:var(--tf);text-align:center;}
  .bs-row{display:flex;align-items:center;gap:7px;}
  .bs-tag{font-family:var(--f-disp);font-weight:700;font-size:11px;width:9px;}
  .bs-tag.a{color:var(--deck-a);} .bs-tag.b{color:var(--deck-b);}
  .bs-bar{flex:1;height:8px;background:var(--bg-inset);border-radius:4px;overflow:hidden;
    border:1px solid var(--border-soft);}
  .bs-fill{height:100%;border-radius:4px;transition:width .1s linear;}
  .bs-fill.a{background:var(--deck-a);} .bs-fill.b{background:var(--deck-b);}

  /* autopilot decision log */
  .logpanel{background:var(--bg-panel);border:1px solid var(--border);border-radius:18px;
    padding:18px 20px;display:flex;flex-direction:column;gap:12px;}
  .log-head{display:flex;align-items:center;justify-content:space-between;font-family:var(--f-mono);
    font-size:10px;letter-spacing:.14em;color:var(--td);}
  .log-body{display:flex;flex-direction:column;gap:7px;font-family:var(--f-mono);font-size:11px;
    max-height:150px;overflow:hidden;}
  .log-line{display:flex;gap:10px;align-items:baseline;}
  .log-t{color:var(--tf);font-size:9px;flex:none;}
  .log-m{color:var(--td);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  .log-line.mix .log-m{color:var(--accent);}
  .log-line.cue .log-m{color:var(--deck-a);}
  .log-line.start .log-m,.log-line.live .log-m{color:var(--ok);}

  @media (max-width:1120px){
    .decks{flex-direction:column;height:auto;}
    .deck{min-height:480px;}
    .xfader{width:auto;min-height:200px;}
    .midrow{flex-direction:column;height:auto;}
    .crowd{flex-direction:column;}
    .cam{width:auto;height:160px;}
    .controls{width:auto;}
  }
  @media (max-width:700px){
    .cards{flex-wrap:wrap;} .card{flex:1 1 140px;}
  }

  /* ===================================================================== */
  /* HARDWARE CONSOLE: jog decks + center mixer (overrides the panels above) */
  /* ===================================================================== */
  .console{display:grid;grid-template-columns:minmax(0,1fr) 392px minmax(0,1fr);
    background:linear-gradient(180deg,#0e1116,#080a0d);border:1px solid #20262f;border-radius:20px;
    box-shadow:inset 0 1px 0 #ffffff0a, 0 12px 44px #0009;overflow:hidden;}
  .console .deck{background:transparent;border:none;border-radius:0;padding:22px 24px;gap:13px;}
  .console .deck.a{border-right:1px solid #1b212a;}
  .console .deck.b{border-left:1px solid #1b212a;}
  .console .mixer{background:linear-gradient(180deg,#0b0d11,#060708);padding:20px 18px;
    display:flex;flex-direction:column;gap:15px;}
  .console .wave{height:44px;}

  .deck-top{display:flex;align-items:center;justify-content:space-between;}
  .deck-leds{display:flex;gap:7px;}
  .led{font-family:var(--f-mono);font-size:9px;letter-spacing:.12em;padding:4px 8px;border-radius:5px;
    border:1px solid var(--border);color:var(--tf);background:var(--bg-inset);transition:.15s;}
  .led.sync.on{color:var(--ok);border-color:#1C3A2B;background:#0F2018;box-shadow:0 0 9px #3dd68c4d;}
  .led.play.on{color:var(--deck-a);border-color:var(--deck-a-dim);background:#0C2630;box-shadow:0 0 9px #27c3f24d;}
  .deck.b .led.play.on{color:var(--deck-b);border-color:var(--deck-b-dim);background:#261A0C;box-shadow:0 0 9px #ff8a1e4d;}

  /* jog wheel + pitch fader */
  .jog-row{display:flex;align-items:center;gap:16px;justify-content:center;padding:4px 0;}
  .deck.b .jog-row{flex-direction:row-reverse;}
  .pitch-col{display:flex;flex-direction:column;align-items:center;gap:8px;}
  .pitch-col .vrange{height:150px;}
  .jog{position:relative;width:218px;height:218px;flex:none;}
  .jog-ring{position:absolute;inset:0;width:100%;height:100%;transform:rotate(-90deg);}
  .jog-ring .ring-bg{fill:none;stroke:#171b21;stroke-width:3.5;}
  .jog-ring .ring-fg{fill:none;stroke:var(--deck-a);stroke-width:3.5;stroke-linecap:round;}
  .deck.b .jog-ring .ring-fg{stroke:var(--deck-b);}
  .jog-platter{position:absolute;inset:12px;border-radius:50%;will-change:transform;
    background:repeating-conic-gradient(from 0deg,#1d2128 0deg 7.5deg,#262b34 7.5deg 15deg),
      radial-gradient(circle at 50% 36%,#3a414b,#20242b 56%,#0c0e12 100%);
    background-blend-mode:soft-light,normal;
    box-shadow:inset 0 3px 9px #0009,inset 0 -4px 13px #000,0 7px 20px #000a;border:1px solid #2b313a;}
  .jog-platter::after{content:"";position:absolute;inset:33%;border-radius:50%;
    background:radial-gradient(circle at 50% 36%,#484e58,#191d23 72%);
    box-shadow:inset 0 1px 3px #0008,0 1px 2px #000;}
  .jog-mark{position:absolute;left:50%;top:5.5%;width:4px;height:21%;margin-left:-2px;border-radius:2px;
    background:var(--deck-a);box-shadow:0 0 11px var(--deck-a);}
  .deck.b .jog-mark{background:var(--deck-b);box-shadow:0 0 11px var(--deck-b);}
  .jog-center{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;
    justify-content:center;pointer-events:none;gap:1px;}
  .jog-pos{font-family:var(--f-num);font-weight:700;font-size:30px;color:var(--tp);line-height:1;}
  .jog-rem{font-family:var(--f-mono);font-size:10px;letter-spacing:.08em;color:var(--td);}

  /* center mixer */
  .mx-head{display:flex;align-items:center;justify-content:space-between;}
  .mx-title{font-family:var(--f-disp);font-weight:700;font-size:13px;letter-spacing:.16em;color:var(--td);}
  .mx-strips{display:grid;grid-template-columns:1fr auto 1fr;gap:12px;align-items:stretch;}
  .mx-ch{display:flex;flex-direction:column;align-items:center;gap:12px;
    background:#0c0f13;border:1px solid #1a1f27;border-radius:12px;padding:13px 9px;}
  .ch-lab{font-family:var(--f-mono);font-size:10px;letter-spacing:.14em;}
  .mx-ch.a .ch-lab{color:var(--deck-a);} .mx-ch.b .ch-lab{color:var(--deck-b);}
  .knobs{display:flex;flex-direction:column;gap:11px;width:100%;}
  .knob-cell{display:flex;align-items:center;justify-content:center;gap:8px;}
  .knob{width:42px;height:42px;flex:none;border-radius:50%;cursor:ns-resize;touch-action:none;position:relative;
    background:radial-gradient(circle at 50% 30%,#333a45,#171b21 72%);
    border:1px solid #2d333d;box-shadow:inset 0 1px 2px #0008,0 2px 5px #0009;}
  .knob-dial{position:absolute;inset:0;border-radius:50%;transition:transform .08s ease-out;}
  .knob-ind{position:absolute;left:50%;top:3px;width:3px;height:14px;margin-left:-1.5px;border-radius:2px;
    background:var(--deck-a);box-shadow:0 0 5px var(--deck-a);}
  .mx-ch.b .knob-ind{background:var(--deck-b);box-shadow:0 0 5px var(--deck-b);}
  .knob.killed{box-shadow:inset 0 0 0 1.5px var(--danger),inset 0 1px 2px #0008;}
  .knob.killed .knob-ind{background:var(--danger);box-shadow:0 0 7px var(--danger);}
  .knob-cell .killbtn{width:42px;}
  .fader-cell{display:flex;align-items:flex-end;gap:9px;height:150px;padding-top:2px;}
  .fader-cell .vrange{height:150px;}
  .chmeter{width:7px;height:100%;border-radius:4px;background:#070809;border:1px solid #1a1f27;
    overflow:hidden;display:flex;align-items:flex-end;}
  .chmeter-fill{width:100%;height:0%;border-radius:4px 4px 0 0;
    background:linear-gradient(0deg,#3DD68C,#84CC16 55%,#F59E0B 80%,#FF2D6E);transition:height .1s linear;}
  .mx-center{display:flex;flex-direction:column;justify-content:center;gap:12px;min-width:92px;}

  .xfader-h{display:flex;align-items:center;gap:12px;padding:9px 8px;
    background:#0c0f13;border:1px solid #1a1f27;border-radius:12px;}
  .xfader-h .xf-end{font-family:var(--f-disp);font-weight:700;font-size:14px;}
  .xf-rail-h{position:relative;flex:1;height:10px;border-radius:5px;cursor:ew-resize;touch-action:none;
    background:linear-gradient(90deg,var(--deck-a),#15181d 50%,var(--deck-b));box-shadow:inset 0 1px 3px #000a;}
  .xf-knob-h{position:absolute;top:50%;left:50%;width:26px;height:34px;border-radius:5px;
    transform:translate(-50%,-50%);background:linear-gradient(#eef2f7,#aab2bf);
    box-shadow:0 3px 8px #000b,inset 0 1px 0 #fff,inset 0 -2px 3px #0003;border:1px solid #0006;transition:left .12s;}
  .xfader-h.mixing .xf-knob-h{background:linear-gradient(#cdbcff,#8b6cf0);}

  .mx-transport{display:flex;gap:9px;}
  .mx-btn{flex:1;padding:11px 0;border-radius:10px;background:var(--bg-raised);border:1px solid var(--border);
    color:var(--tp);font-family:var(--f-disp);font-weight:600;font-size:12px;letter-spacing:.07em;cursor:pointer;
    transition:border-color .15s,transform .05s;}
  .mx-btn:hover{border-color:var(--td);} .mx-btn:active{transform:translateY(1px);}
  .mx-btn.cue{color:var(--deck-a);border-color:var(--deck-a-dim);background:#0C2630;}
  .mx-btn.skip{color:var(--ok);border-color:#1C3A2B;background:#0F2018;}
  .mx-btn.force{color:var(--accent);border-color:#2E2547;background:#16121F;}

  @media (max-width:1120px){
    .console{grid-template-columns:1fr;}
    .console .deck.a,.console .deck.b{border:none;border-bottom:1px solid #1b212a;}
  }
</style>
</head>
<body>
<div class="app">

  <div class="topbar">
    <div class="brand">
      <span class="logo"><svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="2.4" fill="currentColor" stroke="none"/></svg></span>
      <span class="brand-name">VIBE DJ</span>
      <span class="brand-pill" id="mode-tag">AUTOPILOT &middot; CROWD-STEERED</span>
    </div>
    <div class="topstats">
      <div class="tstat"><span class="tstat-l">SET TIME</span><span class="tstat-v" id="set-time">0:00:00</span></div>
      <div class="tstat"><span class="tstat-l">IN ROTATION</span><span class="tstat-v" id="rotation">0 TRACKS</span></div>
      <button class="transport" id="playpause" onclick="send('pause')">
        <span class="pp-ic" id="pp-ic">&#9208;</span><span id="pp-txt">PAUSE</span></button>
      <div class="live-pill" id="live-pill"><span class="dot" id="live-dot"></span><span id="live-txt">CONNECTING</span></div>
    </div>
  </div>

  <div class="console">
    <div class="deck a" id="deckA">
      <div class="deck-top">
        <div class="chips"><span class="chip-deck">DECK A</span><span class="status" id="A-status">IDLE</span></div>
        <div class="deck-leds">
          <span class="led sync" id="A-sync-led">SYNC</span>
          <span class="led play" id="A-play-led">PLAY</span>
        </div>
      </div>
      <div class="deck-title"><div class="t-name" id="A-title">&mdash;</div><div class="t-art" id="A-art"></div></div>
      <div class="bpm-row">
        <div class="bpm"><span class="bpm-v" id="A-bpm">0.0</span><span class="bpm-u">BPM</span></div>
        <div class="bm-side"><span class="bm-pill" id="A-bm">&bull; 0.0%</span><span class="bm-orig" id="A-orig"></span></div>
      </div>
      <div class="beatline">
        <div class="beats" id="A-beats"><span></span><span></span><span></span><span></span></div>
        <span class="phrasetxt" id="A-phrase">&mdash;</span>
      </div>
      <div class="wavewrap">
        <div class="wave" id="A-wave"></div>
        <div class="cue cue-in" id="A-cue-in"></div>
        <div class="cue cue-out" id="A-cue-out"></div>
        <div class="playhead" id="A-head"></div>
      </div>
      <div class="deck-foot"><span id="A-pos">0:00</span><span class="foot-mid" id="A-mid"></span><span id="A-dur">0:00</span></div>
      <div class="jog-row">
        <div class="pitch-col">
          <input class="vrange pitch" id="A-pitch" type="range" min="-80" max="80" value="0"
                 oninput="setPitch('A',this.value)">
          <span class="pitch-v" id="A-pitch-v">+0.0%</span>
          <span class="strip-lab">PITCH</span>
        </div>
        <div class="jog">
          <svg class="jog-ring" viewBox="0 0 100 100">
            <circle class="ring-bg" cx="50" cy="50" r="47"></circle>
            <circle class="ring-fg" id="A-ring" cx="50" cy="50" r="47"></circle>
          </svg>
          <div class="jog-platter" id="A-platter"><div class="jog-mark"></div></div>
          <div class="jog-center">
            <span class="jog-pos" id="A-jog-pos">0:00</span>
            <span class="jog-rem" id="A-jog-rem">-0:00</span>
          </div>
        </div>
      </div>
    </div>

    <div class="mixer">
      <div class="mx-head">
        <span class="mx-title">MIXER</span>
        <span class="sync-pill" id="sync-pill">SYNC 0.0</span>
      </div>
      <div class="mx-strips">
        <div class="mx-ch a">
          <span class="ch-lab">CH A</span>
          <div class="knobs">
            <div class="knob-cell"><div class="knob" id="A-eq-high" data-deck="A" data-band="high"></div><button class="killbtn" id="A-kill-high" onclick="killEq('A','high')">HIGH</button></div>
            <div class="knob-cell"><div class="knob" id="A-eq-mid" data-deck="A" data-band="mid"></div><button class="killbtn" id="A-kill-mid" onclick="killEq('A','mid')">MID</button></div>
            <div class="knob-cell"><div class="knob" id="A-eq-low" data-deck="A" data-band="low"></div><button class="killbtn" id="A-kill-low" onclick="killEq('A','low')">LOW</button></div>
          </div>
          <div class="fader-cell">
            <div class="chmeter"><div class="chmeter-fill" id="A-chmeter"></div></div>
            <input class="vrange" id="A-vol" type="range" min="0" max="100" value="100"
                   oninput="post({cmd:'trim',deck:'A',value:this.value/100})">
          </div>
          <span class="strip-lab">VOL A</span>
        </div>

        <div class="mx-center">
          <div class="bass-swap">
            <div class="bs-lab">BASS SWAP</div>
            <div class="bs-row"><span class="bs-tag a">A</span><div class="bs-bar"><div class="bs-fill a" id="bs-A"></div></div></div>
            <div class="bs-row"><span class="bs-tag b">B</span><div class="bs-bar"><div class="bs-fill b" id="bs-B"></div></div></div>
          </div>
          <div class="xf-next"><span class="xf-next-l" id="xf-next-l">NEXT MIX</span><span class="xf-next-v" id="xf-next">0:00</span></div>
        </div>

        <div class="mx-ch b">
          <span class="ch-lab">CH B</span>
          <div class="knobs">
            <div class="knob-cell"><div class="knob" id="B-eq-high" data-deck="B" data-band="high"></div><button class="killbtn" id="B-kill-high" onclick="killEq('B','high')">HIGH</button></div>
            <div class="knob-cell"><div class="knob" id="B-eq-mid" data-deck="B" data-band="mid"></div><button class="killbtn" id="B-kill-mid" onclick="killEq('B','mid')">MID</button></div>
            <div class="knob-cell"><div class="knob" id="B-eq-low" data-deck="B" data-band="low"></div><button class="killbtn" id="B-kill-low" onclick="killEq('B','low')">LOW</button></div>
          </div>
          <div class="fader-cell">
            <input class="vrange" id="B-vol" type="range" min="0" max="100" value="100"
                   oninput="post({cmd:'trim',deck:'B',value:this.value/100})">
            <div class="chmeter"><div class="chmeter-fill" id="B-chmeter"></div></div>
          </div>
          <span class="strip-lab">VOL B</span>
        </div>
      </div>

      <div class="xfader-h" id="xfader">
        <span class="xf-end xf-a">A</span>
        <div class="xf-rail-h" id="xf-rail"><div class="xf-knob-h" id="xf-knob"></div></div>
        <span class="xf-end xf-b">B</span>
      </div>

      <div class="mx-transport">
        <button class="mx-btn cue" onclick="send('cue')">CUE NEXT</button>
        <button class="mx-btn skip" onclick="send('skip')">SKIP</button>
        <button class="mx-btn force" onclick="send('force')">FORCE MIX</button>
      </div>
    </div>

    <div class="deck b" id="deckB">
      <div class="deck-top">
        <div class="chips"><span class="chip-deck">DECK B</span><span class="status" id="B-status">IDLE</span></div>
        <div class="deck-leds">
          <span class="led sync" id="B-sync-led">SYNC</span>
          <span class="led play" id="B-play-led">PLAY</span>
        </div>
      </div>
      <div class="deck-title"><div class="t-name" id="B-title">&mdash;</div><div class="t-art" id="B-art"></div></div>
      <div class="bpm-row">
        <div class="bpm"><span class="bpm-v" id="B-bpm">0.0</span><span class="bpm-u">BPM</span></div>
        <div class="bm-side"><span class="bm-pill" id="B-bm">&bull; 0.0%</span><span class="bm-orig" id="B-orig"></span></div>
      </div>
      <div class="beatline">
        <div class="beats" id="B-beats"><span></span><span></span><span></span><span></span></div>
        <span class="phrasetxt" id="B-phrase">&mdash;</span>
      </div>
      <div class="wavewrap">
        <div class="wave" id="B-wave"></div>
        <div class="cue cue-in" id="B-cue-in"></div>
        <div class="cue cue-out" id="B-cue-out"></div>
        <div class="playhead" id="B-head"></div>
      </div>
      <div class="deck-foot"><span id="B-pos">0:00</span><span class="foot-mid" id="B-mid"></span><span id="B-dur">0:00</span></div>
      <div class="jog-row">
        <div class="pitch-col">
          <input class="vrange pitch" id="B-pitch" type="range" min="-80" max="80" value="0"
                 oninput="setPitch('B',this.value)">
          <span class="pitch-v" id="B-pitch-v">+0.0%</span>
          <span class="strip-lab">PITCH</span>
        </div>
        <div class="jog">
          <svg class="jog-ring" viewBox="0 0 100 100">
            <circle class="ring-bg" cx="50" cy="50" r="47"></circle>
            <circle class="ring-fg" id="B-ring" cx="50" cy="50" r="47"></circle>
          </svg>
          <div class="jog-platter" id="B-platter"><div class="jog-mark"></div></div>
          <div class="jog-center">
            <span class="jog-pos" id="B-jog-pos">0:00</span>
            <span class="jog-rem" id="B-jog-rem">-0:00</span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="midrow">
    <div class="crowd">
      <div class="cam" id="cam">
        <div class="cam-blobs"></div>
        <img id="cam-img" alt="">
        <div class="cam-tag"><span class="dot dot-live" id="cam-dot"></span><span id="cam-tag-t">SIM CAM</span></div>
        <div class="cam-foot" id="cam-foot">MOTION SENSE</div>
      </div>
      <div class="energy">
        <div class="en-head">
          <span class="en-title">CROWD ENERGY</span>
          <div class="crowd-toggle">
            <button class="seg-btn active" id="cm-auto" onclick="setCrowdMode(false)">AUTO</button>
            <button class="seg-btn" id="cm-manual" onclick="setCrowdMode(true)">MANUAL</button>
          </div>
        </div>
        <div class="en-big">
          <span class="en-val" id="crowd-big">0.00</span>
          <div class="en-cap"><div class="en-tier" id="en-tier">&mdash;</div><div class="en-sub" id="en-sub"></div></div>
        </div>
        <div class="meter-block" id="crowd-block">
          <div class="meter-lab"><span id="crowd-meter-lab">CROWD</span><span class="mval" id="crowd-val">0.00</span></div>
          <div class="segs" id="crowd-segs"></div>
        </div>
        <div class="meter-block">
          <div class="meter-lab"><span>TARGET <span class="bias" id="bias">+0.00</span></span><span class="mval" id="target-val">0.00</span></div>
          <div class="segs" id="target-segs"></div>
        </div>
      </div>
    </div>

    <div class="controls">
      <div class="ctl-head"><span class="ctl-title">MANUAL OVERRIDE</span><span class="auto-pill"><span class="dot dot-ok"></span>AUTOPILOT</span></div>
      <div class="ctl-hint">Jog-wheel transport &middot; CUE / SKIP / FORCE live on the mixer. Trim the room here.</div>
      <div class="nudge">
        <div class="nudge-lab">NUDGE ENERGY</div>
        <div class="nudge-row">
          <button class="nudge-btn cool" onclick="send('nudge',-0.1)">&minus; COOLER</button>
          <div class="bias-box"><div class="bias-v" id="bias2">+0.00</div><div class="bias-l">BIAS</div></div>
          <button class="nudge-btn hot" onclick="send('nudge',0.1)">HOTTER +</button>
        </div>
      </div>
    </div>
  </div>

  <div class="upnext">
    <div class="up-head">
      <div class="up-head-l"><svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg><span>UP NEXT</span></div>
      <div class="up-add">
        <input id="add-url" class="add-input" type="text" spellcheck="false"
               placeholder="paste a YouTube link to queue it…"
               onkeydown="if(event.key==='Enter')addUrl()">
        <button class="add-btn" onclick="addUrl()">+ ADD</button>
        <span class="up-head-r" id="up-meta">MIX QUEUE</span>
      </div>
    </div>
    <div class="cards" id="cards"></div>
  </div>

  <div class="logpanel">
    <div class="log-head"><span>AUTOPILOT DECISION LOG</span><span id="log-meta">LIVE FEED</span></div>
    <div class="log-body" id="log-body"></div>
  </div>

</div>

<script>
const $ = id => document.getElementById(id);
function fmt(s){ if(s==null||isNaN(s)) return "0:00"; s=Math.max(0,s|0);
  return Math.floor(s/60)+":"+String(s%60).padStart(2,"0"); }
function esc(x){ return String(x).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

// ---- control transport -------------------------------------------------
function post(o){
  fetch("/control",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify(o)}).catch(()=>{});
}
// Legacy helper kept so the existing buttons (skip/force/cue/pause/nudge) work
// unchanged: backend reads `value` then falls back to `delta`.
function send(cmd, delta){ post({cmd, delta: delta||0}); }
function setPitch(p, v){
  $(p+"-pitch-v").textContent = (v>=0?"+":"")+(v/10).toFixed(1)+"%";
  post({cmd:"bend", deck:p, value:v/1000});
}
function killEq(p, band){ post({cmd:"eq_kill", deck:p, band:band}); }
// Don't clobber a control the DJ is actively touching with an SSE frame.
function syncSlider(el, val){ if(el && document.activeElement!==el) el.value = val; }

// ---- rotary EQ knobs ---------------------------------------------------
// Each knob maps a gain of 0..1.5 onto a -135°..+135° dial sweep (unity = 12
// o'clock). Vertical drag turns it; double-click snaps back to unity. The
// engine's eq_manual is the source of truth — SSE frames repaint the dial
// unless the DJ is mid-grab on that exact knob.
const KNOB_MAX=1.5;
function knobAngle(g){ return -135 + (Math.max(0,Math.min(KNOB_MAX,g))/KNOB_MAX)*270; }
let draggingKnob=null;
function paintKnob(id, gain){
  const k=$(id); if(!k) return;
  k.dataset.gain=gain;
  const dial=k.querySelector(".knob-dial");
  if(dial) dial.style.transform="rotate("+knobAngle(gain)+"deg)";
  k.classList.toggle("killed", gain<=0.001);
}
function setKnob(id, gain){ if(draggingKnob!==id) paintKnob(id, gain); }   // from SSE
function initKnob(id){
  const k=$(id); if(!k) return;
  k.innerHTML='<div class="knob-dial"><div class="knob-ind"></div></div>';
  paintKnob(id, 1);
  let startY=0, startG=1;
  k.addEventListener("pointerdown", ev=>{
    draggingKnob=id; k.setPointerCapture(ev.pointerId);
    startY=ev.clientY; startG=parseFloat(k.dataset.gain||"1"); ev.preventDefault();
  });
  k.addEventListener("pointermove", ev=>{
    if(draggingKnob!==id) return;
    const g=Math.max(0,Math.min(KNOB_MAX, startG + (startY-ev.clientY)*0.01));
    paintKnob(id, g);
    post({cmd:"eq", deck:k.dataset.deck, band:k.dataset.band, value:g});
  });
  const end=ev=>{ if(draggingKnob===id){ draggingKnob=null;
    try{ k.releasePointerCapture(ev.pointerId); }catch(_){} } };
  k.addEventListener("pointerup", end);
  k.addEventListener("pointercancel", end);
  k.addEventListener("dblclick", ()=>{ paintKnob(id, 1);
    post({cmd:"eq", deck:k.dataset.deck, band:k.dataset.band, value:1}); });
}

function energyColor(e){
  if(e<0.2)return"#1E3A8A"; if(e<0.4)return"#14B8A6"; if(e<0.6)return"#84CC16";
  if(e<0.8)return"#F59E0B"; return"#FF2D6E";
}
function segColor(i){
  const f=i/19;
  if(f<0.2)return"#1E3A8A"; if(f<0.4)return"#14B8A6"; if(f<0.6)return"#84CC16";
  if(f<0.8)return"#F59E0B"; return"#FF2D6E";
}
function energyTier(e){
  if(e<0.2)return{label:"LOW ENERGY",sub:"warming up the room"};
  if(e<0.4)return{label:"BUILDING",sub:"energy is climbing"};
  if(e<0.6)return{label:"STEADY",sub:"holding the groove"};
  if(e<0.8)return{label:"HIGH ENERGY",sub:"crowd is locked in"};
  return{label:"PEAK",sub:"room is going off"};
}

function buildMeter(id){
  const c=$(id); c.innerHTML="";
  for(let i=0;i<20;i++) c.appendChild(document.createElement("span"));
}
function updateMeter(id,val){
  const lit=Math.round(Math.max(0,Math.min(1,val))*20), segs=$(id).children;
  for(let i=0;i<20;i++) segs[i].style.background = i<lit ? segColor(i) : "var(--seg-off)";
}
function buildWave(id,seed){
  const c=$(id); c.innerHTML=""; let r=seed>>>0;
  const rnd=()=>{ r=(r*1103515245+12345)&0x7fffffff; return r/0x7fffffff; };
  for(let i=0;i<64;i++){
    const t=i/63;
    const env=0.3+0.7*Math.abs(Math.sin(t*Math.PI*3))*(0.6+0.4*Math.abs(Math.cos(t*Math.PI*7)));
    const h=Math.max(5,Math.round(env*52*(0.6+0.5*rnd())));
    const b=document.createElement("span"); b.style.height=h+"px"; c.appendChild(b);
  }
}
function updateWave(id,frac){
  const lit=Math.round(Math.max(0,Math.min(1,frac))*64), bars=$(id).children;
  for(let i=0;i<64;i++) bars[i].classList.toggle("lit", i<lit);
}

function deck(p, d){
  $(p+"-status").textContent = d.role==="live"?"NOW PLAYING":(d.role==="cued"?"CUED · NEXT":"IDLE");
  let name=d.title||"", art="";
  const ix=name.indexOf(" - ");
  if(ix>0){ art=name.slice(0,ix); name=name.slice(ix+3); }
  $(p+"-title").textContent = name || "—";
  $(p+"-art").textContent = art;
  $(p+"-bpm").textContent = (d.bpm||0).toFixed(1);
  const base=d.base_bpm||0;
  const pct = base>0 ? (d.bpm/base - 1)*100 : 0;
  const arrow = pct>0.05?"▲":(pct<-0.05?"▼":"●");
  $(p+"-bm").textContent = arrow+" "+(pct>0?"+":"")+pct.toFixed(1)+"%";
  $(p+"-orig").textContent = base>0 ? "ORIG "+base.toFixed(1)+" BPM" : "";
  $(p+"-pos").textContent = fmt(d.position);
  $(p+"-dur").textContent = fmt(d.duration);
  $(p+"-mid").textContent = d.role==="live" ? "−"+fmt(d.remaining)+" TO MIX"
                          : (d.role==="cued" ? "CUE READY" : "");

  // Phrase cue markers over the waveform (mix-in past the intro, mix-out at
  // the outro). Hidden when there's no real structure (cue == 0 / == end).
  const dur=d.duration||0;
  const ci=$(p+"-cue-in"), co=$(p+"-cue-out");
  if(dur>0 && d.mix_in>0.5){ ci.style.left=(100*d.mix_in/dur)+"%"; ci.style.display="block"; }
  else ci.style.display="none";
  if(dur>0 && d.mix_out>0 && d.mix_out<dur-0.5){ co.style.left=(100*d.mix_out/dur)+"%"; co.style.display="block"; }
  else co.style.display="none";

  // SYNC/PLAY LEDs: PLAY tracks the playhead, SYNC lights whenever a track is
  // loaded (it's always beatmatched to the live deck on this engine).
  $(p+"-play-led").classList.toggle("on", !!d.playing && !paused);
  $(p+"-sync-led").classList.toggle("on", d.role!=="idle");

  // Reflect the live engine state back into the channel controls (unless the
  // DJ is mid-grab on that control). EQ is rotary; VOL is a fader.
  const eq=d.eq||{low:1,mid:1,high:1};
  setKnob(p+"-eq-high", eq.high);
  setKnob(p+"-eq-mid",  eq.mid);
  setKnob(p+"-eq-low",  eq.low);
  $(p+"-kill-high").classList.toggle("killed", eq.high<=0.001);
  $(p+"-kill-mid").classList.toggle("killed",  eq.mid<=0.001);
  $(p+"-kill-low").classList.toggle("killed",  eq.low<=0.001);
  syncSlider($(p+"-vol"), Math.round((d.trim==null?1:d.trim)*100));
  const bend=d.bend||0;
  syncSlider($(p+"-pitch"), Math.round(bend*1000));
  if(document.activeElement!==$(p+"-pitch"))
    $(p+"-pitch-v").textContent = (bend>=0?"+":"")+(bend*100).toFixed(1)+"%";

  // Hand the beat grid to the rAF loop, which extrapolates between snapshots
  // for a smooth playhead + beat pulse.
  setAnim(p, d);
}

function render(s){
  paused = !!s.paused;
  const pp=$("playpause");
  $("pp-ic").innerHTML = paused ? "&#9654;" : "&#9208;";   // ▶ resume / ⏸ pause
  $("pp-txt").textContent = paused ? "RESUME" : "PAUSE";
  pp.classList.toggle("paused", paused);

  deck("A", s.decks.A); deck("B", s.decks.B);

  const ga=s.decks.A.gain||0, gb=s.decks.B.gain||0, sum=ga+gb;
  const xpos = sum>0 ? gb/sum : (s.live==="B"?1:0);
  if(!draggingXf) $("xf-knob").style.left = (100*xpos)+"%";
  const lb = s.decks[s.live] ? s.decks[s.live].bpm : 0;
  $("sync-pill").textContent = "SYNC "+(lb||0).toFixed(1);
  const tr=s.transition, xf=$("xfader");
  if(tr && tr.active){
    xf.classList.add("mixing");
    $("xf-next-l").textContent = "MIXING " + tr.from + "→" + tr.to;
    $("xf-next").textContent = Math.round(tr.progress*100)+"%";
  } else {
    xf.classList.remove("mixing");
    $("xf-next-l").textContent = "NEXT MIX";
    $("xf-next").textContent = fmt(s.decks[s.live] ? s.decks[s.live].remaining : 0);
  }
  // Bass-swap meter: the low-band energy each deck is actually putting out
  // (DJ EQ × auto bass-swap). During a clean blend these cross over.
  $("bs-A").style.width = (100*(s.decks.A.eq.low * s.decks.A.bass_auto))+"%";
  $("bs-B").style.width = (100*(s.decks.B.eq.low * s.decks.B.bass_auto))+"%";

  const ce=s.crowd.energy, te=s.target_energy;
  crowdManual = !!s.crowd.manual;
  $("cm-auto").classList.toggle("active", !crowdManual);
  $("cm-manual").classList.toggle("active", crowdManual);
  $("crowd-block").classList.toggle("manual", crowdManual);
  $("crowd-segs").classList.toggle("fader", crowdManual);
  $("crowd-meter-lab").textContent = crowdManual
    ? "VIBE · DRAG TO SET · ROOM "+(s.crowd.sensor||0).toFixed(2) : "CROWD";
  $("mode-tag").textContent = crowdManual ? "MANUAL · DJ-STEERED" : "AUTOPILOT · CROWD-STEERED";
  // While the DJ is dragging the fader, don't let the SSE frame fight the grab.
  if(!draggingVibe){
    $("crowd-big").textContent = ce.toFixed(2);
    $("crowd-big").style.color = energyColor(ce);
    const tier=energyTier(ce);
    $("en-tier").textContent = tier.label; $("en-tier").style.color = energyColor(ce);
    $("en-sub").textContent = tier.sub;
    $("crowd-val").textContent = ce.toFixed(2);
    updateMeter("crowd-segs", ce);
  }
  $("target-val").textContent = te.toFixed(2);
  updateMeter("target-segs", te);
  const bias=s.energy_bias||0, bt=(bias>0?"+":"")+bias.toFixed(2);
  $("bias").textContent = bt; $("bias2").textContent = bt;

  const cam=$("cam");
  if(s.crowd.has_cam){
    cam.classList.add("has-cam");
    $("cam-img").src = "/frame.jpg?t="+Date.now();
    $("cam-dot").className = "dot dot-live";
    $("cam-tag-t").textContent = "LIVE CAM";
    $("cam-foot").textContent = "MOTION SENSE · LIVE";
  } else {
    cam.classList.remove("has-cam");
    $("cam-img").removeAttribute("src");
    $("cam-dot").className = "dot dot-faint";
    $("cam-tag-t").textContent = "SIM CAM";
    $("cam-foot").textContent = "MOTION SENSE · "+(s.crowd.mode||"").toUpperCase();
  }

  $("rotation").textContent = s.buffer.length+(s.buffer.length===1?" TRACK":" TRACKS");
  $("up-meta").textContent = "MIX QUEUE · "+s.buffer.length+" BUFFERED · LOOPING";

  const cards = s.buffer.slice(0,8).map((t,i)=>{
    const ec=energyColor(t.energy);
    let cls="card", badge;
    if(t.loaded){ cls+=" on"; badge='<span class="cbadge live">ON DECK</span>'; }
    else if(t.play_count>0){ badge='<span class="cbadge"><span class="dot dot-ok"></span>PLAYED '+t.play_count+'×</span>'; }
    else { badge='<span class="cbadge"><span class="dot dot-faint"></span>QUEUED</span>'; }
    let nm=t.name||"", art="";
    const ix=nm.indexOf(" - ");
    if(ix>0){ art=nm.slice(0,ix); nm=nm.slice(ix+3); }
    return '<div class="'+cls+'">'
      +'<div class="card-top"><span class="cnum">'+String(i+1).padStart(2,"0")+'</span>'+badge+'</div>'
      +'<div class="card-title"><div class="cname">'+esc(nm)+'</div><div class="cart">'+esc(art)+'</div></div>'
      +'<div class="card-bot"><div class="cbpm">'+t.bpm.toFixed(0)+'<small>BPM</small></div>'
      +'<div class="cen"><span class="cdot" style="background:'+ec+'"></span>'
      +'<span style="color:'+ec+'">'+t.energy.toFixed(2)+'</span></div></div></div>';
  }).join("");
  $("cards").innerHTML = cards || '<div class="card-empty">buffering…</div>';

  renderLog(s.log||[]);
}

function setCrowdMode(manual){ send("crowd_manual", manual?1:0); }

// Paste a YouTube link → enqueue it. Download + analysis happen server-side on
// a worker thread; the track shows up in UP NEXT and becomes a pick candidate
// once it's ready (watch the decision log for "[add] ready …").
function addUrl(){
  const el=$("add-url"); const u=(el.value||"").trim();
  if(!/^https?:\/\//i.test(u)){ el.placeholder="enter an http(s) link…"; el.value=""; return; }
  post({cmd:"add_url", url:u});
  el.value=""; el.blur(); el.placeholder="queued — fetching in the background…";
  setTimeout(()=>{ el.placeholder="paste a YouTube link to queue it…"; }, 5000);
}

// ---- autopilot decision log -------------------------------------------
function renderLog(items){
  const body=$("log-body"); if(!body) return;
  const rows = items.slice().reverse().map(it=>{
    const m=it.m||"";
    let cls="log-line";
    if(m.indexOf("[mix]")>=0) cls+=" mix";
    else if(m.indexOf("[cue]")>=0) cls+=" cue";
    else if(m.indexOf("[start]")>=0 || m.indexOf("[live]")>=0) cls+=" live";
    const d=new Date((it.t||0)*1000);
    const ts=String(d.getHours()).padStart(2,"0")+":"
            +String(d.getMinutes()).padStart(2,"0")+":"
            +String(d.getSeconds()).padStart(2,"0");
    return '<div class="'+cls+'"><span class="log-t">'+ts+'</span>'
         + '<span class="log-m">'+esc(m)+'</span></div>';
  }).join("");
  body.innerHTML = rows || '<div class="log-line"><span class="log-m">waiting…</span></div>';
}

// ---- live beat pulse + playhead (client-side extrapolation) ------------
// Each snapshot hands us a deck's playhead + beat grid; between snapshots we
// advance the playhead in wall-clock time scaled by the deck's varispeed rate,
// so the beat dots and playhead move at ~60fps instead of the 3Hz SSE rate.
let anim={}, paused=false;
function setAnim(p, d){
  anim[p] = {
    pos: d.position||0,
    rate: (d.base_bpm>0 ? d.bpm/d.base_bpm : 1),  // source-sec advanced per wall-sec
    off: d.beat_offset||0,
    per: d.beat_period||0,
    dur: d.duration||0,
    playing: !!d.playing,
    lvl: (d.gain||0)*(d.trim==null?1:d.trim),     // channel output level for the VU meter
    tcap: performance.now(),
  };
}
function pulse(){
  const now=performance.now();
  for(const p of ["A","B"]){
    const a=anim[p];
    const dots=$(p+"-beats").children;
    if(!a){ requestAnimationFrame(pulse); return; }
    let local=a.pos;
    if(a.playing && !paused) local += (now - a.tcap)/1000 * a.rate;
    const frac = a.dur>0 ? Math.max(0,Math.min(1, local/a.dur)) : 0;
    $(p+"-head").style.left = (100*frac)+"%";
    updateWave(p+"-wave", frac);

    // Jog wheel: ring fills with track progress, platter spins one turn per bar
    // (slowing to a stop when paused/stopped), center shows elapsed / -remaining.
    const ring=$(p+"-ring");
    if(ring){ const C=2*Math.PI*47;
      ring.style.strokeDasharray=C; ring.style.strokeDashoffset=C*(1-frac); }
    const plat=$(p+"-platter");
    if(plat){
      const ang = a.per>0 ? ((local-a.off)/(4*a.per))*360 : local*90;
      plat.style.transform="rotate("+(((ang%360)+360)%360)+"deg)";
    }
    if(a.dur>0){
      $(p+"-jog-pos").textContent = fmt(local);
      $(p+"-jog-rem").textContent = "-"+fmt(Math.max(0, a.dur-local));
    } else { $(p+"-jog-pos").textContent="0:00"; $(p+"-jog-rem").textContent="-0:00"; }

    // Channel VU: output level (gain×trim) pumped on the beat for a live feel.
    const mtr=$(p+"-chmeter");
    if(mtr){
      let h=0;
      if(a.playing && !paused){
        const ph = a.per>0 ? ((((local-a.off)%a.per)+a.per)%a.per)/a.per : 0.5;
        h=(a.lvl||0)*(0.45+0.55*(1-ph));
      }
      mtr.style.height=(100*Math.max(0,Math.min(1,h)))+"%";
    }

    if(a.per>0 && a.playing){
      const tb=local - a.off;
      const beatIx=Math.floor(tb/a.per);
      const phase=(((tb % a.per)+a.per)%a.per)/a.per;
      const beatInBar=((beatIx%4)+4)%4;
      const barInPhrase=((Math.floor(beatIx/4)%8)+8)%8;
      for(let i=0;i<4;i++){
        const on=(i===beatInBar);
        dots[i].style.opacity = on ? (0.30+0.70*(1-phase)) : 0.12;
        dots[i].classList.toggle("hit", on && phase<0.20);
      }
      $(p+"-phrase").textContent = "BAR "+(barInPhrase+1)+"/8 · BEAT "+(beatInBar+1)+"/4";
    } else {
      for(let i=0;i<4;i++){ dots[i].style.opacity=0.12; dots[i].classList.remove("hit"); }
      $(p+"-phrase").textContent = "—";
    }
  }
  requestAnimationFrame(pulse);
}
requestAnimationFrame(pulse);

// ---- crossfader scrub (drag the rail to push/pull a live transition) --
let draggingXf=false;
(function(){
  const rail=$("xf-rail");
  const frac=ev=>{ const r=rail.getBoundingClientRect();
    return Math.max(0,Math.min(1,(ev.clientX-r.left)/r.width)); };
  rail.addEventListener("pointerdown", ev=>{
    draggingXf=true; rail.setPointerCapture(ev.pointerId);
    const f=frac(ev); $("xf-knob").style.left=(100*f)+"%"; post({cmd:"xfade", value:f});
  });
  rail.addEventListener("pointermove", ev=>{ if(draggingXf){
    const f=frac(ev); $("xf-knob").style.left=(100*f)+"%"; post({cmd:"xfade", value:f}); } });
  const end=ev=>{ if(draggingXf){ draggingXf=false;
    try{ rail.releasePointerCapture(ev.pointerId); }catch(_){} } };
  rail.addEventListener("pointerup", end);
  rail.addEventListener("pointercancel", end);
})();

// Vibe fader: when manual is engaged, drag the CROWD meter to dictate the room.
let crowdManual=false, draggingVibe=false;
function vibeFrac(ev){
  const r=$("crowd-segs").getBoundingClientRect();
  return Math.max(0,Math.min(1,(ev.clientX-r.left)/r.width));
}
function vibeSet(ev){
  const f=vibeFrac(ev);
  updateMeter("crowd-segs", f);
  $("crowd-val").textContent=f.toFixed(2);
  $("crowd-big").textContent=f.toFixed(2);
  $("crowd-big").style.color=energyColor(f);
  const tier=energyTier(f);
  $("en-tier").textContent=tier.label; $("en-tier").style.color=energyColor(f);
  $("en-sub").textContent=tier.sub;
  send("crowd_set", f);
}
(function(){
  const m=$("crowd-segs");
  m.addEventListener("pointerdown", ev=>{
    if(!crowdManual) return;
    draggingVibe=true; m.setPointerCapture(ev.pointerId); vibeSet(ev);
  });
  m.addEventListener("pointermove", ev=>{ if(draggingVibe) vibeSet(ev); });
  const end=ev=>{ if(draggingVibe){ draggingVibe=false;
    try{ m.releasePointerCapture(ev.pointerId); }catch(_){} } };
  m.addEventListener("pointerup", end);
  m.addEventListener("pointercancel", end);
})();

buildWave("A-wave", 0x51ED); buildWave("B-wave", 0xB0A7);
buildMeter("crowd-segs"); buildMeter("target-segs");
["A","B"].forEach(p=>["high","mid","low"].forEach(b=>initKnob(p+"-eq-"+b)));

const t0=Date.now();
setInterval(()=>{
  const s=Math.floor((Date.now()-t0)/1000), h=Math.floor(s/3600), m=Math.floor(s%3600/60), ss=s%60;
  $("set-time").textContent = h+":"+String(m).padStart(2,"0")+":"+String(ss).padStart(2,"0");
}, 1000);

let es;
function connect(){
  es = new EventSource("/events");
  es.onopen = () => { const p=$("live-pill"); p.classList.add("live"); $("live-txt").textContent="LIVE"; };
  es.onmessage = e => { try{ render(JSON.parse(e.data)); }catch(_){} };
  es.onerror = () => { const p=$("live-pill"); p.classList.remove("live"); $("live-txt").textContent="OFFLINE"; };
}
connect();
</script>
</body>
</html>
"""
