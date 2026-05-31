"""Web dashboard: a thin state-broadcast + control layer over the live engine.

Stdlib only (http.server + Server-Sent Events) to keep the project dep-light.
The engine already runs in-process, so this just snapshots the
controller/mixer/crowd/library each tick and streams JSON to the browser, while
POSTs from the browser inject control commands (skip, force-transition, nudge).

Endpoints:
    GET  /            the single-page dashboard
    GET  /events      SSE stream of engine snapshots (~3 Hz)
    GET  /frame.jpg   latest crowd-cam frame (204 when simulated / no camera)
    POST /control     {"cmd": "skip" | "force" | "nudge" | "crowd_manual"
                              | "crowd_set", "delta": <float>}
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
        return {
            "role": role,
            "title": d.title or (track.name if track else ""),
            "base_bpm": round(d.analysis.bpm, 1) if d.analysis else 0.0,
            "bpm": round(d.effective_bpm, 1),
            "energy": round(track.energy, 3) if track else None,
            "gain": round(d.gain, 3),
            "playing": d.playing,
            "position": round(d.position_sec, 1),
            "duration": round(d.analysis.duration, 1) if d.analysis else 0.0,
            "remaining": round(d.remaining_sec, 1),
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
        }

    def handle_command(self, cmd: str, delta: float = 0.0) -> None:
        if cmd == "skip":
            self.controller.request_skip()
        elif cmd == "force":
            self.controller.request_transition()
        elif cmd == "nudge":
            self.controller.nudge_energy(delta)
        elif cmd == "crowd_manual":
            self.controller.set_crowd_manual(delta >= 0.5)
        elif cmd == "crowd_set":
            self.controller.set_crowd_energy(delta)

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
                    dashboard.handle_command(str(payload.get("cmd", "")),
                                             float(payload.get("delta", 0.0)))
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
  .decks{display:flex;gap:16px;height:440px;}
  .deck{flex:1;min-width:0;background:var(--bg-panel);border-radius:18px;padding:24px;
    display:flex;flex-direction:column;justify-content:space-between;}
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

  @media (max-width:1120px){
    .decks{flex-direction:column;height:auto;}
    .xfader{width:auto;min-height:160px;}
    .midrow{flex-direction:column;height:auto;}
    .crowd{flex-direction:column;}
    .cam{width:auto;height:160px;}
    .controls{width:auto;}
  }
  @media (max-width:700px){
    .cards{flex-wrap:wrap;} .card{flex:1 1 140px;}
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
      <div class="live-pill" id="live-pill"><span class="dot" id="live-dot"></span><span id="live-txt">CONNECTING</span></div>
    </div>
  </div>

  <div class="decks">
    <div class="deck a" id="deckA">
      <div class="deck-head">
        <div class="chips"><span class="chip-deck">DECK A</span><span class="status" id="A-status">IDLE</span></div>
        <span class="deck-ch">A</span>
      </div>
      <div class="deck-title"><div class="t-name" id="A-title">&mdash;</div><div class="t-art" id="A-art"></div></div>
      <div class="bpm-row">
        <div class="bpm"><span class="bpm-v" id="A-bpm">0.0</span><span class="bpm-u">BPM</span></div>
        <div class="bm-side"><span class="bm-pill" id="A-bm">&bull; 0.0%</span><span class="bm-orig" id="A-orig"></span></div>
      </div>
      <div class="wave" id="A-wave"></div>
      <div class="deck-foot"><span id="A-pos">0:00</span><span class="foot-mid" id="A-mid"></span><span id="A-dur">0:00</span></div>
    </div>

    <div class="xfader" id="xfader">
      <div class="xf-label">CROSSFADE</div>
      <div class="xf-track">
        <span class="xf-end xf-a">A</span>
        <div class="xf-rail"><div class="xf-knob" id="xf-knob"></div></div>
        <span class="xf-end xf-b">B</span>
      </div>
      <div class="xf-info">
        <div class="sync-pill" id="sync-pill">SYNC 0.0</div>
        <div class="xf-next"><span class="xf-next-l" id="xf-next-l">NEXT MIX</span><span class="xf-next-v" id="xf-next">0:00</span></div>
        <div class="xf-sub" id="xf-sub">BEAT SYNC</div>
      </div>
    </div>

    <div class="deck b" id="deckB">
      <div class="deck-head">
        <div class="chips"><span class="chip-deck">DECK B</span><span class="status" id="B-status">IDLE</span></div>
        <span class="deck-ch">B</span>
      </div>
      <div class="deck-title"><div class="t-name" id="B-title">&mdash;</div><div class="t-art" id="B-art"></div></div>
      <div class="bpm-row">
        <div class="bpm"><span class="bpm-v" id="B-bpm">0.0</span><span class="bpm-u">BPM</span></div>
        <div class="bm-side"><span class="bm-pill" id="B-bm">&bull; 0.0%</span><span class="bm-orig" id="B-orig"></span></div>
      </div>
      <div class="wave" id="B-wave"></div>
      <div class="deck-foot"><span id="B-pos">0:00</span><span class="foot-mid" id="B-mid"></span><span id="B-dur">0:00</span></div>
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
      <div class="ctl-actions">
        <button class="big-btn" onclick="send('skip')"><svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 4 15 12 5 20 5 4"/><line x1="19" y1="5" x2="19" y2="19"/></svg><span>SKIP</span></button>
        <button class="big-btn force" onclick="send('force')"><svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="18" r="3"/><circle cx="6" cy="6" r="3"/><path d="M6 9v3a9 9 0 0 0 9 9"/></svg><span>FORCE MIX</span></button>
      </div>
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
      <span class="up-head-r" id="up-meta">MIX QUEUE</span>
    </div>
    <div class="cards" id="cards"></div>
  </div>

</div>

<script>
const $ = id => document.getElementById(id);
function fmt(s){ if(s==null||isNaN(s)) return "0:00"; s=Math.max(0,s|0);
  return Math.floor(s/60)+":"+String(s%60).padStart(2,"0"); }
function esc(x){ return String(x).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

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
  updateWave(p+"-wave", d.duration>0 ? d.position/d.duration : 0);
}

function render(s){
  deck("A", s.decks.A); deck("B", s.decks.B);

  const ga=s.decks.A.gain||0, gb=s.decks.B.gain||0, sum=ga+gb;
  const pos = sum>0 ? gb/sum : (s.live==="B"?1:0);
  $("xf-knob").style.top = (100*pos)+"%";
  const lb = s.decks[s.live] ? s.decks[s.live].bpm : 0;
  $("sync-pill").textContent = "SYNC "+(lb||0).toFixed(1);
  const tr=s.transition, xf=$("xfader");
  if(tr && tr.active){
    xf.classList.add("mixing");
    $("xf-next-l").textContent="MIXING";
    $("xf-next").textContent=Math.round(tr.progress*100)+"%";
    $("xf-sub").textContent=tr.from+" → "+tr.to;
  } else {
    xf.classList.remove("mixing");
    $("xf-next-l").textContent="NEXT MIX";
    $("xf-next").textContent=fmt(s.decks[s.live] ? s.decks[s.live].remaining : 0);
    $("xf-sub").textContent="BEAT SYNC";
  }

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
}

function send(cmd, delta){
  fetch("/control",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({cmd, delta: delta||0})});
}

function setCrowdMode(manual){ send("crowd_manual", manual?1:0); }

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
