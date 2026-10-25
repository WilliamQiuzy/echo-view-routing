#!/usr/bin/env python
"""a5c_expert_pilot_v1 private blind annotation server (standard library only; binds 127.0.0.1 only).

  python scripts/a5c_pilot_server.py --package <dir> --annotations <dir> --readers <readers.json> [--synthetic] [--port 8765]

Each reader opens http://127.0.0.1:<port>/?t=<their token> through an SSH tunnel. Before a reader submits the blind
record of a recording, no response carries a set name, public label, model output, source name or session; the reveal
layer is returned only after that reader asks to unblind. A reader's token reaches only that reader's own records.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a5c_pilot_records as rec  # noqa: E402

UI = Path(__file__).resolve().parent / "a5c_pilot_ui.html"
BLIND_KEYS = ("pid", "n_frames", "time_base", "pts_time", "fps_nominal", "fps_from_pts", "constant_rate", "source", "sha256", "resolution_note", "frame_ext")
PID = re.compile(r"^[A-Za-z0-9]{1,16}$")


class App:
    def __init__(self, package: Path, annotations: Path, readers: Path, synthetic: bool) -> None:
        self.package = Path(package); man = json.loads((self.package / "cases.json").read_text())
        if bool(man.get("synthetic")) != bool(synthetic):
            raise SystemExit("the package's synthetic flag differs from --synthetic")
        self.order, self.cases, self.synthetic = man["order"], man["cases"], bool(synthetic)
        self.reveal = json.loads((self.package / "reveal.json").read_text()) if (self.package / "reveal.json").is_file() else {}
        self.tokens = json.loads(Path(readers).read_text())["tokens"]; self.store = rec.Store(annotations, synthetic)

    def axis(self, pid: str) -> dict:
        return self.cases[pid]

    def blind_case(self, pid: str) -> dict:
        c = self.cases[pid]; return {k: c[k] for k in BLIND_KEYS if k in c}


def make_handler(app: App):
    class H(BaseHTTPRequestHandler):
        server_version = "a5c-pilot"

        def log_message(self, *a) -> None:  # no request log: paths carry neutral ids only, but keep stderr clean
            return

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff"); self.end_headers(); self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _reader(self, q: dict) -> str | None:
            t = (q.get("t") or [self.headers.get("X-Reader-Token", "")])[0]; return app.tokens.get(t)

        def _route(self, method: str) -> None:
            u = urlparse(self.path); q = parse_qs(u.query); parts = [p for p in u.path.split("/") if p]
            if method == "GET" and not parts:
                return self._send(200, UI.read_bytes(), "text/html; charset=utf-8")
            reader = self._reader(q)
            if reader is None:
                return self._json(403, {"error": "unknown reader token"})
            try:
                if method == "GET" and parts == ["api", "me"]:
                    st = {p: app.store.load(reader, p, app.axis(p))["state"] for p in app.order}
                    return self._json(200, {"reader": reader, "synthetic": app.synthetic, "order": app.order, "states": st})
                if len(parts) == 3 and parts[0] == "media" and PID.match(parts[1]) and parts[1] in app.cases:
                    m = re.match(r"^(\d{6})\.(png|jpg)$", parts[2])
                    if not m or int(m.group(1)) >= app.cases[parts[1]]["n_frames"]:
                        return self._json(404, {"error": "no such frame"})
                    f = app.package / "media" / parts[1] / parts[2]
                    return self._send(200, f.read_bytes(), "image/png" if m.group(2) == "png" else "image/jpeg")
                if len(parts) == 3 and parts[0] == "api" and PID.match(parts[2]) and parts[2] in app.cases:
                    pid = parts[2]; axis = app.axis(pid); r = app.store.load(reader, pid, axis)
                    if method == "GET" and parts[1] == "case":
                        return self._json(200, {"case": app.blind_case(pid), "record": r})
                    if method == "GET" and parts[1] == "reveal":
                        if not r["unblinded"]:
                            return self._json(403, {"error": "submit the blind record and unblind first"})
                        return self._json(200, {"reveal": app.reveal.get(pid, {})})
                    if method == "POST" and parts[1] == "save":
                        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0") or 0)) or b"{}")
                        r = rec.add_version(r, body.get("fields", {}), body.get("phase", "blind"), bool(body.get("submit")), axis["n_frames"]); app.store.save(r)
                        return self._json(200, {"record": r})
                    if method == "POST" and parts[1] == "unblind":
                        r = rec.unblind(r); app.store.save(r); return self._json(200, {"record": r})
                if method == "GET" and parts == ["api", "export"]:
                    rs = app.store.all(reader)
                    if (q.get("fmt") or ["json"])[0] == "csv":
                        return self._send(200, rec.to_csv(rs).encode(), "text/csv; charset=utf-8")
                    return self._json(200, {"records": rs})
            except rec.RecordError as e:
                return self._json(400, {"error": str(e)})
            return self._json(404, {"error": "not found"})

        def do_GET(self) -> None:
            self._route("GET")

        def do_POST(self) -> None:
            self._route("POST")

    return H


def serve(app: App, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True); ap.add_argument("--annotations", required=True); ap.add_argument("--readers", required=True)
    ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(); app = App(Path(a.package), Path(a.annotations), Path(a.readers), a.synthetic); httpd = serve(app, a.port)
    print(f"serving {len(app.order)} recordings on http://127.0.0.1:{a.port}/ ({'SYNTHETIC' if a.synthetic else 'real'} store {a.annotations})", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
