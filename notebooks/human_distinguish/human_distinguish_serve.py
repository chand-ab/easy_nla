"""Serve the distinguish page locally so its progress lands in this folder.

Browser storage turned out not to be a place a run can live: Brave rebuilt its
localStorage database twice in one day and took a half-finished run with it
each time. Served from here instead, the page PUTs its state to

    results_progress.json

after every answer and reads it back on load, so a closed tab, a cleared cache
or a different browser all resume at the right trial. The file has the same
shape as the final export, so once all 42 are answered it IS results.json.

Only the page itself is served -- never the key -- and only on the loopback
interface.

Usage:
    python human_distinguish_serve.py            # opens the browser at 127.0.0.1:8765
    python human_distinguish_score.py results_progress.json
"""

import argparse
import json
import os
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE = HERE / "human_distinguish.html"
PROGRESS = HERE / "results_progress.json"


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body=b"", ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/human_distinguish.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/progress":
            if PROGRESS.exists():
                self._send(200, PROGRESS.read_bytes())
            else:
                self._send(404, b"{}")
        else:
            self._send(404, b"{}")

    def do_PUT(self):
        if self.path != "/progress":
            return self._send(404, b"{}")
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            doc = json.loads(body)
            assert isinstance(doc.get("answers"), list) and doc.get("build")
        except (ValueError, AssertionError):
            return self._send(400, b'{"error":"not a progress document"}')
        # Write-then-rename so a crash mid-write cannot leave a truncated file
        # where the only copy of the run used to be.
        tmp = PROGRESS.with_suffix(".json.tmp")
        tmp.write_bytes(body)
        os.replace(tmp, PROGRESS)
        self._send(200, b'{"ok":true}')

    def log_message(self, fmt, *args):
        if "/progress" not in (args[0] if args else ""):
            super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"serving {PAGE.name} at {url}\nprogress -> {PROGRESS}\nCtrl-C to stop")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
