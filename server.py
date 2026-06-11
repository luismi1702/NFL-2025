"""
server.py — Backend mínimo para ejecutar scripts Python
Uso: python server.py   →  http://localhost:8765
"""
import os, sys, time, json, threading, subprocess
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, unquote

BASE = Path(__file__).parent
jobs = {}
lock = threading.Lock()


def run_job(jid, script_file, stdin_data):
    path = BASE / script_file
    if not path.exists():
        with lock:
            jobs[jid] = {"s": "err", "log": f"No encontrado: {script_file}", "imgs": []}
        return
    # Snapshot antes: nombre → mtime
    before = {p.name: p.stat().st_mtime for p in BASE.glob("*.png")}
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        r = subprocess.run(
            [sys.executable, str(path)],
            input=stdin_data, capture_output=True, text=True, encoding="utf-8",
            cwd=str(BASE), timeout=600, env=env,
        )
        log = r.stdout + ("\nSTDERR:\n" + r.stderr if r.stderr.strip() else "")
        st  = "ok" if r.returncode == 0 else "err"
    except subprocess.TimeoutExpired:
        log, st = "Timeout (>10 min)", "err"
    except Exception as e:
        log, st = str(e), "err"
    # PNG incluido si es NUEVO o si su mtime cambió (sobreescrito)
    after = {p.name: p.stat().st_mtime for p in BASE.glob("*.png")}
    imgs = sorted(
        [name for name, mtime in after.items()
         if name not in before or mtime != before[name]],
        key=lambda f: after[f], reverse=True,
    )
    with lock:
        jobs[jid] = {"s": st, "log": log, "imgs": imgs}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass   # silencia logs

    def _send(self, data, ct="application/json", code=200):
        b = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", len(b))
        self.end_headers()
        self.wfile.write(b)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        p = urlparse(self.path).path

        if p == "/api/outputs":
            pngs = sorted(BASE.glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True)
            self._send([x.name for x in pngs])

        elif p.startswith("/api/job/"):
            jid = p.split("/api/job/")[-1]
            with lock:
                self._send(jobs.get(jid, {"s": "?", "log": "", "imgs": []}))

        elif p.startswith("/img/"):
            name = unquote(p[5:])
            fp   = BASE / name
            if fp.exists() and fp.suffix.lower() == ".png":
                data = fp.read_bytes()
                self._send(data, "image/png")
            else:
                self._send(b"Not found", "text/plain", 404)

        elif p == "/" or p.endswith(".html"):
            name = "index.html" if p == "/" else p.lstrip("/")
            fp   = BASE / name
            if fp.exists():
                self._send(fp.read_bytes(), "text/html; charset=utf-8")
            else:
                self._send(b"Not found", "text/plain", 404)

        else:
            self._send(b"Not found", "text/plain", 404)

    def do_POST(self):
        p = urlparse(self.path).path
        if p == "/api/run":
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length))
            script = body.get("file", "")
            inputs = body.get("inputs", [])
            stdin  = "\n".join(str(v) for v in inputs) + ("\n" if inputs else "")
            jid    = f"j{int(time.time()*1000)}"
            with lock:
                jobs[jid] = {"s": "run", "log": "", "imgs": []}
            threading.Thread(target=run_job, args=(jid, script, stdin), daemon=True).start()
            self._send({"jid": jid})
        else:
            self._send(b"Not found", "text/plain", 404)


if __name__ == "__main__":
    port = 8765
    print(f"\n  @CuartayDato NFL 2025  >>  http://localhost:{port}\n")
    import webbrowser; webbrowser.open(f"http://localhost:{port}/galeria.html")
    HTTPServer(("127.0.0.1", port), H).serve_forever()
