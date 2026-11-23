"""SmileCheck SG (Expo) server: the Expo web build plus the screening API, reachable on the LAN.

The API is prototype/Daniel/server.py's, imported rather than copied: the same /api/predict,
/api/meta and /api/health functions, so this app and the original page always run the same
models, thresholds, video handling and result combining. All of that server's environment
settings (PROTO_CLASSIFIER, VIDEO_MAX_FRAMES, DEMO_WEIGHTS, ...) apply here too.

What this file adds:
  * serves the exported web app (`npx expo export -p web` -> dist/) at /
  * listens on all interfaces (PROTO_HOST, default 0.0.0.0) so a phone running Expo Go on the
    same Wi-Fi can reach it; prototype/Daniel/server.py only listens on 127.0.0.1
  * allows cross-origin calls from localhost and private LAN addresses, so the web app also
    works from the Expo dev server (`npx expo start --web`, port 8081)

Run from the repo root:  python "prototype/Daniel Expo/server.py"   (or run.bat)
"""

from __future__ import annotations

import importlib.util
import os
import socket
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
PROTO_SERVER = HERE.parent / "Daniel" / "server.py"
DIST = HERE / "dist"

_spec = importlib.util.spec_from_file_location("smilecheck_proto", PROTO_SERVER)
proto = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(proto)  # loads the classifier, mouth gate and detector once

app = FastAPI(title="SmileCheck SG (Expo)", docs_url="/api/docs", openapi_url="/api/openapi.json")

# Browsers on this computer or the LAN (Expo dev server, or another device's browser). Expo Go
# itself sends no Origin header, so CORS doesn't apply to it.
_LAN = r"(localhost|127\.0\.0\.1|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)"
app.add_middleware(CORSMiddleware, allow_origin_regex=rf"https?://{_LAN}(:\d+)?",
                   allow_methods=["GET", "POST"], allow_headers=["*"])


@app.middleware("http")
async def no_store(request, call_next):
    response = await call_next(request)
    if not request.url.path.startswith("/api"):
        response.headers["Cache-Control"] = "no-store"
    return response


app.add_api_route("/api/health", proto.health, methods=["GET"])
app.add_api_route("/api/meta", proto.meta, methods=["GET"])
app.add_api_route("/api/predict", proto.predict, methods=["POST"])

if (DIST / "index.html").exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="web")
else:
    @app.get("/", response_class=HTMLResponse)
    def no_build():
        return ("<!doctype html><meta charset=utf-8><title>SmileCheck SG</title>"
                "<body style='font-family:system-ui;padding:40px;max-width:640px'>"
                "<h1>The web app hasn't been built yet</h1>"
                "<p>The screening API is running. To serve the web app here, run this in "
                "<code>prototype/Daniel Expo</code>, then restart the server:</p>"
                "<pre>npm install\nnpx expo export -p web</pre>"
                "<p>Or use the dev server instead: <code>npx expo start --web</code>.</p>")


def lan_ip() -> str | None:
    """This computer's LAN address (the route to a private address; nothing is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
        return None if ip.startswith("127.") else ip
    except OSError:
        return None


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("PROTO_HOST", "0.0.0.0")
    port = int(os.environ.get("PROTO_PORT", "8713"))
    demo = proto.demo
    if not proto.CLASSIFIER_DEFAULT:
        print(f"classifier: off (PROTO_CLASSIFIER=0) | detector only, flag box conf>={demo.DETECTOR.conf:.2f}")
    elif demo.TASK == "binary":
        print(f"classifier: {demo.BACKBONE} issue gate on {demo.DEVICE} | flag p(issue)>={demo.ABSTAIN_THRESHOLD:.2f}")
    else:
        print(f"classifier: {demo.BACKBONE} ({len(demo.CLASSES)} classes) on {demo.DEVICE}")
    print("mouth gate: " + ("on" if demo.GATE else f"off ({demo.GATE_ERROR or 'DEMO_MOUTH_GATE=0'})"))
    print("detector: " + (f"on ({demo.DETECTOR.run}, conf>={demo.DETECTOR.conf:.2f})" if demo.DETECTOR
                          else f"off ({demo.DETECTOR_ERROR or 'DEMO_DETECTOR=0'})"))
    print("web app: " + ("dist/ (exported build)" if (DIST / "index.html").exists()
                         else "not built (run `npx expo export -p web` in prototype/Daniel Expo)"))
    print(f"SmileCheck SG: http://127.0.0.1:{port}")
    ip = lan_ip() if host == "0.0.0.0" else None
    if ip:
        print(f"On your network (phones, Expo Go): http://{ip}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
