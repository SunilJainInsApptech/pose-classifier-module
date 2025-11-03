# Lightweight Flask API to control the Jetson stream manager remotely.
# Relies on the functions in jetson_stream_manager.py (start_stream, stop_stream, etc).
# Run: pip install flask
# Secure with API_SECRET environment variable (see below).

from flask import Flask, request, jsonify
import os
import signal
import threading
import logging
import importlib

# Import the existing stream manager module (does not run main because it guards with __main__)
import jetson_stream_manager as jsm

# --- Configuration ---
API_SECRET = os.getenv("API_SECRET", "please-set-a-secret")  # override in env for security
HOST = os.getenv("JETSON_API_HOST", "0.0.0.0")
PORT = int(os.getenv("JETSON_API_PORT", "5000"))

app = Flask(__name__)
lock = threading.Lock()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def require_secret(req):
    hdr = req.headers.get("X-API-SECRET") or req.headers.get("Api-Secret")
    return hdr and hdr == API_SECRET


@app.before_request
def check_secret():
    # allow health check without secret
    if request.path == "/health":
        return
    if not require_secret(request):
        return jsonify({"ok": False, "error": "Unauthorized"}), 401


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "jetson_api_service"})


@app.route("/stream/start", methods=["POST"])
def api_start_stream():
    data = request.get_json(silent=True) or {}
    camera_id = data.get("camera_id")
    if not camera_id:
        return jsonify({"ok": False, "error": "camera_id required"}), 400

    with lock:
        # refresh module in case code changed on disk during development
        importlib.reload(jsm)

        # start_stream returns output directory on success
        output = jsm.start_stream(camera_id)
        if not output:
            return jsonify({"ok": False, "error": f"failed to start stream {camera_id}"}), 500

        return jsonify({
            "ok": True,
            "camera_id": camera_id,
            "output_dir": output,
            "message": "stream started"
        })


@app.route("/stream/stop", methods=["POST"])
def api_stop_stream():
    with lock:
        importlib.reload(jsm)
        # stop_stream will stop rsync and ffmpeg
        jsm.stop_stream()
        return jsonify({"ok": True, "message": "stream stopped"})


@app.route("/stream/status", methods=["GET"])
def api_status():
    # read live state from the module
    ff = jsm.ffmpeg_process
    rs = jsm.rsync_process
    active = jsm.active_camera_id

    running = False
    pid = None
    if ff and getattr(ff, "poll", lambda: 1)() is None:
        running = True
        pid = getattr(ff, "pid", None)

    return jsonify({
        "ok": True,
        "active_camera": active,
        "ffmpeg_running": running,
        "ffmpeg_pid": pid,
        "rsync_thread_alive": bool(rs and getattr(rs, "is_alive", lambda: False)()),
    })


def shutdown_handler(signum, frame):
    logging.info("Signal received, stopping stream and exiting.")
    try:
        jsm.stop_stream()
    except Exception:
        logging.exception("Error stopping stream during shutdown.")
    # terminate Flask gracefully
    os._exit(0)


if __name__ == "__main__":
    # install signal handlers to stop ffmpeg/rsync on process termination
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    logging.info(f"Starting Jetson API service on {HOST}:{PORT}")
    logging.info("Set API_SECRET env var and include X-API-SECRET header in requests.")
    app.run(host=HOST, port=PORT, threaded=True)