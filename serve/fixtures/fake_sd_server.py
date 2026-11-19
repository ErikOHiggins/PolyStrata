"""A stand-in for stable-diffusion.cpp's sd-server (serve/test_images.py): the parts of the native API the Web UI and
serve/images.py use - capabilities, recipes, model-files, models/status|load|unload, img_gen and its jobs (queued ->
generating -> completed with one small PNG per image), job cancel - on the --listen-ip / --listen-port it is given.
A load checks that the files are in --model-dir.

FAKE_SD_ARGS=<file>: writes its argument list there (JSON).  FAKE_SD_JOB_S=<s>: how long a job generates (0.3).
FAKE_SD_FAIL=1: jobs fail.  FAKE_SD_EXIT=1: exits 1 at its start, after a log line.  FAKE_SD_LOAD_S=<s>: a load's time.
"""
import base64
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def arg(name, default=None):
    a = sys.argv
    return a[a.index(name) + 1] if name in a else default


STATE = {"state": "empty", "recipe": None, "models": {}, "error": None, "revision": 0, "load_elapsed_seconds": None}
JOBS = {}
LOCK = threading.Lock()
MODEL_DIR = Path(arg("--model-dir", "models"))


def run_job(job_id, batch):
    time.sleep(0.05)
    with LOCK:
        job = JOBS[job_id]
        if job["status"] == "cancelled":
            return
        job.update(status="generating", started=int(time.time() * 1000))
    total = 4
    for step in range(total):
        time.sleep(float(os.environ.get("FAKE_SD_JOB_S", "0.3")) / total)
        with LOCK:
            if job["status"] == "cancelled":
                return
            job["progress"] = {"stage": "sampling", "detail": "", "pass": 1, "completed": step + 1, "total": total,
                               "unit": "steps", "stage_elapsed_ms": 1, "elapsed_ms": 1}
    with LOCK:
        if os.environ.get("FAKE_SD_FAIL"):
            job.update(status="failed", completed=int(time.time() * 1000),
                       error={"code": "generation_failed", "message": "generate_image returned empty results"})
        else:
            job.update(status="completed", completed=int(time.time() * 1000),
                       result={"output_format": "png", "images": [
                           {"index": i, "b64_json": base64.b64encode(PNG).decode()} for i in range(batch)]})


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, obj, ctype="application/json"):
        data = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", self.headers.get("Origin") or "*")   # as sd-server does
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/":
            return self.send(200, b"Stable Diffusion Server is running", "text/html")
        if p == "/sdcpp/v1/models/status":
            with LOCK:
                return self.send(200, dict(STATE))
        if p == "/sdcpp/v1/capabilities":
            return self.send(200, {"modes": ["img_gen"], "runtime": dict(STATE), "query": self.path.partition("?")[2]})
        if p == "/sdcpp/v1/recipes":
            return self.send(200, {"recipes": [{"id": "flux2-klein-4b", "name": "FLUX.2 Klein 4B"}], "errors": []})
        if p == "/sdcpp/v1/model-files":
            files = [{"id": f.relative_to(MODEL_DIR).as_posix()} for f in MODEL_DIR.rglob("*") if f.is_file()]
            return self.send(200, {"files": files, "errors": []})
        if p.startswith("/sdcpp/v1/jobs/"):
            with LOCK:
                job = JOBS.get(p.rsplit("/", 1)[1])
                return self.send(200, dict(job)) if job else self.send(404, {"error": "job not found"})
        self.send(404, {"error": "not found"})

    def do_POST(self):
        p = self.path.split("?")[0]
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        if p == "/sdcpp/v1/models/load":
            missing = [v for v in (body.get("models") or {}).values() if not (MODEL_DIR / v).is_file()]
            if missing:
                return self.send(400, {"error": f"unknown model or incorrect category: {missing[0]}",
                                       "code": "invalid_model_config"})
            time.sleep(float(os.environ.get("FAKE_SD_LOAD_S", "0")))
            with LOCK:
                STATE.update(state="ready", recipe=body.get("recipe"), models=body.get("models") or {},
                             revision=STATE["revision"] + 1, load_elapsed_seconds=0.1)
                return self.send(200, dict(STATE))
        if p == "/sdcpp/v1/models/unload":
            with LOCK:
                STATE.update(state="empty", recipe=None, models={}, revision=STATE["revision"] + 1)
                return self.send(200, dict(STATE))
        if p == "/sdcpp/v1/img_gen":
            with LOCK:
                if STATE["state"] != "ready":
                    return self.send(503, {"error": "No model is loaded", "code": "model_not_loaded"})
                job_id = "job_" + uuid.uuid4().hex[:12]
                JOBS[job_id] = {"id": job_id, "kind": "img_gen", "status": "queued", "created": int(time.time() * 1000),
                                "started": None, "completed": None, "queue_position": 0, "result": None,
                                "error": None, "progress": {"stage": "queued"}, "prompt": body.get("prompt")}
            threading.Thread(target=run_job, args=(job_id, int(body.get("batch_count") or 1)), daemon=True).start()
            return self.send(202, {"id": job_id, "kind": "img_gen", "status": "queued",
                                   "created": JOBS[job_id]["created"], "poll_url": f"/sdcpp/v1/jobs/{job_id}"})
        if p.startswith("/sdcpp/v1/jobs/") and p.endswith("/cancel"):
            with LOCK:
                job = JOBS.get(p.split("/")[-2])
                if not job:
                    return self.send(404, {"error": "job not found"})
                if job["status"] in ("queued", "generating"):
                    job.update(status="cancelled", completed=int(time.time() * 1000),
                               error={"code": "cancelled", "message": "job cancelled by client"})
                return self.send(200, dict(job))
        self.send(404, {"error": "not found"})


def main():
    if os.environ.get("FAKE_SD_ARGS"):
        Path(os.environ["FAKE_SD_ARGS"]).write_text(json.dumps(sys.argv[1:]))
    if os.environ.get("FAKE_SD_EXIT"):
        print("ggml_vulkan: ErrorDeviceLost (a fake failure)", flush=True)
        sys.exit(1)
    print(f"listening on: http://{arg('--listen-ip')}:{arg('--listen-port')}", flush=True)
    ThreadingHTTPServer((arg("--listen-ip", "127.0.0.1"), int(arg("--listen-port", "1234"))), H).serve_forever()


if __name__ == "__main__":
    main()
