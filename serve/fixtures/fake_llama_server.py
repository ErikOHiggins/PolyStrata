"""A stand-in for llama.cpp's llama-server, for serve/test_llama_server.py: the parts of its HTTP API that
serve/server.py's LlamaServerEngine uses, as llama-server answers them at the pinned commit (08c252e22):

    GET /health        503 {"error": {..."Loading model"...}} while loading, then 200 {"status": "ok"} (no key needed)
    GET /props         default_generation_settings.n_ctx, total_slots, modalities, media_marker, build_info
    POST /completion   a token-id prompt; with "stream": an SSE stream (chunked) of prompt-progress events, one event
                       per token ({"content", "tokens": [id], "tokens_predicted"}) and a last one with "stop": true

Like llama-server it sends a token only once the text ends on a whole character (a character split over tokens comes
with its last token), and it ends a reply at its end-of-generation tokens.  The key is LLAMA_API_KEY.

    python fake_llama_server.py --host 127.0.0.1 --port N --fake-vocab vocab.json --fake-reply reply.json
        [--fake-eog 1,2] [--fake-load-s 0.2] [--fake-ctx 4096] [--fake-slots 1] [--fake-delay-s 0]
        [--fake-record log.jsonl] [--fake-die-after N] [--fake-bind-fail-once marker] [--fake-exit-at-start]

vocab.json: a list of each token's bytes as hex, and "special" ids (rendered as "" like llama.cpp's control tokens);
reply.json: the token ids of each reply, in turn (the last one repeats).  --fake-record writes each request (its body,
whether it had the key, and how it ended) as a line of JSON.  Other arguments (llama-server's own) are ignored.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--host", required=True)
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--fake-vocab", required=True)
ap.add_argument("--fake-reply", required=True)
ap.add_argument("--fake-eog", default="")
ap.add_argument("--fake-load-s", type=float, default=0.0)
ap.add_argument("--fake-ctx", type=int, default=4096)
ap.add_argument("--fake-slots", type=int, default=1)
ap.add_argument("--fake-delay-s", type=float, default=0.0)
ap.add_argument("--fake-record")
ap.add_argument("--fake-die-after", type=int)
ap.add_argument("--fake-bind-fail-once")
ap.add_argument("--fake-exit-at-start", action="store_true")
a, rest = ap.parse_known_args()

print(f"fake llama-server: args {sys.argv[1:]}", flush=True)
if a.fake_exit_at_start:
    print("error: the model could not be loaded", flush=True)
    sys.exit(1)
if a.fake_bind_fail_once and not os.path.exists(a.fake_bind_fail_once):
    open(a.fake_bind_fail_once, "w").close()
    print(f"srv    start: couldn't bind HTTP server socket, hostname: {a.host}, port: {a.port}", flush=True)
    sys.exit(1)

vocab = json.load(open(a.fake_vocab, encoding="utf-8"))
PIECES = [bytes.fromhex(h) for h in vocab["tokens"]]
SPECIAL = set(vocab.get("special", []))
REPLIES = json.load(open(a.fake_reply, encoding="utf-8"))
EOG = {int(x) for x in a.fake_eog.split(",") if x.strip()}
KEY = os.environ.get("LLAMA_API_KEY", "")
loaded_at = time.monotonic() + a.fake_load_s
state = {"turn": 0, "cached": []}
lock = threading.Lock()


def record(entry: dict) -> None:
    if a.fake_record:
        with lock, open(a.fake_record, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def incomplete(text: bytes) -> bool:
    """llama-server's validate_utf8: the text ends inside a multi-byte character."""
    for i in range(1, min(4, len(text)) + 1):
        c = text[-i]
        if (c & 0xE0) == 0xC0 and i < 2 or (c & 0xF0) == 0xE0 and i < 3 or (c & 0xF8) == 0xF0 and i < 4:
            return True
    return False


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, status: int, obj) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _keyed(self) -> bool:
        return not KEY or self.headers.get("Authorization", "") == f"Bearer {KEY}"

    def do_GET(self):
        if self.path == "/health":
            if time.monotonic() < loaded_at:
                return self._send(503, {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}})
            return self._send(200, {"status": "ok"})
        if not self._keyed():
            return self._send(401, {"error": {"code": 401, "message": "Invalid API Key", "type": "authentication_error"}})
        if self.path == "/props":
            return self._send(200, {"default_generation_settings": {"n_ctx": a.fake_ctx, "params": {}},
                                    "total_slots": a.fake_slots, "modalities": {"vision": False, "audio": False},
                                    "media_marker": "<__media__>", "build_info": "b0-fake"})
        self._send(404, {"error": {"code": 404, "message": "File Not Found", "type": "not_found_error"}})

    def _chunk(self, obj) -> None:
        data = b"data: " + json.dumps(obj).encode("utf-8") + b"\n\n"
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self.wfile.flush()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        entry = {"path": self.path, "keyed": self._keyed(), "body": body}
        if not self._keyed():
            record({**entry, "end": "401"})
            return self._send(401, {"error": {"code": 401, "message": "Invalid API Key", "type": "authentication_error"}})
        prompt = body.get("prompt") or []
        if len(prompt) >= a.fake_ctx:
            record({**entry, "end": "400"})
            return self._send(400, {"error": {"code": 400, "message": "the request exceeds the available context size, "
                                              "try increasing it", "type": "exceed_context_size_error"}})
        with lock:
            reply = REPLIES[min(state["turn"], len(REPLIES) - 1)]
            state["turn"] += 1
            cached = state["cached"]
        cache = 0
        while cache < min(len(cached), len(prompt) - 1) and cached[cache] == prompt[cache]:
            cache += 1
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        sent_text, text, generated, n = 0, b"", [], 0
        stop_type = "limit"
        try:
            if body.get("return_progress"):
                for done in [*range(cache, len(prompt), 4), len(prompt)]:      # per batch, then the whole prompt
                    self._chunk({"content": "", "tokens": [-12345], "stop": False, "tokens_predicted": 0,
                                 "tokens_evaluated": len(prompt),
                                 "prompt_progress": {"total": len(prompt), "cache": cache,
                                                     "processed": max(done, cache), "time_ms": 10}})
            for t in reply[:body.get("n_predict", -1) if body.get("n_predict", -1) >= 0 else None]:
                if a.fake_delay_s:
                    time.sleep(a.fake_delay_s)
                if a.fake_die_after is not None and n >= a.fake_die_after:
                    os._exit(3)
                n += 1
                generated.append(t)
                text += b"" if t in SPECIAL else PIECES[t]
                if not incomplete(text):
                    piece, sent_text = text[sent_text:], len(text)
                    self._chunk({"content": piece.decode("utf-8", "replace"), "tokens": [t], "stop": False,
                                 "tokens_predicted": n, "tokens_evaluated": len(prompt)})
                if t in EOG:
                    stop_type = "eos"
                    break
            with lock:
                state["cached"] = list(prompt) + generated
            record({**entry, "end": stop_type, "generated": n})   # before the last event: a test reads it after that
            self._chunk({"content": "", "tokens": [], "stop": True, "tokens_predicted": n,
                         "tokens_evaluated": len(prompt), "stop_type": stop_type, "truncated": False,
                         "tokens_cached": len(prompt) + n,
                         "timings": {"cache_n": cache, "prompt_n": len(prompt) - cache, "prompt_ms": 12.5,
                                     "predicted_n": n, "predicted_ms": 40.0}})
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            with lock:
                state["cached"] = list(prompt) + generated
            record({**entry, "end": "client left", "generated": n})


ThreadingHTTPServer.daemon_threads = True
ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
