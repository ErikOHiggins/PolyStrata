"""serve/images.py - image generation: stable-diffusion.cpp's own server (sd-server) and its Web UI behind Puddle.

The web app's Images tab shows sd.cpp's Web UI (sdcpp-webui, built with sd-server by setup --images flux2-klein) at
/sd/.  The page calls sd-server's native API relative to its own address (/sd/sdcpp/v1/...), and Puddle's server
passes those calls on to one sd-server it runs on 127.0.0.1 (ImageServer.handle), so every control of the Web UI
(recipes and models, samplers, schedulers, guidance, reference images, LoRAs, VAE tiling, caches, previews) works as
against a plain sd-server.  The model at the start is the config's recipe and files (FLUX.2 klein 4B); the Web UI can
load another one from the model folder.  Only /sdcpp/v1/* is passed on: sd-server's OpenAI and sdapi routes are not.

sd-server starts on first use (the page asks for its capabilities, or a job comes) and stops after "idle_s" seconds
without a job or a model change (0: never), which gives its RAM back.  While it is stopped Puddle answers the
page's catalog and status requests with what sd-server said last, so the page still shows the model as loaded; the
next job starts sd-server again, loads the model that was loaded before, and runs (it takes that much longer).

The GPU is shared with the text engine one job at a time: a job (img_gen, vid_gen, upscale) takes the service's turn
(the FIFO every chat request takes) from its submission until sd-server reports it completed, failed or cancelled,
and the config's "gpu_policy" says what happens to the text engine meanwhile (gpu_turn):
  keep    - it stays loaded and idle (an APU's shared memory, a card with room for both)
  shrink  - an engine started with --vram-elastic gives the image job VRAM from its expert cache and takes it back
            afterwards (POST /v1/vram's mechanism; the conversation stays in the engine)
  unload  - its conversation is saved to a session file, the engine is stopped, and the next start restores it
  auto    - (default) keep when there is room for the image job next to it, else shrink when the engine can,
            else unload
setup's "args" start sd-server with --offload-to-cpu: the weights stay in RAM, and each part of the model (text
encoder, denoiser, VAE) is copied to the GPU while it runs and released after it, so between jobs sd-server keeps
only its GPU context.

The config's "images" section (written by setup --images flux2-klein, tools/image/install.py):
  {"exe": "engine-image/sd-server", "webui": "engine-image/sd-webui.html", "backend": "cuda", "quant": "Q8_0",
   "model_dir": ".../models/images", "recipe": "flux2-klein-4b",
   "models": {"diffusion_model": "diffusion_models/...gguf", "llm": "text_encoders/...gguf", "vae": "vae/..."},
   "args": ["--offload-to-cpu"], "gpu_policy": "auto", "reload": "lazy"|"eager", "idle_s": 600, "timeout_s": 3600,
   "start_timeout_s": 120, "load_timeout_s": 1800, "port": null, "lib_dirs": [...], "gpu": null, "env": {},
   "uma": false, "need_mib": null, "out_dir": null}
"""
from __future__ import annotations

import contextlib
import http.client
import json
import os
import queue
import secrets
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path

MODEL_NAME = "flux2-klein-4b"
POLICIES = ("auto", "keep", "shrink", "unload")
API = "/sdcpp/v1/"
JOB_PATHS = ("/sdcpp/v1/img_gen", "/sdcpp/v1/vid_gen", "/sdcpp/v1/upscale")
LOAD, UNLOAD, STATUS = "/sdcpp/v1/models/load", "/sdcpp/v1/models/unload", "/sdcpp/v1/models/status"
# what the page reads at its start and polls: kept from sd-server's last answer, and answered from it while stopped
CACHED = ("/sdcpp/v1/capabilities", "/sdcpp/v1/recipes", "/sdcpp/v1/model-files", STATUS)
TERMINAL = ("completed", "failed", "cancelled")
EMPTY_STATUS = {"state": "empty", "recipe": None, "models": {}, "error": None, "revision": 0,
                "load_elapsed_seconds": None}
POLL_S = 0.5                                   # how often a running job's state is read
IDLE_CHECK_S = 5.0
LOG_NAME = "sd-server.log"


class ImageServerError(RuntimeError):
    """sd-server did not start or stopped answering (503); the message carries the end of its log."""


def old_section(cfg: dict) -> bool:
    """An "images" section of the Qwen-Image version (one sd-cli process per image, files by path): it has no model
    folder, and its program is sd-cli."""
    exe = Path(str(cfg.get("exe") or "")).name.lower()
    return "model_dir" not in cfg and (exe.startswith("sd-cli") or "diffusion_model" in cfg)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def server_args(cfg: dict, port: int) -> list[str]:
    """sd-server's command: 127.0.0.1 only (Puddle's server is the way in), the model folder whose catalog the Web
    UI lists, and the config's own arguments (setup's: --offload-to-cpu, on Vulkan also the VAE tiling)."""
    return [str(cfg["exe"]), "--listen-ip", "127.0.0.1", "--listen-port", str(port),
            "--model-dir", str(cfg["model_dir"]), *[str(a) for a in cfg.get("args") or []]]


def image_env(cfg: dict, base: dict | None = None) -> dict:
    """sd-server's environment: the libraries its backend needs (ROCm's, setup's lib_dirs) first on the search path,
    the GPU to use when the config names one, and the config's own "env"."""
    env = dict(os.environ if base is None else base)
    dirs = [str(d) for d in cfg.get("lib_dirs") or [] if Path(d).is_dir()]
    if dirs:
        var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[var] = os.pathsep.join(dirs + ([env[var]] if env.get(var) else []))
    gpu = cfg.get("gpu")
    if gpu is not None:
        var = {"cuda": "CUDA_VISIBLE_DEVICES", "hip": "HIP_VISIBLE_DEVICES",
               "vulkan": "GGML_VK_VISIBLE_DEVICES"}.get(cfg.get("backend"))
        if var:
            env[var] = str(gpu)
            if var == "CUDA_VISIBLE_DEVICES":
                env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    for k, v in (cfg.get("env") or {}).items():
        env[str(k)] = str(v)
    return env


def need_mib(cfg: dict, width: int, height: int, sizes: dict | None = None) -> int:
    """The GPU memory an image job is planned with.  With --offload-to-cpu one part of the model is on the GPU at a
    time, so: the largest of the denoiser and the text encoder, the VAE, and the working buffers (1 GiB plus 3 GiB per
    megapixel: an estimate, not measured with FLUX.2 klein yet).  `sizes`: the bytes of the loaded model's files by
    slot.  The config's "need_mib" overrides it."""
    if cfg.get("need_mib"):
        return int(cfg["need_mib"])
    sizes = sizes or {}
    big = max((v for k, v in sizes.items() if k not in ("vae", "tae")), default=0)
    work = (1 << 30) + int(3 * (1 << 30) * width * height / (1024 * 1024))
    return (big + sizes.get("vae", 0) + work) >> 20


def job_size(body: bytes) -> tuple[int, int]:
    """A job request's width and height (1024 x 1024 when it names none or is not JSON)."""
    try:
        req = json.loads(body or b"{}")
        w, h = int(req.get("width") or 1024), int(req.get("height") or 1024)
        return (w, h) if w > 0 and h > 0 else (1024, 1024)
    except (ValueError, TypeError, AttributeError):
        return 1024, 1024


def json_reply(code: int, obj) -> tuple[int, str, bytes]:
    return code, "application/json", json.dumps(obj).encode()


def _parse(body: bytes):
    try:
        return json.loads(body or b"null")
    except ValueError:
        return None


if os.name != "nt":
    try:                                       # prepared here: the child of a fork must not import anything
        import ctypes
        _PRCTL = ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError):
        _PRCTL = None
else:
    _PRCTL = None


def _die_with_parent():
    """Linux (preexec_fn): sd-server gets SIGTERM when the thread that started it ends (PR_SET_PDEATHSIG) - the
    launcher thread, which lives as long as Puddle's server, however that ends."""
    if _PRCTL is not None:
        _PRCTL(1, 15)


# The page's own settings, seeded once (a key the user has set stays): its "Enhance prompt" button asks Puddle's
# own text model (an OpenAI-compatible /v1 at the same address), with the web app's API key when it has one.
SEED_SCRIPT = ("<script>(function(){try{var s=localStorage,base=location.origin+location.pathname.replace(/\\/sd\\/?$/,"
               "\"\")+\"/v1\",k={\"sdcpp-webui-enhance-base-url\":JSON.stringify(base),"
               "\"sdcpp-webui-enhance-model\":JSON.stringify(%s)},key=s.getItem(\"strata.apikey\");"
               "if(key&&JSON.parse(key))k[\"sdcpp-webui-enhance-api-key\"]=key;"
               "for(var n in k)if(s.getItem(n)===null)s.setItem(n,k[n]);}catch(e){}})();</script>")


def seed_page(html: bytes, model: str) -> bytes:
    """The Web UI's HTML with SEED_SCRIPT first in its head (module scripts run after it)."""
    tag = (SEED_SCRIPT % json.dumps(model or "strata").replace("</", "<\\/")).encode()
    at = html.lower().find(b"<head>")
    if at < 0:
        return tag + html
    at += len(b"<head>")
    return html[:at] + tag + html[at:]


class ImageServer:
    """One sd-server, started on demand, stopped when idle; jobs one at a time in the service's turn (gpu_turn)."""

    def __init__(self, cfg: dict, log=None):
        self.cfg = dict(cfg)
        policy = self.cfg.get("gpu_policy") or "auto"
        if policy not in POLICIES:
            raise ValueError(f'images "gpu_policy" must be one of {", ".join(POLICIES)}, not {policy!r}')
        reload = self.cfg.get("reload") or "lazy"
        if reload not in ("lazy", "eager"):
            raise ValueError(f'images "reload" must be lazy or eager, not {reload!r}')
        for k in ("exe", "model_dir"):
            if not isinstance(self.cfg.get(k), str) or not self.cfg[k]:
                raise ValueError(f'the config\'s "images" section needs "{k}" (run setup with --images flux2-klein '
                                 'again)')
        models = self.cfg.get("models")
        if models is not None and (not isinstance(models, dict) or
                                   not all(isinstance(v, str) for v in models.values())):
            raise ValueError('images "models" must map slot names to files in the model folder')
        self.policy, self.reload = policy, reload
        self.model = self.cfg.get("model_name") or MODEL_NAME
        self.log = log
        out = self.cfg.get("out_dir")
        self.dir = Path(out) if out else Path(tempfile.mkdtemp(prefix="strata-images-"))
        self.dir.mkdir(parents=True, exist_ok=True)
        idle = self.cfg.get("idle_s")
        self.idle_s = 600.0 if idle is None else float(idle)
        self.timeout_s = float(self.cfg.get("timeout_s") or 3600)
        self.start_timeout_s = float(self.cfg.get("start_timeout_s") or 120)
        self.load_timeout_s = float(self.cfg.get("load_timeout_s") or 1800)
        self.proc = None
        self.port = None
        self.ready = False                           # the running sd-server has answered
        self.logf = None
        self.lock = threading.RLock()                # the process and the state below
        self.start_lock = threading.RLock()          # one start (or stop) at a time
        self.want = self.default_load()              # what sd-server loads when it starts: the last model loaded
        self.loaded = threading.Event()              # the start's model load has ended (either way)
        self.loaded.set()
        self.cache = {}                              # path -> sd-server's last answer (CACHED)
        self.active = None                           # the running job: id, status, progress, policy
        self.inflight = 0                            # POSTs being passed on
        self.last_used = time.monotonic()
        self.totals = {"jobs": 0, "images": 0, "failed": 0, "cancelled": 0, "seconds": 0.0, "starts": 0}
        self.last = None                             # the last job's figures (/v1/status)
        self.stopped_because = None
        self.sessions = []                           # the page's cookies when the server has an API key
        self._launch_q = queue.Queue()
        self._launcher_thread = None
        self._idle_thread = None
        self._closed = False

    # ------------------------------------------------------------------------------------------- setup and state
    def default_load(self) -> dict | None:
        """The config's model as sd-server's models/load body (recipe + slot -> file ID), or None."""
        recipe, models = self.cfg.get("recipe"), self.cfg.get("models")
        return {"recipe": recipe, "models": dict(models)} if recipe and models else None

    def missing(self) -> list[str]:
        root = Path(self.cfg["model_dir"])
        gone = [p for p in (self.cfg["exe"], self.cfg["model_dir"]) if not Path(p).exists()]
        gone += [str(root / rel) for rel in (self.cfg.get("models") or {}).values() if not (root / rel).is_file()]
        return gone

    def _say(self, text: str) -> None:
        print(f"[strata] {text}", flush=True)
        if self.log is not None:
            try:
                self.log.write(f"[images] {text}\n")
                self.log.flush()
            except (OSError, ValueError):
                pass

    def running(self) -> bool:
        """The sd-server process is alive (it may still be starting)."""
        with self.lock:
            return self.proc is not None and self.proc.poll() is None

    def serving(self) -> bool:
        """sd-server is alive and has answered since its start."""
        with self.lock:
            return self.ready and self.running()

    def log_tail(self, lines: int = 12) -> str:
        try:
            text = (self.dir / LOG_NAME).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.strip().splitlines()[-lines:])

    def sizes(self) -> dict:
        """The bytes of the model files sd-server loads (want), by slot."""
        root = Path(self.cfg["model_dir"])
        out = {}
        for slot, rel in ((self.want or {}).get("models") or {}).items():
            try:
                out[slot] = (root / rel).stat().st_size
            except (OSError, TypeError):
                pass
        return out

    def need_mib(self, width: int, height: int) -> int:
        return need_mib(self.cfg, width, height, self.sizes())

    def new_session(self) -> str:
        """A token for the page's cookie (the server has an API key; the page's iframe cannot send it as a header)."""
        token = secrets.token_urlsafe(24)
        with self.lock:
            self.sessions = (self.sessions + [token])[-64:]
        return token

    def session_ok(self, token: str | None) -> bool:
        if not token:
            return False
        with self.lock:
            return any(secrets.compare_digest(token, t) for t in self.sessions)

    # ------------------------------------------------------------------------------------------- the process
    def _launcher(self):
        while True:
            item = self._launch_q.get()
            if item is None:
                return
            cmd, kw, done, ev = item
            try:
                done["proc"] = subprocess.Popen(cmd, preexec_fn=_die_with_parent, **kw)
            except Exception as e:                   # noqa: BLE001 - handed to the caller
                done["error"] = e
            ev.set()

    def _popen(self, cmd, **kw) -> subprocess.Popen:
        """Popen on the launcher thread on Linux: PR_SET_PDEATHSIG fires when the *thread* that started a child ends,
        and a request's thread ends with its request."""
        if os.name == "nt" or _PRCTL is None:
            return subprocess.Popen(cmd, **kw)
        if self._launcher_thread is None:
            self._launcher_thread = threading.Thread(target=self._launcher, daemon=True, name="strata-image-launcher")
            self._launcher_thread.start()
        done, ev = {}, threading.Event()
        self._launch_q.put((cmd, kw, done, ev))
        ev.wait()
        if "error" in done:
            raise done["error"]
        return done["proc"]

    def ensure_running(self, replay: bool = True) -> None:
        """Start sd-server if it is not running and wait until it answers; `replay`: then load the model it had
        (in the background: self.loaded is set when that has ended).  ImageServerError when it does not start."""
        with self.start_lock:
            if self.serving():
                return
            if self._closed:
                raise ImageServerError("the image server is shut down")
            with self.lock:
                if self.proc is not None:
                    if self.proc.poll() is None:     # alive, not answering: a start that went wrong
                        self.proc.kill()
                        with contextlib.suppress(subprocess.TimeoutExpired):
                            self.proc.wait(timeout=15)
                    self.ready = False
                    self._say(f"image server: sd-server had ended (exit code {self.proc.poll()}); starting it again\n"
                              f"{self.log_tail()}")
                    self.proc = self.port = None
            port = int(self.cfg.get("port") or 0) or free_port()
            cmd = server_args(self.cfg, port)
            self._close_log()
            self.logf = open(self.dir / LOG_NAME, "w", encoding="utf-8", errors="replace")
            try:
                proc = self._popen(cmd, stdout=self.logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   env=image_env(self.cfg), cwd=str(self.dir))
            except OSError as e:
                self._close_log()
                raise ImageServerError(f"sd-server did not start ({cmd[0]}): {e}") from e
            try:
                from serve.winjob import contain
                contain(proc)                         # ends with the server (Windows)
            except Exception:                         # noqa: BLE001
                pass
            with self.lock:
                self.proc, self.port = proc, port
                self.last_used = time.monotonic()
                self.totals["starts"] += 1
                self.stopped_because = None
            self._say(f"image server: sd-server started on 127.0.0.1:{port} (pid {proc.pid})")
            self._wait_listening(proc)
            with self.lock:
                self.ready = True
            self.start_idle()
            want = self.want if replay else None
            self.loaded.clear()
            if want:
                threading.Thread(target=self._load_at_start, args=(want,), daemon=True,
                                 name="strata-image-load").start()
            else:
                self.loaded.set()

    def _wait_listening(self, proc) -> None:
        deadline = time.monotonic() + self.start_timeout_s
        while True:
            if proc.poll() is not None:
                tail = self.log_tail()
                with self.lock:
                    self.proc = self.port = None
                    self.ready = False
                self._close_log()
                raise ImageServerError(f"sd-server ended at its start (exit code {proc.returncode}); the end of its "
                                       f"log:\n{tail}")
            with contextlib.suppress(OSError):
                if self._call("GET", STATUS, timeout=2)[0] == 200:
                    return
            if time.monotonic() > deadline:
                tail = self.log_tail()
                self.stop("it did not answer")
                raise ImageServerError(f"sd-server did not answer within start_timeout_s ({self.start_timeout_s:g} "
                                       f"s); the end of its log:\n{tail}")
            time.sleep(0.2)

    def _load_at_start(self, want: dict) -> None:
        try:
            code, _, body = self._call("POST", LOAD, json.dumps(want).encode(), "application/json",
                                       timeout=self.load_timeout_s)
            if code == 200:
                self._say(f"image server: {want.get('recipe')} loaded")
            else:
                err = _parse(body)
                self._say(f"image server: loading {want.get('recipe')} failed ({code}: "
                          f"{(err or {}).get('error') if isinstance(err, dict) else body[:300]!r})")
        except OSError as e:
            self._say(f"image server: loading {want.get('recipe')} failed ({e})")
        finally:
            self.loaded.set()

    def stop(self, why: str) -> None:
        with self.start_lock:
            with self.lock:
                proc, self.proc, self.port = self.proc, None, None
                self.ready = False
                if proc is not None:
                    self.stopped_because = why
            if proc is None:
                self._close_log()                    # it may have ended by itself
                return
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        proc.wait(timeout=15)
            self._close_log()
            self.loaded.set()
            self._say(f"image server: sd-server stopped ({why})")

    def _close_log(self) -> None:
        if self.logf is not None:
            with contextlib.suppress(OSError):
                self.logf.close()
            self.logf = None

    def close(self) -> None:
        self._closed = True
        self.stop("Puddle's server stopped")
        if self._launcher_thread is not None:
            self._launch_q.put(None)

    def start_idle(self) -> None:
        if self._idle_thread is None and self.idle_s > 0:
            self._idle_thread = threading.Thread(target=self._idle_loop, daemon=True, name="strata-image-idle")
            self._idle_thread.start()

    def _idle_loop(self):
        while not self._closed:
            time.sleep(IDLE_CHECK_S)
            self.idle_check()

    def idle_check(self, now: float | None = None) -> bool:
        """Stop sd-server after idle_s without a job, a model change or another POST (the page's polls do not count:
        an open page alone does not keep it running).  -> whether it was stopped."""
        if self.idle_s <= 0:
            return False
        now = time.monotonic() if now is None else now
        with self.lock:
            if not self.serving() or self.active is not None or self.inflight or now - self.last_used < self.idle_s:
                return False
        self.stop(f"no image job for {self.idle_s:g} s; the next one starts it again")
        return True

    # ------------------------------------------------------------------------------------------- talking to it
    def _call(self, method: str, path: str, body: bytes | None = None, ctype: str | None = None,
              timeout: float = 60) -> tuple[int, str, bytes]:
        """One request to sd-server -> (status, content type, body).  OSError when it does not answer."""
        port = self.port
        if port is None:
            raise ConnectionRefusedError("sd-server is not running")
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        try:
            headers = {"Content-Type": ctype or "application/json"} if body is not None else {}
            conn.request(method, path, body=body, headers=headers)
            r = conn.getresponse()
            return r.status, r.getheader("Content-Type") or "application/json", r.read()
        except http.client.HTTPException as e:
            raise ConnectionError(f"{type(e).__name__}: {e}") from e
        finally:
            conn.close()

    def handle(self, svc, method: str, path: str, query: str = "", body: bytes | None = None,
               ctype: str | None = None, cancel: threading.Event | None = None) -> tuple[int, str, bytes]:
        """One request of the page, `path` without the /sd prefix -> (status, content type, body)."""
        if not path.startswith(API):
            return json_reply(404, {"error": "not found", "code": "not_found"})
        target = path + (f"?{query}" if query else "")
        if method == "GET":
            return self._get(path, target)
        with self.lock:
            self.inflight += 1
            self.last_used = time.monotonic()
        try:
            if path in JOB_PATHS:
                return self._job(svc, path, target, body, ctype, cancel)
            if path == UNLOAD and not self.serving():
                self.want = None
                reply = json_reply(200, EMPTY_STATUS)
                self.cache[STATUS] = reply
                return reply
            if path.startswith("/sdcpp/v1/jobs/") and not self.serving():
                return json_reply(410, {"error": "the image server was stopped: the job is gone", "code": "job_gone"})
            change = path in (LOAD, UNLOAD)
            self.ensure_running(replay=not change)
            if change:
                self.loaded.wait(self.load_timeout_s)
            reply = self._call("POST", target, body if body is not None else b"", ctype,
                               timeout=self.load_timeout_s if change else 120)
            if reply[0] == 200 and change:
                req = _parse(body) if path == LOAD else None
                self.want = req if isinstance(req, dict) and req.get("recipe") else None
            return reply
        except ImageServerError as e:
            return json_reply(503, {"error": str(e), "code": "image_server_failed"})
        except OSError as e:
            return json_reply(503, {"error": f"sd-server did not answer ({e})", "code": "image_server_down"})
        finally:
            with self.lock:
                self.inflight -= 1
                self.last_used = time.monotonic()

    def _get(self, path: str, target: str) -> tuple[int, str, bytes]:
        if not self.serving():
            if path in self.cache:
                return self.cache[path]
            if path.startswith("/sdcpp/v1/jobs/"):
                return json_reply(410, {"error": "the image server was stopped: the job is gone", "code": "job_gone"})
            try:
                self.ensure_running()
            except ImageServerError as e:
                return json_reply(503, {"error": str(e), "code": "image_server_failed"})
        try:
            reply = self._call("GET", target)
        except OSError as e:
            if path in self.cache:
                return self.cache[path]
            return json_reply(503, {"error": f"sd-server did not answer ({e})", "code": "image_server_down"})
        if reply[0] == 200 and path in CACHED:
            self.cache[path] = reply
        return reply

    def _job(self, svc, path, target, body, ctype, cancel) -> tuple[int, str, bytes]:
        """A job's submission: in a thread that takes the GPU turn, starts sd-server if it is stopped, submits the job
        and holds the turn until the job has ended.  The page gets sd-server's answer to the submission."""
        width, height = job_size(body)
        out, submitted = {}, threading.Event()

        def work():
            try:
                with gpu_turn(svc, self, {"width": width, "height": height}, cancel) as policy:
                    self.ensure_running()
                    if not self.loaded.wait(self.load_timeout_s):
                        raise ImageServerError(f"the model did not load within load_timeout_s "
                                               f"({self.load_timeout_s:g} s)")
                    reply = self._call("POST", target, body or b"", ctype, timeout=120)
                    out["reply"] = reply
                    job = _parse(reply[2]) if reply[0] in (200, 202) else None
                    job_id = job.get("id") if isinstance(job, dict) else None
                    if not job_id:
                        return
                    with self.lock:
                        self.active = {"id": job_id, "kind": job.get("kind") or path.rsplit("/", 1)[-1],
                                       "status": job.get("status"), "progress": None, "policy": policy,
                                       "since": time.time()}
                    submitted.set()
                    self._say(f"image job {job_id}: {width}x{height} (GPU policy {policy})")
                    self._follow(job_id)
            except ImageServerError as e:
                out["error"] = str(e)
            except OSError as e:
                out["error"] = f"sd-server did not answer ({e})"
            except Exception as e:                   # noqa: BLE001 - ClientGone and the like: the caller's
                out["raise"] = e
            finally:
                with self.lock:
                    self.active = None
                    self.last_used = time.monotonic()
                submitted.set()

        threading.Thread(target=work, daemon=True, name="strata-image-job").start()
        submitted.wait()
        if "reply" in out:
            return out["reply"]
        if "raise" in out:
            raise out["raise"]
        self._say(f"image job not submitted: {out.get('error')}")
        return json_reply(503, {"error": out.get("error") or "the image job did not start",
                                "code": "image_server_failed"})

    def _follow(self, job_id: str) -> None:
        """Read the job's state until it has ended (or sd-server has), cancelling it past timeout_s."""
        t0, deadline, misses = time.monotonic(), time.monotonic() + self.timeout_s, 0
        while True:
            time.sleep(POLL_S)
            if deadline is not None and time.monotonic() > deadline:
                self._say(f"image job {job_id} ran past timeout_s ({self.timeout_s:g} s): cancelling it")
                with contextlib.suppress(OSError):
                    self._call("POST", f"/sdcpp/v1/jobs/{job_id}/cancel", b"", None, timeout=30)
                deadline = None
                t0 = time.monotonic()                   # at most a minute more for it to end
            elif deadline is None and time.monotonic() - t0 > 60:
                self.totals["failed"] += 1
                return
            try:
                code, _, raw = self._call("GET", f"/sdcpp/v1/jobs/{job_id}", timeout=30)
            except OSError:
                if not self.running():
                    self.totals["failed"] += 1
                    self._say(f"image job {job_id}: sd-server ended during the job\n{self.log_tail()}")
                    return
                misses += 1
                if misses > 20:
                    self.totals["failed"] += 1
                    return
                continue
            misses = 0
            if code in (404, 410):
                return
            job = _parse(raw) if code == 200 else None
            if not isinstance(job, dict):
                continue
            with self.lock:
                if self.active is not None:
                    self.active.update(status=job.get("status"), progress=job.get("progress"))
            if job.get("status") in TERMINAL:
                self._finish(job)
                return

    def _finish(self, job: dict) -> None:
        status = job.get("status")
        started, done = job.get("started"), job.get("completed")
        secs = round((done - started) / 1000, 2) if isinstance(started, int) and isinstance(done, int) else None
        result = job.get("result") if isinstance(job.get("result"), dict) else {}
        n = len(result.get("images") or []) if status == "completed" else 0
        with self.lock:
            if status == "completed":
                self.totals["jobs"] += 1
                self.totals["images"] += n
            else:
                self.totals["failed" if status == "failed" else "cancelled"] += 1
            if secs is not None:
                self.totals["seconds"] += secs
            self.last = {"id": job.get("id"), "kind": job.get("kind"), "status": status, "seconds": secs,
                         "images": n, "policy": (self.active or {}).get("policy"),
                         "backend": self.cfg.get("backend"), "quant": self.cfg.get("quant")}
        err = job.get("error") if isinstance(job.get("error"), dict) else {}
        self._say(f"image job {job.get('id')} {status}" + (f" in {secs:.1f} s" if secs is not None else "") +
                  (f": {err.get('message')}" if status == "failed" and err.get("message") else ""))

    # ------------------------------------------------------------------------------------------- the page
    def page(self, model: str) -> bytes | None:
        """The Web UI's HTML (setup's copy beside sd-server), seeded for this server; None when it is missing."""
        path = self.cfg.get("webui")
        try:
            html = Path(path).read_bytes() if path else None
        except OSError:
            html = None
        return seed_page(html, model) if html else None

    def status(self) -> dict:
        with self.lock:
            out = {"enabled": True, "model": self.model, "backend": self.cfg.get("backend"),
                   "quant": self.cfg.get("quant"), "gpu_policy": self.policy,
                   "server": "running" if self.serving() else "starting" if self.running() else "stopped",
                   "idle_s": self.idle_s, **self.totals}
            out["seconds"] = round(out["seconds"], 2)
            if self.stopped_because and out["server"] == "stopped":
                out["stopped_because"] = self.stopped_because
            st = _parse(self.cache[STATUS][2]) if STATUS in self.cache else None
            if isinstance(st, dict):
                out["state"], out["recipe"] = st.get("state"), st.get("recipe")
            elif self.want:
                out["recipe"] = self.want.get("recipe")
            if self.active is not None:
                p = self.active.get("progress") or {}
                out["job"] = {"id": self.active["id"], "status": self.active.get("status"), "stage": p.get("stage"),
                              "completed": p.get("completed"), "total": p.get("total")}
                if p.get("unit") == "steps" and p.get("total"):
                    out["step"], out["steps"] = p.get("completed") or 0, p["total"]
            if self.last:
                out["last"] = self.last
            return out


# ------------------------------------------------------------------------------------------- the GPU turn
def resolve_policy(svc, gen: ImageServer, need: int) -> str:
    """auto -> keep / shrink / unload for this job (see the module's docstring); a fixed policy is kept, except that
    an engine that is not running has nothing to give back (keep)."""
    eng = svc.engine
    running = not hasattr(eng, "alive") or eng.alive()
    if not running or not hasattr(eng, "unload"):
        return "keep"
    if gen.policy != "auto":
        return gen.policy
    if gen.cfg.get("uma"):
        return "keep"                         # an APU: the image job and the engine share the RAM either way
    free = svc.free_vram_mib()
    if free is None or free >= need:
        return "keep"
    if engine_elastic(eng):
        return "shrink"
    return "unload"


def engine_elastic(eng) -> bool:
    """An engine that resizes its expert cache on request (--vram-elastic, #533)."""
    if not hasattr(eng, "vram"):
        return False
    if (getattr(eng, "info", None) or {}).get("vram_elastic"):
        return True
    spawn = getattr(eng, "spawn", None)
    args = spawn[1] if isinstance(spawn, (list, tuple)) and len(spawn) > 1 else []
    return "--vram-elastic" in (args or [])


def _client_gone(svc) -> Exception:
    """The server module's ClientGone (it may run as __main__, so not imported by name)."""
    import sys
    return getattr(sys.modules.get(type(svc).__module__), "ClientGone", RuntimeError)()


@contextlib.contextmanager
def gpu_turn(svc, gen: ImageServer, job: dict, cancel: threading.Event | None = None):
    """The image job's turn on the GPU: the service's FIFO (and, with parallel requests, the engine's control lines
    once no slot is generating), and the text engine made room for it per the policy.  Yields the policy used."""
    with svc.status_lock:
        svc.status["queued"] += 1
    queued = True
    try:
        with svc._turn(cancel):
            with svc.status_lock:
                svc.status["queued"] -= 1
            queued = False
            ctl = getattr(svc.engine, "ctl", None) if getattr(svc.engine, "batch", 0) else None
            if ctl is not None:
                while not ctl.acquire(timeout=0.25):
                    if cancel is not None and cancel.is_set():
                        raise _client_gone(svc)
            try:
                if ctl is not None:                   # running parallel requests finish first
                    while getattr(svc, "live_reqs", None):
                        if cancel is not None and cancel.is_set():
                            raise _client_gone(svc)
                        time.sleep(0.25)
                need = gen.need_mib(job["width"], job["height"])
                policy = resolve_policy(svc, gen, need)
                with svc.status_lock:
                    svc.status.update(busy=True, phase="generating an image", started=time.time(),
                                      first_token=None, prompt_tokens=None, generated=None, max_tokens=None)
                restore = _make_room(svc, gen, policy, need)
                try:
                    yield policy
                finally:
                    _give_back(svc, gen, policy, restore)
                    with svc.status_lock:
                        svc.status["busy"] = False
                    svc.last_request_at = time.time()
            finally:
                if ctl is not None:
                    ctl.release()
    finally:
        if queued:
            with svc.status_lock:
                svc.status["queued"] -= 1


def _make_room(svc, gen: ImageServer, policy: str, need: int):
    """Do what `policy` says before the job; -> what _give_back needs."""
    eng = svc.engine
    if policy == "shrink":
        try:
            out = eng.vram(need)
            gen._say(f"image job: the engine's expert cache gave VRAM back ({out.get('vram_free_mib')} MiB free)")
            return "shrunk"
        except Exception as e:                       # noqa: BLE001 - the job still runs, with what is free
            gen._say(f"image job: the engine could not give VRAM back ({e}); the job runs with what is free")
            return None
    if policy == "unload":                           # strata-poly keeps no disk sessions: the next request reads its
        gen._say("image job: unloading the model")   # prompt again
        eng.unload()
        if svc.vision is not None and hasattr(svc.vision, "unload"):
            svc.vision.unload()
        return "unloaded"
    return None


def _give_back(svc, gen: ImageServer, policy: str, restore) -> None:
    eng = svc.engine
    if restore == "shrunk":
        try:
            eng.vram(svc.vram_reserve)               # the reserve asked for last (None: the start's)
        except Exception as e:                       # noqa: BLE001
            gen._say(f"image job: the expert cache did not grow back ({e}); POST /v1/vram can retry")
    elif policy == "unload" and restore:
        if gen.reload == "eager":
            try:
                svc.ensure_loaded()                  # the caller still holds the FIFO
            except Exception as e:                   # noqa: BLE001 - the next request tries again
                gen._say(f"image job: the model did not load again yet ({e}); the next request loads it")


