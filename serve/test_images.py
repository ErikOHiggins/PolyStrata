"""Image generation (serve/images.py): stable-diffusion.cpp's Web UI at /sd/ and its API passed on to an sd-server
started on demand - the command, the GPU turn and its policies, the idle stop and the answers while stopped, the
page's access rules - over HTTP with a stand-in sd-server (serve/fixtures/fake_sd_server.py) and the mock engine.
No GPU, no model files.

    python -m serve.test_images
"""
import json
import os
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve import images as images_api
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, api_key_of, serve

ROOT = Path(__file__).resolve().parents[1]
FAKE = ROOT / "serve" / "fixtures" / "fake_sd_server.py"
REAL_ARGS = images_api.server_args
JOB = "/sdcpp/v1/img_gen"


def fake_args(cfg, port):
    """sd-server's command, run through Python (the stand-in is a script)."""
    return [sys.executable, str(FAKE), *REAL_ARGS(cfg, port)[1:]]


def image_cfg(d, **kw):
    root = Path(d) / "images"
    models = {"diffusion_model": "diffusion_models/dit.gguf", "llm": "text_encoders/te.gguf",
              "vae": "vae/vae.safetensors"}
    for rel in models.values():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"x" * 1024)
    page = Path(d) / "sd-webui.html"
    page.write_text("<!doctype html><html><head><title>sd</title><script type=\"module\">/*webui*/</script></head>"
                    "<body></body></html>", encoding="utf-8")
    return {"exe": str(FAKE), "webui": str(page), "backend": "vulkan", "quant": "Q8_0", "model_dir": str(root),
            "recipe": "flux2-klein-4b", "models": models, "args": ["--offload-to-cpu"], "idle_s": 0,
            "out_dir": str(Path(d) / "out"), **kw}


def wait_for(cond, timeout=20.0, what="the condition"):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def body_of(reply):
    return json.loads(reply[2])


class Pieces(unittest.TestCase):
    def test_server_args(self):
        cfg = {"exe": "sd-server", "model_dir": "/m/images", "args": ["--offload-to-cpu", "--vae-tiling"]}
        self.assertEqual(images_api.server_args(cfg, 4321), [
            "sd-server", "--listen-ip", "127.0.0.1", "--listen-port", "4321", "--model-dir", "/m/images",
            "--offload-to-cpu", "--vae-tiling"])

    def test_env(self):
        env = images_api.image_env({"backend": "cuda", "gpu": 1, "env": {"X": 2}}, {"PATH": "p"})
        self.assertEqual((env["CUDA_VISIBLE_DEVICES"], env["CUDA_DEVICE_ORDER"], env["X"]), ("1", "PCI_BUS_ID", "2"))
        with tempfile.TemporaryDirectory() as d:
            env = images_api.image_env({"backend": "hip", "gpu": 0, "lib_dirs": [d, d + "-missing"]}, {})
            self.assertEqual(env["HIP_VISIBLE_DEVICES"], "0")
            self.assertEqual(env["PATH" if os.name == "nt" else "LD_LIBRARY_PATH"], d)
        self.assertEqual(images_api.image_env({"backend": "vulkan", "gpu": 2}, {})["GGML_VK_VISIBLE_DEVICES"], "2")

    def test_need_mib(self):
        gib = 1 << 30
        sizes = {"diffusion_model": 4 * gib, "llm": 5 * gib, "vae": gib // 2}
        self.assertEqual(images_api.need_mib({}, 1024, 1024, sizes), (5 * gib + gib // 2 + 4 * gib) >> 20)
        self.assertEqual(images_api.need_mib({}, 512, 512, {}), (gib + 3 * gib // 4) >> 20)
        self.assertEqual(images_api.need_mib({"need_mib": 777}, 1024, 1024, sizes), 777)

    def test_job_size(self):
        self.assertEqual(images_api.job_size(b'{"width": 768, "height": 512}'), (768, 512))
        for bad in (b"", b"[]", b"not json", b'{"width": -5}', b'{"width": "x"}'):
            self.assertEqual(images_api.job_size(bad), (1024, 1024), bad)

    def test_old_section(self):
        self.assertTrue(images_api.old_section({"exe": "engine-image/sd-cli", "diffusion_model": "q.gguf"}))
        self.assertTrue(images_api.old_section({"exe": "C:/x/sd-cli.exe"}))
        self.assertFalse(images_api.old_section({"exe": "engine-image/sd-server", "model_dir": "m"}))

    def test_seed_page(self):
        html = b"<html><HEAD><script type=module>app()</script></head></html>"
        out = images_api.seed_page(html, "Qwen3.8 </script>")
        head = out.index(b"<HEAD>") + len(b"<HEAD>")
        self.assertTrue(out[head:].startswith(b"<script>"))
        self.assertLess(out.index(b"sdcpp-webui-enhance-base-url"), out.index(b"app()"))
        self.assertIn(b'"Qwen3.8 <\\/script>"', out)              # the name cannot end the script early
        self.assertTrue(images_api.seed_page(b"<p>no head</p>", "m").startswith(b"<script>"))

    def test_config_checked(self):
        with tempfile.TemporaryDirectory() as d:
            for bad in ({"exe": "x"}, {"model_dir": "m"}, image_cfg(d, gpu_policy="sometimes"),
                        image_cfg(d, reload="never"), image_cfg(d, models=["a"])):
                with self.subTest(cfg=bad), self.assertRaises(ValueError):
                    images_api.ImageServer(bad)
            gen = images_api.ImageServer(image_cfg(d))
            self.assertEqual(gen.missing(), [])
            Path(gen.cfg["model_dir"], "vae", "vae.safetensors").unlink()
            self.assertEqual(len(gen.missing()), 1)
            self.assertEqual(gen.default_load()["recipe"], "flux2-klein-4b")

    def test_sessions(self):
        with tempfile.TemporaryDirectory() as d:
            gen = images_api.ImageServer(image_cfg(d))
            t = gen.new_session()
            self.assertTrue(gen.session_ok(t))
            self.assertFalse(gen.session_ok(t + "x"))
            self.assertFalse(gen.session_ok(""))


class FakeEngine:
    """An engine with the parts the GPU turn uses: alive/unload/restart, vram (elastic) and session files."""

    def __init__(self, elastic=False):
        self.up, self.calls, self.info = True, [], {"vram_elastic": elastic}
        self.spawn = (["strata"], ["--vram-elastic"] if elastic else [])
        self.unloaded = False

    def alive(self):
        return self.up

    def exit_code(self):
        return 0

    def unload(self):
        self.calls.append("unload")
        self.up, self.unloaded = False, True

    def restart(self):
        self.calls.append("restart")
        self.up, self.unloaded = True, False

    def vram(self, reserve):
        self.calls.append(("vram", reserve))
        return {"vram_free_mib": 9000}

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        return iter(())


def make_svc(engine):
    return Service(engine, ByteTokenizer(), ChatTemplate(ROOT / "serve/chat_template.jinja"))


class Policy(unittest.TestCase):
    def test_resolve(self):
        with tempfile.TemporaryDirectory() as d:
            gen = images_api.ImageServer(image_cfg(d))
            svc = make_svc(FakeEngine())
            with mock.patch.object(svc, "free_vram_mib", return_value=20000):
                self.assertEqual(images_api.resolve_policy(svc, gen, 15000), "keep")
            with mock.patch.object(svc, "free_vram_mib", return_value=None):
                self.assertEqual(images_api.resolve_policy(svc, gen, 15000), "keep")
            with mock.patch.object(svc, "free_vram_mib", return_value=4000):
                self.assertEqual(images_api.resolve_policy(svc, gen, 15000), "unload")
                svc.engine = FakeEngine(elastic=True)
                self.assertEqual(images_api.resolve_policy(svc, gen, 15000), "shrink")
                gen.cfg["uma"] = True
                self.assertEqual(images_api.resolve_policy(svc, gen, 15000), "keep")
                gen.cfg["uma"] = False
                gen.policy = "unload"
                self.assertEqual(images_api.resolve_policy(svc, gen, 15000), "unload")
                svc.engine.up = False                   # nothing loaded: nothing to give back
                self.assertEqual(images_api.resolve_policy(svc, gen, 15000), "keep")
            self.assertEqual(images_api.resolve_policy(make_svc(MockEngine(ByteTokenizer(), "x")), gen, 1), "keep")

    def test_unload_and_the_next_start(self):
        # strata-poly keeps no disk sessions: the model unloads, and the next request reads its prompt again
        with tempfile.TemporaryDirectory() as d:
            gen = images_api.ImageServer(image_cfg(d, gpu_policy="unload"))
            eng = FakeEngine()
            svc = make_svc(eng)
            with images_api.gpu_turn(svc, gen, {"width": 1024, "height": 1024}) as policy:
                self.assertEqual(policy, "unload")
                self.assertTrue(svc.fifo.locked())          # no chat turn meanwhile
                self.assertFalse(eng.up)
                self.assertTrue(svc.status["busy"])
            self.assertFalse(svc.fifo.locked())
            self.assertFalse(svc.status["busy"])
            self.assertEqual([c if isinstance(c, str) else c[0] for c in eng.calls], ["unload"])
            with svc.fifo:
                svc.ensure_loaded()                          # the next request starts the engine again
            self.assertEqual([c if isinstance(c, str) else c[0] for c in eng.calls], ["unload", "restart"])
            self.assertEqual(list(Path(gen.dir).glob("llm-session-*")), [])

    def test_eager_reload(self):
        with tempfile.TemporaryDirectory() as d:
            gen = images_api.ImageServer(image_cfg(d, gpu_policy="unload", reload="eager"))
            eng = FakeEngine()
            svc = make_svc(eng)
            with images_api.gpu_turn(svc, gen, {"width": 512, "height": 512}):
                pass
            self.assertEqual([c if isinstance(c, str) else c[0] for c in eng.calls], ["unload", "restart"])
            self.assertTrue(eng.up)

    def test_shrink_gives_the_cache_back(self):
        with tempfile.TemporaryDirectory() as d:
            gen = images_api.ImageServer(image_cfg(d, gpu_policy="shrink"))
            eng = FakeEngine(elastic=True)
            svc = make_svc(eng)
            svc.vram_reserve = 512
            with images_api.gpu_turn(svc, gen, {"width": 1024, "height": 1024}):
                self.assertEqual(eng.calls[0][0], "vram")
                self.assertEqual(eng.calls[0][1], 4 * 1024)     # 1 GiB + 3 GiB per megapixel (+ 2 KiB of files)
            self.assertEqual(eng.calls[-1], ("vram", 512))
            self.assertTrue(eng.up)

    def test_keep_touches_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            gen = images_api.ImageServer(image_cfg(d, gpu_policy="keep"))
            eng = FakeEngine()
            with images_api.gpu_turn(make_svc(eng), gen, {"width": 1024, "height": 1024}):
                pass
            self.assertEqual(eng.calls, [])


@mock.patch.object(images_api, "server_args", fake_args)
class Lifecycle(unittest.TestCase):
    """ImageServer.handle against the stand-in: on-demand start, the GPU turn per job, the idle stop."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.svc = make_svc(MockEngine(ByteTokenizer(), "</think>\n\nok"))
        self.args_file = Path(self.tmp.name) / "args.json"
        self.env = mock.patch.dict(os.environ, {"FAKE_SD_ARGS": str(self.args_file)})
        self.env.start()

    def tearDown(self):
        gen = getattr(self, "gen", None)
        if gen is not None:
            gen.close()
        self.env.stop()
        self.tmp.cleanup()

    def make(self, **kw):
        self.gen = images_api.ImageServer(image_cfg(self.tmp.name, **kw))
        return self.gen

    def ready(self, gen):
        wait_for(lambda: body_of(gen.handle(self.svc, "GET", images_api.STATUS))["state"] == "ready",
                 what="the model to load")

    def job(self, gen, **body):
        reply = gen.handle(self.svc, "POST", JOB, body=json.dumps({"prompt": "a fox", **body}).encode(),
                           ctype="application/json")
        self.assertEqual(reply[0], 202, reply)
        return body_of(reply)["id"]

    def test_first_use_starts_it_and_loads_the_model(self):
        gen = self.make()
        self.assertFalse(gen.running())
        code, ctype, data = gen.handle(self.svc, "GET", "/sdcpp/v1/capabilities", "a=1")
        self.assertEqual((code, json.loads(data)["query"]), (200, "a=1"))
        self.ready(gen)
        st = body_of(gen.handle(self.svc, "GET", images_api.STATUS))
        self.assertEqual((st["recipe"], st["models"]), ("flux2-klein-4b", gen.cfg["models"]))
        args = json.loads(self.args_file.read_text())
        self.assertEqual(args[:4], ["--listen-ip", "127.0.0.1", "--listen-port", str(gen.port)])
        self.assertEqual(args[4:], ["--model-dir", gen.cfg["model_dir"], "--offload-to-cpu"])
        self.assertEqual(gen.handle(self.svc, "GET", "/elsewhere")[0], 404)       # only the native API

    def test_a_job_holds_the_turn_until_it_ends(self):
        gen = self.make()
        with mock.patch.dict(os.environ, {"FAKE_SD_JOB_S": "1.0"}):
            gen.ensure_running()
            self.ready(gen)
            job_id = self.job(gen, width=512, height=512, batch_count=2)
            self.assertTrue(self.svc.fifo.locked())                  # no chat turn while it runs
            self.assertEqual(gen.status()["job"]["id"], job_id)
            wait_for(lambda: gen.active is None, what="the job to end")
        self.assertFalse(self.svc.fifo.locked())
        self.assertEqual((gen.totals["jobs"], gen.totals["images"]), (1, 2))
        self.assertEqual(gen.last["status"], "completed")
        done = body_of(gen.handle(self.svc, "GET", f"/sdcpp/v1/jobs/{job_id}"))
        self.assertEqual(len(done["result"]["images"]), 2)

    def test_a_failed_job_is_counted(self):
        gen = self.make()
        with mock.patch.dict(os.environ, {"FAKE_SD_FAIL": "1"}):
            gen.ensure_running()
            self.ready(gen)
            self.job(gen)
            wait_for(lambda: gen.active is None, what="the job to end")
        self.assertEqual((gen.totals["jobs"], gen.totals["failed"]), (0, 1))
        self.assertFalse(self.svc.fifo.locked())

    def test_idle_stop_answers_from_what_it_said_last_and_the_next_job_starts_it(self):
        gen = self.make(idle_s=30)
        for path in images_api.CACHED:
            self.assertEqual(gen.handle(self.svc, "GET", path)[0], 200)
        self.ready(gen)
        self.assertFalse(gen.idle_check())                           # not idle yet
        self.assertTrue(gen.idle_check(now=time.monotonic() + 31))
        self.assertFalse(gen.running())
        self.assertEqual(gen.status()["server"], "stopped")
        st = body_of(gen.handle(self.svc, "GET", images_api.STATUS))  # the page still sees its model loaded
        self.assertEqual(st["state"], "ready")
        self.assertEqual(gen.handle(self.svc, "GET", "/sdcpp/v1/recipes")[0], 200)
        self.assertEqual(gen.handle(self.svc, "GET", "/sdcpp/v1/jobs/job_x")[0], 410)
        self.assertFalse(gen.running())                              # none of that started it
        self.job(gen)                                                # this does, and loads the model first
        wait_for(lambda: gen.active is None, what="the job to end")
        self.assertEqual((gen.totals["starts"], gen.totals["jobs"]), (2, 1))

    def test_a_model_the_page_loads_is_the_one_loaded_after_a_stop(self):
        gen = self.make()
        gen.ensure_running()
        self.ready(gen)
        other = {"recipe": "flux2-klein-4b", "models": {"diffusion_model": "diffusion_models/dit.gguf",
                                                        "llm": "text_encoders/te.gguf"}}
        reply = gen.handle(self.svc, "POST", images_api.LOAD, body=json.dumps(other).encode(), ctype="application/json")
        self.assertEqual(reply[0], 200)
        self.assertEqual(gen.want, other)
        bad = {"recipe": "x", "models": {"vae": "vae/missing.safetensors"}}
        self.assertEqual(gen.handle(self.svc, "POST", images_api.LOAD, body=json.dumps(bad).encode())[0], 400)
        self.assertEqual(gen.want, other)                            # a failed load changes nothing
        gen.stop("test")
        gen.handle(self.svc, "GET", "/sdcpp/v1/model-files")         # not cached yet: starts it
        self.ready(gen)
        self.assertEqual(body_of(gen.handle(self.svc, "GET", images_api.STATUS))["models"], other["models"])

    def test_unload_while_stopped(self):
        gen = self.make()
        gen.ensure_running()
        self.ready(gen)
        gen.stop("test")
        reply = gen.handle(self.svc, "POST", images_api.UNLOAD)
        self.assertEqual((reply[0], body_of(reply)["state"]), (200, "empty"))
        self.assertIsNone(gen.want)
        self.assertFalse(gen.running())
        self.assertEqual(body_of(gen.handle(self.svc, "GET", images_api.STATUS))["state"], "empty")

    def test_a_start_that_fails_says_why(self):
        gen = self.make()
        with mock.patch.dict(os.environ, {"FAKE_SD_EXIT": "1"}):
            code, _, data = gen.handle(self.svc, "GET", "/sdcpp/v1/capabilities")
        self.assertEqual(code, 503)
        self.assertIn("ErrorDeviceLost", json.loads(data)["error"])
        self.assertFalse(gen.running())
        with mock.patch.dict(os.environ, {"FAKE_SD_EXIT": "1"}):
            reply = gen.handle(self.svc, "POST", JOB, body=b'{"prompt": "x"}')
        self.assertEqual(reply[0], 503)
        self.assertFalse(self.svc.fifo.locked())


@mock.patch.object(images_api, "server_args", fake_args)
class OverHttp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tok = ByteTokenizer()
        self.engine = MockEngine(tok, "</think>\n\nok")
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.images = images_api.ImageServer(image_cfg(self.tmp.name))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        if self.svc.images is not None:
            self.svc.images.close()
        self.tmp.cleanup()

    def call(self, path, body=None, headers=None, method=None):
        data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
        h = {"Content-Type": "application/json"} if data is not None else {}
        req = urllib.request.Request(self.base + path, data=data, headers={**h, **(headers or {})}, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, r.read(), r.headers
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.read(), e.headers

    def test_the_page_and_its_api(self):
        code, html, h = self.call("/sd/")
        self.assertEqual((code, h["Content-Type"]), (200, "text/html; charset=utf-8"))
        self.assertIn(b"/*webui*/", html)
        self.assertIn(b"sdcpp-webui-enhance-base-url", html)
        self.assertEqual(h["X-Frame-Options"], "SAMEORIGIN")
        self.assertEqual(self.call("/sd")[0], 200)
        code, data, h = self.call("/sd/sdcpp/v1/capabilities?x=1")
        self.assertEqual((code, json.loads(data)["query"]), (200, "x=1"))
        self.assertIsNone(h.get("Access-Control-Allow-Origin"))      # sd-server's open CORS stays behind
        self.assertEqual(self.call("/sd/v1/images/generations", {"prompt": "x"})[0], 404)   # its other APIs too

    def test_a_chat_request_waits_for_the_image_job(self):
        gen = self.svc.images
        wait_for(lambda: json.loads(self.call("/sd/sdcpp/v1/models/status")[1])["state"] == "ready",
                 what="the model to load")
        with mock.patch.dict(os.environ, {"FAKE_SD_JOB_S": "1.5"}):
            code, data, _ = self.call("/sd/sdcpp/v1/img_gen", {"prompt": "a lighthouse"})
            self.assertEqual(code, 202, data)
            job_id = json.loads(data)["id"]
            code, _, _ = self.call("/v1/chat/completions", {"model": "m", "max_tokens": 8,
                                                            "messages": [{"role": "user", "content": "hi"}]})
            self.assertEqual(code, 200)
            self.assertEqual(gen.totals["jobs"], 1)                  # the answer came after the image job
        code, data, _ = self.call(f"/sd/sdcpp/v1/jobs/{job_id}")
        self.assertEqual(json.loads(data)["status"], "completed")

    def test_other_sites_are_refused(self):
        self.assertEqual(self.call("/sd/", headers={"Sec-Fetch-Site": "cross-site"})[0], 403)
        self.assertEqual(self.call("/sd/sdcpp/v1/models/status", headers={"Sec-Fetch-Site": "same-site"})[0], 403)
        self.assertEqual(self.call("/sd/", headers={"Sec-Fetch-Site": "same-origin"})[0], 200)
        code, _, _ = self.call("/sd/sdcpp/v1/img_gen", {"prompt": "x"}, headers={"Origin": "http://evil.example"})
        self.assertEqual(code, 403)
        self.assertIsNone(self.svc.images.active)

    def test_with_an_api_key_the_page_needs_its_cookie(self):
        self.svc.api_key = api_key_of("sekrit")
        self.assertEqual(self.call("/sd/")[0], 401)
        self.assertEqual(self.call("/sd/session", {})[0], 401)
        code, _, h = self.call("/sd/session", {}, headers={"Authorization": "Bearer sekrit"})
        self.assertEqual(code, 200)
        cookie = h["Set-Cookie"]
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Strict", cookie)
        self.assertNotIn("Path=", cookie)
        pair = cookie.split(";")[0]
        self.assertEqual(self.call("/sd/", headers={"Cookie": f"other=1; {pair}"})[0], 200)
        self.assertEqual(self.call("/sd/", headers={"Cookie": pair + "x"})[0], 401)
        self.assertEqual(self.call("/sd/sdcpp/v1/recipes", headers={"Authorization": "Bearer sekrit"})[0], 200)

    def test_the_openai_images_api_is_gone_and_the_status_says_what_runs(self):
        self.assertEqual(self.call("/v1/images/generations", {"prompt": "x"})[0], 404)
        ids = [m["id"] for m in json.loads(self.call("/v1/models")[1])["data"]]
        self.assertNotIn("flux2-klein-4b", ids)
        self.assertTrue(json.loads(self.call("/health")[1])["image_generation"])
        st = json.loads(self.call("/v1/status")[1])["image_generation"]
        self.assertEqual((st["enabled"], st["server"], st["model"]), (True, "stopped", "flux2-klein-4b"))

    def test_not_set_up(self):
        self.svc.images = None
        code, data, _ = self.call("/sd/")
        self.assertEqual((code, json.loads(data)["error"]["code"]), (404, "images_not_set_up"))


if __name__ == "__main__":
    unittest.main()
