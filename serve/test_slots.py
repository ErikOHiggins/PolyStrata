"""serve/test_slots.py - llama-server's /slots/0?action=save|restore: strata-poly keeps no disk sessions (the Qwen
engine's SAVE / RESTORE are gone, phase 1d), so a save or restore is refused plainly (501) and nothing reaches the
engine; a foreign page or a form is refused before that, as for every POST.  logit_bias, also the Qwen engine's
alone, is refused the same plain way (mock engine: no GPU).

    python -m unittest serve.test_slots -v
"""
from __future__ import annotations

import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class Served(unittest.TestCase):
    """A server over the mock engine."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = MockEngine(tok, "hi")
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers=headers or {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")


class Slots(Served):
    def test_save_and_restore_are_refused_plainly(self):
        for action in ("save", "restore"):
            with self.subTest(action=action):
                s, b = self.post(f"/slots/0?action={action}", {"filename": "x.bin"})
                self.assertEqual(s, 501, b)
                self.assertIn("not supported by strata-poly", b["error"]["message"])

    def test_a_foreign_page_or_a_form_is_refused_first(self):
        for action in ("save", "restore"):
            for headers, code in (({"Content-Type": "application/json", "Origin": "http://evil.example.com"}, 403),
                                  ({"Content-Type": "text/plain"}, 415),
                                  ({"Content-Type": "application/x-www-form-urlencoded"}, 415)):
                s, b = self.post(f"/slots/0?action={action}", {"filename": "x.bin"}, headers)
                self.assertEqual(s, code, (action, headers, b))


class LogitBias(Served):
    """logit_bias was the Qwen engine's: a request that asks for one is refused plainly (400); an empty one is not a
    request for it."""

    def test_refused_plainly(self):
        chat = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
        s, b = self.post("/v1/chat/completions", {**chat, "logit_bias": {"42": -100}})
        self.assertEqual(s, 400, b)
        self.assertIn("logit_bias is not supported by strata-poly", b["error"]["message"])
        s, b = self.post("/v1/chat/completions", {**chat, "logit_bias": {}})
        self.assertEqual(s, 200, b)


if __name__ == "__main__":
    unittest.main()
