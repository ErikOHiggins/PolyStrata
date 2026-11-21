"""tools/llama_server_check.py's comparison (the servers themselves run on echo, not here).

    python -m unittest tools.test_llama_server_check
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import llama_server_check as C  # noqa: E402


class Compare(unittest.TestCase):
    def test_first_difference(self):
        self.assertIsNone(C.first_difference("same text", "same text"))
        d = C.first_difference("MEN WALK ON MOON", "MEN WALK ON NOON")
        self.assertIn("at character 12", d)
        self.assertIn("'MEN WALK ON MOON'", d)
        self.assertIn("at character 3", C.first_difference("abc", "abcdef"))

    def test_an_answer_and_its_prompt_count(self):
        reply = {"choices": [{"message": {"content": "Hello!", "reasoning_content": "brief"}}],
                 "usage": {"prompt_tokens": 327}}
        self.assertEqual(C.answer(reply), ("brief", "Hello!", 327))
        self.assertEqual(C.answer({"choices": [{"message": {"content": "x"}}]}), ("", "x", None))

    def test_an_http_error_says_the_servers_own_words(self):
        e = C.urllib.error.HTTPError("http://x", 400, "Bad Request", {}, io.BytesIO(
            b'{"error": {"code": 400, "message": "Failed to tokenize prompt"}}'))
        self.assertEqual(C.error_words(e), "HTTP 400: Failed to tokenize prompt")
        self.assertEqual(C.error_words(RuntimeError("it stopped")), "it stopped")

    def test_only_a_llama_server_config(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "strata-x.json"
            cfg.write_text(json.dumps({"exe": "strata-poly", "args": []}))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(C.main(["--config", str(cfg)]), 2)
            self.assertIn("not a llama-server model's run config", out.getvalue())


if __name__ == "__main__":
    unittest.main()
