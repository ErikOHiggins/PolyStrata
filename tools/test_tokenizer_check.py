"""tools/tokenizer_check.py: the comparison with a running llama-server, against a fake one on this PC (no GPU, no model).

    python -m unittest tools.test_tokenizer_check
"""
from __future__ import annotations

import http.server
import json
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import strata_tokenizer as ST  # noqa: E402
import tokenizer_check as TC  # noqa: E402


def small_tokenizer() -> ST.Tokenizer:
    tokens = [ST.BYTE_TO_UNICODE[b] for b in range(256)] + ["<|im_start|>", "<|im_end|>"]
    return ST.Tokenizer(tokens, [], [1] * 256 + [3, 3], pre="pixtral")


class FakeLlamaServer:
    """POST /tokenize as llama-server answers it, from `tok` (and `wrong`: texts answered one id off)."""

    def __init__(self, tok, wrong=()):
        tok_, wrong_ = tok, set(wrong)

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                ids = tok_.encode(body["content"], parse_special=body["parse_special"])
                if body["content"] in wrong_:
                    ids = ids[:-1] + [ids[-1] + 1]
                data = json.dumps({"tokens": ids}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class Check(unittest.TestCase):
    def setUp(self):
        self.tok = small_tokenizer()

    def run_main(self, server):
        out = []
        with mock.patch.object(TC.st.Tokenizer, "from_gguf", lambda path: self.tok), \
                mock.patch("builtins.print", lambda *a, **k: out.append(" ".join(map(str, a)))):
            code = TC.main(["--gguf", "x.gguf", "--server", server])
        return code, "\n".join(out)

    def test_the_same_ids_everywhere(self):
        srv = FakeLlamaServer(self.tok)
        self.addCleanup(srv.close)
        code, out = self.run_main(srv.url)
        self.assertEqual(code, 0, out)
        self.assertIn("give the same ids as llama-server", out)

    def test_a_difference_is_shown_with_its_tokens(self):
        srv = FakeLlamaServer(self.tok, wrong=["Hello, world!", TC.SPECIAL[0]])
        self.addCleanup(srv.close)
        code, out = self.run_main(srv.url)
        self.assertEqual(code, 1, out)
        self.assertIn("differs: 'Hello, world!'", out)
        self.assertIn("differs (special tokens matched)", out)

    def test_no_server(self):
        code, out = self.run_main("http://127.0.0.1:9")
        self.assertEqual(code, 2, out)
        self.assertIn("did not answer /tokenize", out)

    def test_the_corpus_is_large_and_varied(self):
        c = TC.corpus()
        self.assertGreater(len(c), 150)
        self.assertTrue(any("\u4f60" in s for s in c) and any("\U0001f600" in s for s in c))


if __name__ == "__main__":
    unittest.main()
