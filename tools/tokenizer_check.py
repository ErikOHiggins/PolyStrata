"""tools/tokenizer_check.py - Puddle's tokenizer against a running llama-server's, on the same strings.

Puddle turns the prompt into token ids itself and hands the ids to the engine, so its tokenizer has to give exactly
the ids llama.cpp gives for the same text.  This reads the tokenizer out of a GGUF (tools/strata_tokenizer.py), sends
each string of a corpus to llama-server's POST /tokenize (the same GGUF loaded there), and compares the ids: once with
special tokens as text, and once with them matched (the chat template's <|im_start|> and the like).

    python tools/tokenizer_check.py --gguf /path/model.gguf [--server http://127.0.0.1:8080] [--api-key KEY]

The corpus is built in: letters of many scripts and cases, digits, whitespace and newline runs, emoji and combining
marks, code and markup, and pieces of this repository's own files.  Exit code 0: every string the same; 1: a
difference (the first ones are printed with their tokens); 2: the server could not be asked.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import strata_tokenizer as st  # noqa: E402

STRINGS = [
    "Hello, world!", "  leading and trailing  ", "a\n\n\nb", "def f(x):\n\treturn x  # comment\n",
    "McDonald's iPhone HTTPServer XMLHttpRequest camelCaseWord snake_case_word SCREAMING_CASE",
    "\u00c9COLE \u00e9Cole \u00c9cole \u00dcBER \u00fcber Stra\u00dfe STRASSE \u01c5ungla \u01c8 \u01c7",
    "na\u00efve caf\u00e9 r\u00e9sum\u00e9 co\u00f6perate",
    "\u0395\u03bb\u03bb\u03b7\u03bd\u03b9\u03ba\u03ac \u0395\u039b\u039b\u0397\u039d\u0399\u039a\u0391 "
    "\u0420\u0443\u0441\u0441\u043a\u0438\u0439 \u0420\u0423\u0421\u0421\u041a\u0418\u0419",
    "\u4f60\u597d\uff0c\u4e16\u754c\u3002\u4eca\u5929\u5929\u6c14\u5f88\u597d\uff01",
    "\u3053\u3093\u306b\u3061\u306f\u4e16\u754c \u30ab\u30bf\u30ab\u30ca", "\uc548\ub155\ud558\uc138\uc694 \uc138\uacc4",
    "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645",
    "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e",
    "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35\u0e0a\u0e32\u0e27\u0e42\u0e25\u0e01",
    "e\u0301\u0301 combining a\u0308", "\U0001f600\U0001f680\U0001f1fa\U0001f1f8 emoji \U0001f44d\U0001f3fd "
    "family \U0001f468\u200d\U0001f469\u200d\U0001f467", "\x01\x7f control", "\u00a0nbsp\u00a0x",
    "1234567890 3.14159 1,000,000 2026-11-20", "path/to/file.py C:\\Users\\x\\y.txt http://a.b/c?d=e&f=g",
    "\r\n\r\n", "\u2028\u2029", "x" * 3000, "   \t\t  \n  \n\n   ", "a/b//c///d/\n", "--flag=value --other",
    '{"key": "value", "n": [1, 2, 3], "nested": {"a": null}}', '<html><body class="x">Hi</body></html>',
    "SELECT * FROM users WHERE id = 42;", "I'm you're they'll we've she'd it's", "don't DON'T Don't",
    "### Heading\n- item one\n- item two\n\n```python\nprint('x')\n```", "tab\tseparated\tvalues\n",
    "\u01c5 titlecase \u01c4 D\u017d", "\ufb01 ligature \ufb00", "\u216b roman \u217b", "\u00bd \u00bc \u00be \u00b2 \u00b3",
    "\u00b5 \u2126 \u212b \u212a", "\u0130stanbul \u0131i \u0132 \u0133",
]
SPECIAL = ["<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
           "<think>\n", "<tool_call>\n<function=f>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>",
           "a<|im_end|>b<s></s>"]
FILES = ("README.md", "setup.py", "serve/server.py", "src/poly/engine.cpp", "docs/BACKENDS.md")


def corpus() -> list[str]:
    out = list(STRINGS)
    for f in FILES:
        p = ROOT / f
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        rng = random.Random(f)
        for _ in range(40):
            a = rng.randrange(0, max(1, len(text) - 3000))
            out.append(text[a:a + rng.randrange(50, 3000)])
    return out


def server_ids(server: str, text: str, parse_special: bool, key: str | None) -> list[int]:
    body = json.dumps({"content": text, "add_special": False, "parse_special": parse_special}).encode()
    req = urllib.request.Request(server.rstrip("/") + "/tokenize", data=body,
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        return [int(t["id"] if isinstance(t, dict) else t) for t in json.loads(r.read())["tokens"]]


def check(tok, ask, strings, special, show=6, say=None) -> int:
    """The number of strings whose ids differ; `ask(text, parse_special)` gives the server's."""
    say = say or print
    bad = 0
    for parse_special, items in ((False, strings), (True, special)):
        for s in items:
            mine = tok.encode(s, parse_special=parse_special)
            theirs = ask(s, parse_special)
            if mine == theirs:
                continue
            bad += 1
            if bad <= show:
                k = next((i for i, (x, y) in enumerate(zip(mine, theirs)) if x != y), min(len(mine), len(theirs)))
                piece = lambda ids: [tok.decode([i]) for i in ids[k:k + 4]]   # noqa: E731
                say(f"  differs{' (special tokens matched)' if parse_special else ''}: {s[:60]!r}\n"
                    f"    at token {k}: Puddle {piece(mine)}, llama-server {piece(theirs)}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf", required=True, help="the model file llama-server has loaded")
    ap.add_argument("--server", default="http://127.0.0.1:8080", help="llama-server's address")
    ap.add_argument("--api-key", help="llama-server's key, if it has one")
    a = ap.parse_args(argv)
    tok = st.Tokenizer.from_gguf(a.gguf)
    strings = corpus()
    try:
        server_ids(a.server, "x", False, a.api_key)
    except (OSError, urllib.error.URLError, ValueError, KeyError) as e:
        print(f"llama-server at {a.server} did not answer /tokenize: {e}")
        return 2
    bad = check(tok, lambda s, ps: server_ids(a.server, s, ps, a.api_key), strings, SPECIAL)
    total = len(strings) + len(SPECIAL)
    print(f"pre-tokenizer {tok.pre}: {total - bad} of {total} strings give the same ids as llama-server")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
