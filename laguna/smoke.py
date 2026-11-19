#!/usr/bin/env python3
"""laguna/smoke.py -- run ON ECHO: the Laguna port's first run, through strata-poly's own line protocol.

    sudo systemctl stop polystrata                 # the service holds both cards
    ~/PolyStrata/.venv/bin/python ~/laguna-test/smoke.py [--llama DIR] [--kv q8_0] [--max-new N]
    sudo systemctl start polystrata

1. the tokenizer folder from the GGUF (tools/strata_tokenizer.py, with its round-trip check), once
2. with --llama DIR (the folder of a Laguna-capable llama.cpp's binaries): each prompt's ids from llama-tokenize
   next to ours
3. the dev engine (~/src/PolyStrata/build-poly/strata-poly --serve), loaded once, greedy, on both cards
4. three prompts: a short one, code, and one of about 1,500 tokens (past the 512-token window rings, and read as a
   prompt batch); each one's text, prompt and decode speed
5. the engine's own line on how it turns the full layers (expected: pairs 9 to 18, magnitude 1.3466)

Everything goes to ~/laguna-test/: the engine's log (engine.log) and the results (smoke-<time>.txt).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time

HOME = pathlib.Path.home()
p = argparse.ArgumentParser()
p.add_argument("--src", default=str(HOME / "src/PolyStrata"))
p.add_argument("--gguf", default=str(HOME / "PolyStrata-data/models/laguna-s21-UD-Q3_K_XL/UD-Q3_K_XL/"
                                            "Laguna-S-2.1-UD-Q3_K_XL-00001-of-00003.gguf"))
p.add_argument("--out", default=str(HOME / "laguna-test"))
p.add_argument("--llama", default="", help="folder of a Laguna-capable llama.cpp's binaries (llama-tokenize)")
p.add_argument("--max-new", type=int, default=64)
p.add_argument("--context", type=int, default=16384)
p.add_argument("--kv", default="", help="fp16 (default) or q8_0")
a = p.parse_args()

src, out, gguf = pathlib.Path(a.src), pathlib.Path(a.out), pathlib.Path(a.gguf)
out.mkdir(parents=True, exist_ok=True)
stamp = time.strftime("%Y%m%d-%H%M%S")
report = open(out / f"smoke-{stamp}.txt", "w", encoding="utf-8")


def say(*parts):
    line = " ".join(str(x) for x in parts)
    print(line, flush=True)
    report.write(line + "\n")
    report.flush()


if subprocess.run(["systemctl", "is-active", "--quiet", "polystrata"]).returncode == 0:
    sys.exit("polystrata.service holds the cards: sudo systemctl stop polystrata first, and start it again after")
engine_exe = src / "build-poly/strata-poly"
for need in (engine_exe, gguf, src / "tools/strata_tokenizer.py", src / "src/poly/laguna.cpp"):
    if not need.exists():
        sys.exit(f"missing: {need}" + ("  (git pull in ~/src/PolyStrata first)" if need.name == "laguna.cpp" else ""))
# an engine built before the Laguna code would refuse the model at load
if engine_exe.stat().st_mtime < (src / "src/poly/laguna.cpp").stat().st_mtime:
    sys.exit(f"{engine_exe} is older than src/poly/laguna.cpp: build it first\n"
             "  cd ~/src/PolyStrata && PATH=~/PolyStrata/.venv/bin:$PATH cmake --build build-poly "
             "--target strata-poly poly_sink_attend_test -j 8")

# 1) the tokenizer
tok_dir = out / "tok"
if not (tok_dir / "tokenizer" / "vocab.json").exists():
    say("== extracting the tokenizer from the GGUF (with the round-trip check)")
    subprocess.run([sys.executable, str(src / "tools/strata_tokenizer.py"), "--gguf", str(gguf), "--out",
                    str(tok_dir), "--check"], check=True)
sys.path.insert(0, str(src / "tools"))
from strata_tokenizer import Tokenizer  # noqa: E402

tk = Tokenizer.from_dir(tok_dir / "tokenizer")
say(f"== tokenizer: {len(tk.tokens)} tokens, pre {tk.pre}")

# 2) the prompts.  BOS first: the GGUF has add_bos_token, and llama.cpp puts it before a raw prompt.
readme = (src / "README.md").read_text(encoding="utf-8", errors="replace")
long_text = readme[:6500].rsplit("\n", 1)[0]
prompts = {
    "short": "The capital of France is",
    "code": 'def is_prime(n: int) -> bool:\n    """Return True if n is a prime number."""\n',
    "long": long_text + "\n\nIn one sentence, the text above describes",
}
BOS = 2
ids = {k: [BOS] + tk.encode(v) for k, v in prompts.items()}
for k, v in ids.items():
    say(f"   {k}: {len(v)} tokens")
exe = None
if a.llama:   # the binaries' folder, or llama-tokenize itself
    exe = pathlib.Path(a.llama).expanduser()
    if exe.is_dir():
        exe = exe / "llama-tokenize"
    if not exe.is_file():
        say(f"== no llama-tokenize at {exe}: the tokenizer parity is skipped")
        exe = None
if exe:
    say(f"== tokenizer parity against {exe}")
    for k, text in prompts.items():
        r = subprocess.run([str(exe), "-m", str(gguf), "-p", text, "--ids", "--log-disable"], capture_output=True,
                           text=True)
        got = None
        if r.returncode == 0:   # --ids prints the ids as a list, [2, 785, ...]
            rows = [x for x in r.stdout.splitlines() if x.strip().startswith("[")]
            try:
                got = [int(x) for x in rows[-1].strip().strip("[]").replace(" ", "").split(",") if x]
            except (IndexError, ValueError):
                got = None
        if got is None:
            say(f"   {k}: llama-tokenize failed: {r.stderr.strip()[-300:]}")
        elif got == ids[k]:
            say(f"   {k}: identical ({len(got)} ids)")
        else:
            at = next((i for i, (x, y) in enumerate(zip(got, ids[k])) if x != y), min(len(got), len(ids[k])))
            say(f"   {k}: DIFFERENT at {at}: llama.cpp {got[max(0, at - 3):at + 4]} ours {ids[k][max(0, at - 3):at + 4]}"
                f" (lengths {len(got)} / {len(ids[k])})")

# 3) the engine, once for all the prompts
cmd = [str(engine_exe), "--serve", "--model", str(gguf), "--max-context", str(a.context), "--layer-split", "auto",
       "--eos-ids", "2,24"] + (["--kv", a.kv] if a.kv else [])
say("== engine:", " ".join(cmd))
log = open(out / "engine.log", "w", encoding="utf-8")
t0 = time.time()
eng = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)


def line_from_engine() -> str:
    got = eng.stdout.readline()
    if not got:
        log.flush()
        tail = (out / "engine.log").read_text(encoding="utf-8", errors="replace")[-3000:]
        say("!! the engine stopped. The end of engine.log:\n" + tail)
        sys.exit(1)
    return got.rstrip("\n")


while True:
    got = line_from_engine()
    if got.startswith("READY"):
        break
    if got.startswith("INFO"):
        say("  ", got)
say(f"   loaded in {time.time() - t0:.0f} s")

# 4) the prompts, greedy
for k, v in ids.items():
    eng.stdin.write(f"GEN {a.max_new} temperature=0 {','.join(map(str, v))}\n")
    eng.stdin.flush()
    toks, done = [], None
    while done is None:
        got = line_from_engine()
        if got.startswith("T "):
            toks.append(int(got[2:]))
        elif got.startswith("DONE "):
            done = got.split()
        elif got.startswith("ERR"):
            say(f"!! {k}: {got}")
            break
    if done is None:
        continue
    produced, n_prompt, prompt_ms, decode_ms = int(done[1]), int(done[2]), float(done[3]), float(done[4])
    pp = 1000.0 * n_prompt / prompt_ms if prompt_ms > 0 else 0.0
    tg = 1000.0 * produced / decode_ms if decode_ms > 0 else 0.0
    say(f"== {k}: {n_prompt} prompt tokens at {pp:.0f} tok/s, {produced} generated at {tg:.1f} tok/s ({done[5]})")
    say("   ids:", ",".join(map(str, toks)))
    say("   text:", repr(prompts[k][-60:] if k == "long" else prompts[k]), "->", repr(tk.decode(toks)))

eng.stdin.write("QUIT\n")
eng.stdin.flush()
try:
    eng.wait(timeout=120)
except subprocess.TimeoutExpired:
    eng.kill()
log.close()

# 5) the engine's own account of its rope, and anything it warned about
say("== from engine.log:")
for got in (out / "engine.log").read_text(encoding="utf-8", errors="replace").splitlines():
    if any(w in got for w in ("laguna:", "q8_0 KV", "loaded in", "routed experts in VRAM", "WARNING", "error",
                              "Error", "nan", "NaN")):
        say("  ", got[:240])
say(f"== results: {out / f'smoke-{stamp}.txt'}")
