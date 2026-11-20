#!/usr/bin/env python3
"""laguna/longctx.py -- Laguna's prompt and decode speed as the context grows, per KV format.

    python laguna/longctx.py --gguf MODEL.gguf [--kv fp16,q8_0] [--depths 2048,32768,...] [--lookup] [--service NAME]

One growing conversation per KV format, so the document is read once: the prompt is the first 2K tokens of the
repository's docs (at --corpus), then the first 32K, and so on.  Each one goes on from the one before (the engine keeps a checkpoint
at a prompt's end), so a step reads only its new tokens, and its prompt speed is the speed at that depth.  After
each, 128 tokens are written, greedy.  Prompt-lookup guesses are off (--lookup 0) unless --lookup: the decode figure
is then the model's own pace, not the document's repeats.

Per step: the tokens read and their speed, the decode speed, and the share of the routed experts the decode found in
VRAM (the attention caches take VRAM from the expert cache as they grow).  To --out (laguna-test/ here by default): longctx-<time>.txt,
and the engine's log per format (longctx-<time>-<kv>.log).
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]   # this checkout
p = argparse.ArgumentParser()
p.add_argument("--src", default=str(ROOT), help="the checkout whose engine (build-poly/strata-poly) and tools run "
               "(default: this one)")
p.add_argument("--tools", default="", help="the folder of strata_tokenizer.py (default: <src>/tools); a checkout without Laguna's pre-tokenizer takes another's")
p.add_argument("--gguf", required=True, help="the model's GGUF (the first file of a split one)")
p.add_argument("--out", default="laguna-test", help="where the results go (default: laguna-test/ here)")
p.add_argument("--service", default="", help="a systemd service that holds the GPUs: the run is refused while it is active (none by default)")
p.add_argument("--corpus", default="ac62d979", help="the git revision whose README.md and docs/*.md are the long prompts (the runs so far used ac62d979's; the docs have changed since)")
p.add_argument("--kv", default="fp16,q8_0", help="the KV formats to run, in turn")
p.add_argument("--depths", default="2048,32768,65536,131072,172032,200704")
p.add_argument("--max-new", type=int, default=128)
p.add_argument("--lookup", action="store_true", help="leave prompt-lookup guesses on (the engine's default)")
p.add_argument("--extra", default="", help="more strata-poly options, as one string")
argv = sys.argv[1:]
for i in range(len(argv) - 1):   # --extra "--foo": argparse would take the value for an option
    if argv[i] == "--extra":
        argv[i:i + 2] = ["--extra=" + argv[i + 1]]
        break
a = p.parse_args(argv)

src, out, gguf = (pathlib.Path(x).expanduser() for x in (a.src, a.out, a.gguf))
depths = sorted(int(x) for x in a.depths.split(","))
context = depths[-1] + a.max_new + 1024
stamp = time.strftime("%Y%m%d-%H%M%S")
report = open(out / f"longctx-{stamp}.txt", "w", encoding="utf-8")


def say(*parts):
    line = " ".join(str(x) for x in parts)
    print(line, flush=True)
    report.write(line + "\n")
    report.flush()


if a.service and subprocess.run(["systemctl", "is-active", "--quiet", a.service]).returncode == 0:
    sys.exit(f"{a.service} holds the cards: sudo systemctl stop {a.service} first, and start it again after")
engine_exe = src / "build-poly/strata-poly"
for need in (engine_exe, gguf, out / "tok/tokenizer/vocab.json"):
    if not need.exists():
        sys.exit(f"missing: {need}")
sys.path.insert(0, a.tools or str(src / "tools"))
from strata_tokenizer import Tokenizer  # noqa: E402

tk = Tokenizer.from_dir(out / "tok/tokenizer")
say("== strata-poly:", subprocess.run(["git", "-C", str(src), "log", "-1", "--format=%h %s"], capture_output=True,
                                      text=True).stdout.strip())
def at_rev(rel: str) -> str:
    """A file of the checkout as it was at --corpus: the long prompts stay the same whatever the docs are now."""
    r = subprocess.run(["git", "-C", str(src), "show", f"{a.corpus}:{rel}"], capture_output=True)
    if r.returncode != 0:
        sys.exit(f"cannot read {rel} at {a.corpus}: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout.decode("utf-8", errors="replace").replace("\r\n", "\n")


# the document: the repository's docs at --corpus, one file after another
parts = []
names = subprocess.run(["git", "-C", str(src), "ls-tree", "--name-only", a.corpus, "docs/"], capture_output=True,
                       text=True).stdout.split()
for rel in sorted(n for n in names if n.endswith(".md")):
    parts.append(f"\n\n===== {pathlib.Path(rel).name} =====\n\n" + at_rev(rel))
doc = [2] + tk.encode("".join(parts))
if len(doc) < depths[-1]:
    sys.exit(f"the docs come to {len(doc)} tokens, fewer than the deepest step ({depths[-1]})")
say(f"== document: {len(doc):,} tokens of docs/*.md at {a.corpus}; steps {', '.join(f'{d:,}' for d in depths)}; "
    f"context {context:,}; {a.max_new} tokens written at each; lookup {'on' if a.lookup else 'off'}")

rows = []
for kv in [x.strip() for x in a.kv.split(",") if x.strip()]:
    cmd = [str(engine_exe), "--serve", "--model", str(gguf), "--max-context", str(context), "--layer-split", "auto",
           "--eos-ids", "2,24", "--conversation-cache-mib", "0", "--expert-profile", str(out / "expert-profile.txt")]
    cmd += [] if a.lookup else ["--lookup", "0"]
    cmd += [] if kv == "fp16" else ["--kv", kv]
    cmd += a.extra.split()
    logp = out / f"longctx-{stamp}-{kv}.log"
    say(f"\n== {kv}: {' '.join(cmd)}")
    log = open(logp, "w", encoding="utf-8")
    t0 = time.time()
    eng = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)

    def line_from_engine() -> str:
        got = eng.stdout.readline()
        if not got:
            log.flush()
            say("!! the engine stopped; the end of its log:\n" + logp.read_text(errors="replace")[-3000:])
            sys.exit(1)
        return got.rstrip("\n")

    while True:
        got = line_from_engine()
        if got.startswith("READY"):
            break
        if got.startswith("INFO"):
            say("  ", got)
    say(f"   loaded in {time.time() - t0:.0f} s")
    say(f"   {'depth':>8} {'read':>8} {'prompt tok/s':>13} {'decode tok/s':>13} {'experts from VRAM':>18} {'wall s':>7}")
    for d in depths:
        eng.stdin.write(f"GEN {a.max_new} temperature=0 {','.join(map(str, doc[:d]))}\n")
        eng.stdin.flush()
        t1, done = time.time(), None
        while done is None:
            got = line_from_engine()
            if got.startswith("DONE "):
                done = got.split()
            elif got.startswith("ERR"):
                say(f"!! {kv} at {d}: {got}")
                break
        if done is None:
            break
        produced, prompt_ms, decode_ms = int(done[1]), float(done[3]), float(done[4])
        hits, lookups, read = int(done[9]), int(done[10]), int(done[14])
        pp = 1000.0 * read / prompt_ms if prompt_ms > 0 else 0.0
        tg = 1000.0 * produced / decode_ms if decode_ms > 0 else 0.0
        share = 100.0 * hits / lookups if lookups else 0.0
        rows.append((kv, d, read, pp, tg, share))
        say(f"   {d:>8,} {read:>8,} {pp:>13.0f} {tg:>13.1f} {share:>17.1f}% {time.time() - t1:>7.0f}")
    eng.stdin.write("QUIT\n")
    eng.stdin.flush()
    try:
        eng.wait(timeout=180)
    except subprocess.TimeoutExpired:
        eng.kill()
    log.close()
    for got in logp.read_text(errors="replace").splitlines():
        if any(w in got for w in ("loaded in", "expert cache", "q8_0 KV", "WARNING", "rror")):
            say("   log:", got[:200])

say("\n== summary (decode tok/s by depth)")
for kv in dict.fromkeys(r[0] for r in rows):
    say(f"   {kv:>6}: " + "  ".join(f"{r[1] // 1024}K {r[4]:.1f}" for r in rows if r[0] == kv))
say(f"== results: {report.name}")
