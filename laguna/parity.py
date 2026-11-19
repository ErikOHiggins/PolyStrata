#!/usr/bin/env python3
"""laguna/parity.py -- run ON ECHO: Laguna on strata-poly against llama.cpp on the same GGUF, token by token.

    sudo systemctl stop polystrata
    ~/PolyStrata/.venv/bin/python ~/laguna-test/parity.py --llama ~/llama.cpp/build/bin [--n 64] [--kv q8_0]
    sudo systemctl start polystrata

1. the prompts, as ids from the tokenizer smoke.py extracted (identical to llama-tokenize's there): the three
   smoke prompts and three longer ones (about 600, 1,500 and 4,000 tokens) from the repository's docs
2. llama.cpp (llama-server, every routed expert on the CPU as the handoff's gate asks) answers each one greedily,
   with the probabilities of its 10 likeliest tokens at every step
3. strata-poly is made to write llama.cpp's answer (--follow) and prints its own 10 likeliest at every step
   (--top-probs), with the expert split held still (STRATA_FREEZE_EXPERTS=1) and no guesses (--window 1)
4. per prompt: how many steps the two engines' first choices agree, the first step where they do not (and
   whether it is a near tie), and KL(llama.cpp || strata-poly) over llama.cpp's 10 likeliest tokens

Everything goes to ~/laguna-test/: parity-<time>.txt, and the raw answers in parity-<time>.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

HOME = pathlib.Path.home()
p = argparse.ArgumentParser()
p.add_argument("--src", default=str(HOME / "src/PolyStrata"))
p.add_argument("--tools", default="", help="the folder of strata_tokenizer.py (default: <src>/tools); a checkout without Laguna's pre-tokenizer takes another's")
p.add_argument("--gguf", default=str(HOME / "PolyStrata-data/models/laguna-s21-UD-Q3_K_XL/UD-Q3_K_XL/"
                                            "Laguna-S-2.1-UD-Q3_K_XL-00001-of-00003.gguf"))
p.add_argument("--out", default=str(HOME / "laguna-test"))
p.add_argument("--llama", default="", help="folder of a Laguna-capable llama.cpp's binaries (llama-server)")
p.add_argument("--n", type=int, default=64, help="tokens each answer has")
p.add_argument("--context", type=int, default=16384)
p.add_argument("--kv", default="", help="strata-poly's --kv: fp16 (default) or q8_0")
p.add_argument("--port", type=int, default=8091)
p.add_argument("--only", default="", help="comma-separated prompt names, to run a few")
p.add_argument("--reuse-ref", default="", help="an earlier parity-*.json: take llama.cpp's answers from it, no server")
p.add_argument("--ours-args", default="", help="more strata-poly options, as one string (\"--no-gpu-experts\")")
p.add_argument("--against-ours", default="", help="an earlier parity-*.json on the same answers: also report KL "
               "between its strata-poly probabilities and this run's (the engine against itself)")
# --ours-args "--no-gpu-experts": argparse takes a lone value that starts with -- for an option, so it is joined
argv = sys.argv[1:]
for i in range(len(argv) - 1):
    if argv[i] == "--ours-args":
        argv[i:i + 2] = ["--ours-args=" + argv[i + 1]]
        break
a = p.parse_args(argv)
if not a.reuse_ref and not a.llama:
    p.error("--llama is needed (or --reuse-ref an earlier parity-*.json)")

src, out, gguf, bindir = (pathlib.Path(x).expanduser() for x in (a.src, a.out, a.gguf, a.llama))
stamp = time.strftime("%Y%m%d-%H%M%S")
report = open(out / f"parity-{stamp}.txt", "w", encoding="utf-8")
TOP = 10


def say(*parts):
    line = " ".join(str(x) for x in parts)
    print(line, flush=True)
    report.write(line + "\n")
    report.flush()


if subprocess.run(["systemctl", "is-active", "--quiet", "polystrata"]).returncode == 0:
    sys.exit("polystrata.service holds the cards: sudo systemctl stop polystrata first, and start it again after")
server = bindir / "llama-server"
engine_exe = src / "build-poly/strata-poly"
for need in ((engine_exe, gguf, out / "tok/tokenizer/vocab.json") if a.reuse_ref else
             (server, engine_exe, gguf, out / "tok/tokenizer/vocab.json")):
    if not need.exists():
        sys.exit(f"missing: {need}" + ("  (run smoke.py first: it extracts the tokenizer)" if "tok" in str(need) else ""))
sys.path.insert(0, a.tools or str(src / "tools"))
from strata_tokenizer import Tokenizer  # noqa: E402

tk = Tokenizer.from_dir(out / "tok/tokenizer")
if a.reuse_ref:
    say("== llama.cpp: the answers in", a.reuse_ref)
else:
    ver = subprocess.run([str(server), "--version"], capture_output=True, text=True)
    say("== llama.cpp:", " | ".join(x.strip() for x in (ver.stdout + ver.stderr).strip().splitlines()[-2:]) or "?")
say("== strata-poly options beside the defaults:", a.ours_args or "(none)", f"--kv {a.kv}" if a.kv else "")
say("== strata-poly:", subprocess.run(["git", "-C", str(src), "log", "-1", "--format=%h %s"], capture_output=True,
                                      text=True).stdout.strip())

# 1) the prompts
readme = (src / "README.md").read_text(encoding="utf-8", errors="replace")
details = (src / "docs/DETAILS.md").read_text(encoding="utf-8", errors="replace")


def cut(text: str, tokens: int) -> str:
    """The first lines of text that come to about `tokens` tokens."""
    lines, n, got = text.splitlines(keepends=True), 0, []
    for line in lines:
        n += len(tk.encode(line))
        if n > tokens:
            break
        got.append(line)
    return "".join(got)


prompts = {
    "short": "The capital of France is",
    "code": 'def is_prime(n: int) -> bool:\n    """Return True if n is a prime number."""\n',
    "long": readme[:6500].rsplit("\n", 1)[0] + "\n\nIn one sentence, the text above describes",
    "doc600": cut(details, 600) + "\n\nThe section above explains",
    "doc1500": cut(details, 1500) + "\n\nTo summarize the text above:",
    "doc4000": cut(details, 4000) + "\n\nThe most important setting described above is",
}
if a.only:
    prompts = {k: v for k, v in prompts.items() if k in a.only.split(",")}
BOS = 2
ids = {k: [BOS] + tk.encode(v) for k, v in prompts.items()}
for k, v in ids.items():
    say(f"   {k}: {len(v)} prompt tokens")

# 2) llama.cpp: every routed expert on the CPU, the rest on the cards (or its answers from an earlier run)
ref = {}
if a.reuse_ref:
    old = json.load(open(pathlib.Path(a.reuse_ref).expanduser()))
    for k in ids:
        if k not in old["ref"]:
            sys.exit(f"{a.reuse_ref} has no answer for {k}")
        if old["prompts"].get(k) != len(ids[k]):
            sys.exit(f"{k}: the prompt has {len(ids[k])} tokens now, {old['prompts'].get(k)} in {a.reuse_ref}")
        ref[k] = old["ref"][k]
        ref[k]["steps"] = [{"id": s["id"], "top": [tuple(x) for x in s["top"]]} for s in ref[k]["steps"]]
    say(f"   {len(ref)} answers taken from it")
cmd = [str(server), "-m", str(gguf), "-ngl", "99", "-ot", "exps=CPU", "-c", str(a.context), "-np", "1",
       "--host", "127.0.0.1", "--port", str(a.port)]
srv = None
if not a.reuse_ref:
    say("== llama-server:", " ".join(cmd))
    slog = open(out / f"llama-server-{stamp}.log", "w", encoding="utf-8")
    srv = subprocess.Popen(cmd, stdout=slog, stderr=subprocess.STDOUT)
base = f"http://127.0.0.1:{a.port}"
t0 = time.time()
while srv is not None:
    if srv.poll() is not None:
        sys.exit(f"llama-server stopped (code {srv.returncode}): see {slog.name}")
    try:
        with urllib.request.urlopen(base + "/health", timeout=5) as r:
            if r.status == 200:
                break
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        pass
    if time.time() - t0 > 900:
        srv.terminate()
        sys.exit("llama-server did not come up in 15 minutes")
    time.sleep(3)
if srv is not None:
    say(f"   up in {time.time() - t0:.0f} s")

try:
    for k, v in (ids.items() if srv is not None else ()):
        body = {"prompt": v, "n_predict": a.n, "temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0,
                "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "dry_multiplier": 0.0,
                "n_probs": TOP, "post_sampling_probs": False, "cache_prompt": False, "return_tokens": True}
        req = urllib.request.Request(base + "/completion", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t1 = time.time()
        with urllib.request.urlopen(req, timeout=3600) as r:
            res = json.loads(r.read())
        steps = []
        for e in res.get("completion_probabilities") or []:
            tops = e.get("top_logprobs") or e.get("top_probs") or []
            steps.append({"id": e.get("id"),
                          "top": [(t["id"], math.exp(t["logprob"]) if "logprob" in t else t["prob"]) for t in tops]})
        toks = res.get("tokens") or [s["id"] for s in steps]
        tm = res.get("timings") or {}
        ref[k] = {"tokens": toks, "steps": steps}
        say(f"   llama.cpp {k}: {len(toks)} tokens in {time.time() - t1:.0f} s "
            f"(prompt {tm.get('prompt_per_second', 0):.0f} tok/s, decode {tm.get('predicted_per_second', 0):.1f} tok/s)")
finally:
    if srv is not None:
        srv.terminate()
        try:
            srv.wait(timeout=60)
        except subprocess.TimeoutExpired:
            srv.kill()
        slog.close()


# 3) strata-poly, made to write llama.cpp's answers
def ours(prompt: list[int], follow: list[int]):
    env = dict(os.environ, STRATA_FREEZE_EXPERTS="1")
    cmd = [str(engine_exe), "--model", str(gguf), "--tokens", ",".join(map(str, prompt)),
           "--follow", ",".join(map(str, follow)), "--max-new", str(len(follow)), "--top-probs", str(TOP),
           "--max-context", str(a.context), "--layer-split", "auto", "--window", "1"] + (["--kv", a.kv] if a.kv else []) \
        + a.ours_args.split()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    (out / f"strata-{stamp}.log").open("a", encoding="utf-8").write(r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"strata-poly failed ({r.returncode}): {r.stderr[-1500:]}")
    tops = []
    for line in r.stderr.splitlines():
        if line.startswith("top "):
            tops.append([(int(i), float(q)) for i, q in (x.split(":") for x in line.split()[1:])])
    return tops


results = {}
allk, all_agree = [], [0, 0]
other = json.load(open(pathlib.Path(a.against_ours).expanduser())) if a.against_ours else None
self_k, self_agree = [], [0, 0]


def save():
    """The results so far: a run that is cut off keeps the prompts it finished."""
    json.dump({"ref": ref, "ours": results, "prompts": {k: len(v) for k, v in ids.items()},
               "ours_args": a.ours_args, "kv": a.kv}, open(out / f"parity-{stamp}.json", "w"))


def against_llama(k: str):
    lsteps, mine, toks = ref[k]["steps"], results[k], ref[k]["tokens"]
    n = min(len(toks), len(mine))
    agree, first, kls, ties = 0, None, [], 0
    for t in range(n):
        top1 = mine[t][0][0] if mine[t] else None
        if top1 == toks[t]:
            agree += 1
        else:
            if first is None:
                first = t
            lt = lsteps[t]["top"] if t < len(lsteps) else []
            if len(lt) >= 2 and len(mine[t]) >= 2 and min(lt[0][1] - lt[1][1], mine[t][0][1] - mine[t][1][1]) < 0.1:
                ties += 1
        if t < len(lsteps) and lsteps[t]["top"]:
            have = dict(mine[t])
            kls.append(sum(q * math.log(q / max(have.get(i, 0.0), 1e-6)) for i, q in lsteps[t]["top"] if q > 0))
    line = f"   vs llama.cpp: {agree} of {n} agree ({n - agree - ties} of the others not near ties)"
    if first is not None:
        lt = lsteps[first]["top"][:2] if first < len(lsteps) else []
        line += (f"; first differs at step {first}: llama.cpp {[(i, round(q, 3)) for i, q in lt]} "
                 f"strata-poly {[(i, round(q, 3)) for i, q in mine[first][:2]]}")
        if len(lt) == 2 and len(mine[first]) >= 2:
            gap_l, gap_s = lt[0][1] - lt[1][1], mine[first][0][1] - mine[first][1][1]
            line += " (near tie)" if min(gap_l, gap_s) < 0.1 else " (NOT a near tie)"
    if kls:
        line += f"; KL mean {sum(kls) / len(kls):.2e}, max {max(kls):.2e}"
        allk.extend(kls)
    all_agree[0] += agree
    all_agree[1] += n
    say(line)


def against_itself(k: str):
    theirs, mine = other["ours"].get(k), results[k]
    if not theirs or other["ref"].get(k, {}).get("tokens") != ref[k]["tokens"]:
        say(f"   vs {pathlib.Path(a.against_ours).name}: not in it, or it followed other tokens")
        return
    ks = []
    for t in range(min(len(mine), len(theirs))):
        have = {int(i): q for i, q in theirs[t]}
        ks.append(sum(q * math.log(q / max(have.get(i, 0.0), 1e-6)) for i, q in mine[t] if q > 0))
        self_agree[0] += int(mine[t][0][0] == int(theirs[t][0][0]))
        self_agree[1] += 1
    self_k.extend(ks)
    say(f"   vs strata-poly with options {other.get('ours_args') or '(none)'}: KL mean {sum(ks) / len(ks):.2e}, "
        f"max {max(ks):.2e}")


# 4) each prompt: strata-poly made to write llama.cpp's answer, then the comparisons, printed and saved as it goes
say(f"== KL(llama.cpp || strata-poly) is over llama.cpp's {TOP} likeliest at each step; the one against an earlier "
    "strata-poly run over this run's")
for k, v in ids.items():
    follow = ref[k]["tokens"]
    if not follow:
        say(f"!! {k}: llama.cpp gave no tokens")
        continue
    t1 = time.time()
    results[k] = ours(v, follow)
    say(f"== {k}: {len(v)} prompt tokens, {len(results[k])} steps in {time.time() - t1:.0f} s (load included)")
    against_llama(k)
    if other is not None:
        against_itself(k)
    save()

if allk:
    allk.sort()
    say(f"== all prompts vs llama.cpp: same most likely token {100.0 * all_agree[0] / max(all_agree[1], 1):.1f}% of "
        f"{all_agree[1]} steps; KL mean {sum(allk) / len(allk):.2e}, median {allk[len(allk) // 2]:.2e}, "
        f"p90 {allk[int(0.9 * (len(allk) - 1))]:.2e}")
if self_k:
    self_k.sort()
    say(f"== all prompts vs strata-poly with options {other.get('ours_args') or '(none)'}: same most likely token "
        f"{100.0 * self_agree[0] / self_agree[1]:.1f}%; KL mean {sum(self_k) / len(self_k):.2e}, "
        f"median {self_k[len(self_k) // 2]:.2e}")
save()
say(f"== results: {report.name}")
