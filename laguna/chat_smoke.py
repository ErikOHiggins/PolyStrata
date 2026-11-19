#!/usr/bin/env python3
"""laguna/chat_smoke.py -- run ON ECHO: Laguna through PolyStrata's own server (the dev checkout), on the chat API.

    sudo systemctl stop polystrata
    ~/PolyStrata/.venv/bin/python ~/laguna-test/chat_smoke.py [--kv q8_0] [--context 65536]
    sudo systemctl start polystrata

It writes ~/laguna-test/strata-laguna-ud-q3_k_xl.json (the dev engine, the tokenizer smoke.py extracted, on both
cards), starts ~/src/PolyStrata/serve/server.py with it on 127.0.0.1:8081, and checks over /v1/chat/completions:

  1. thinking on:  the reasoning comes back as reasoning_content, the answer as content
  2. thinking off: chat_template_kwargs {"enable_thinking": false}, no reasoning
  3. a tool call:  the model's GLM-style call comes back as tool_calls with typed arguments
  4. the tool's answer sent back: the reply uses it
  5. streaming:    the same as 1, as server-sent events
  6. a second turn of 1's conversation: how much of the prompt the engine did not read again

Everything goes to ~/laguna-test/: chat-<time>.txt, the server's output (server.log) and the engine's (the config's
"log").  The server is stopped at the end, whatever happens.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

HOME = pathlib.Path.home()
p = argparse.ArgumentParser()
p.add_argument("--src", default=str(HOME / "src/PolyStrata"))
p.add_argument("--gguf", default=str(HOME / "PolyStrata-data/models/laguna-s21-UD-Q3_K_XL/UD-Q3_K_XL/"
                                            "Laguna-S-2.1-UD-Q3_K_XL-00001-of-00003.gguf"))
p.add_argument("--out", default=str(HOME / "laguna-test"))
p.add_argument("--port", type=int, default=8081)
p.add_argument("--context", type=int, default=65536)
p.add_argument("--kv", default="", help="fp16 (default) or q8_0")
a = p.parse_args()

src, out, gguf = (pathlib.Path(x).expanduser() for x in (a.src, a.out, a.gguf))
stamp = time.strftime("%Y%m%d-%H%M%S")
report = open(out / f"chat-{stamp}.txt", "w", encoding="utf-8")
failed = []


def say(*parts):
    line = " ".join(str(x) for x in parts)
    print(line, flush=True)
    report.write(line + "\n")
    report.flush()


def check(ok: bool, what: str):
    say(("   ok   " if ok else "   FAIL ") + what)
    if not ok:
        failed.append(what)


if subprocess.run(["systemctl", "is-active", "--quiet", "polystrata"]).returncode == 0:
    sys.exit("polystrata.service holds the cards: sudo systemctl stop polystrata first, and start it again after")
tok = out / "tok/tokenizer"
for need in (src / "serve/server.py", src / "build-poly/strata-poly", gguf, tok / "chat_template.jinja"):
    if not need.exists():
        sys.exit(f"missing: {need}" + ("  (run smoke.py first: it extracts the tokenizer)" if "tok" in str(need) else ""))
say("== dev checkout:", subprocess.run(["git", "-C", str(src), "log", "-1", "--format=%h %s"], capture_output=True,
                                       text=True).stdout.strip())

# the run config, as setup writes one for a strata-poly model (but on 127.0.0.1 and with no key: a test)
cfg_path = out / "strata-laguna-ud-q3_k_xl.json"
cfg = {
    "exe": str(src / "build-poly/strata-poly"),
    "args": ["--model", str(gguf), "--max-context", str(a.context), "--expert-profile",
             str(out / "expert-profile.txt"), "--conversation-cache-mib", "auto"] + (["--kv", a.kv] if a.kv else []),
    "cwd": str(out),
    "tokenizer": str(tok),
    "model_name": "laguna-s-2.1-ud-q3_k_xl",
    "display_name": "Laguna S 2.1 UD-Q3_K_XL (dev)",
    "log": str(out / "strata-laguna-ud-q3_k_xl.log"),
    "lib_dirs": ["/usr/local/cuda-13.4/bin", "/usr/local/cuda-13.4/lib64"],
    "port": a.port,
    "gpu": [0, 1],
    "gpus_asked": True,
    "layer_split": "auto",
    "host": "127.0.0.1",
}
cfg_path.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
say("== config:", cfg_path)

python = sys.executable
cmd = [python, str(src / "serve/server.py"), "--engine", "strata", "--config", str(cfg_path), "--port", str(a.port)]
say("== server:", " ".join(cmd))
slog = open(out / "server.log", "w", encoding="utf-8")
srv = subprocess.Popen(cmd, cwd=str(src), stdout=slog, stderr=subprocess.STDOUT)
base = f"http://127.0.0.1:{a.port}"


def post(path: str, body: dict, timeout=1800):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(messages, **extra):
    body = {"model": "laguna-s-2.1-ud-q3_k_xl", "messages": messages, "temperature": 0, "max_tokens": 2048, **extra}
    t0 = time.time()
    res = post("/v1/chat/completions", body)
    msg = res["choices"][0]["message"]
    u = res.get("usage") or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    say(f"   {time.time() - t0:.1f} s; prompt {u.get('prompt_tokens')} tokens ({cached} cached), "
        f"{u.get('completion_tokens')} written; finish {res['choices'][0].get('finish_reason')}")
    return msg, res


try:
    t0 = time.time()
    while True:
        if srv.poll() is not None:
            raise RuntimeError(f"the server stopped (code {srv.returncode}); see {out / 'server.log'}")
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as r:
                if r.status == 200:
                    break
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        except urllib.error.HTTPError:
            pass
        if time.time() - t0 > 900:
            raise RuntimeError("the server did not answer /health in 15 minutes")
        time.sleep(3)
    say(f"== up in {time.time() - t0:.0f} s")
    models = json.loads(urllib.request.urlopen(base + "/v1/models", timeout=30).read())
    say("   /v1/models:", [m.get("id") for m in models.get("data", [])])

    q = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
    say("== 1. thinking on")
    m1, _ = chat(q)
    say("   reasoning:", repr((m1.get("reasoning_content") or "")[:200]))
    say("   content:  ", repr(m1.get("content")))
    check(bool((m1.get("reasoning_content") or "").strip()), "the reasoning comes back as reasoning_content")
    check("paris" in (m1.get("content") or "").lower(), "the answer is Paris, in content")
    check("<think>" not in (m1.get("content") or "") and "</think>" not in (m1.get("content") or ""),
          "no thinking tags in content")

    say("== 2. thinking off")
    m2, _ = chat(q, chat_template_kwargs={"enable_thinking": False})
    say("   reasoning:", repr(m2.get("reasoning_content")), " content:", repr(m2.get("content")))
    check(not (m2.get("reasoning_content") or "").strip(), "no reasoning")
    check("paris" in (m2.get("content") or "").lower(), "the answer is Paris")

    say("== 3. a tool call")
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "The weather forecast for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "the city"},
                                                        "days": {"type": "integer", "description": "days ahead"}},
                       "required": ["city"]}}}]
    q3 = [{"role": "user", "content": "What will the weather be in Lyon for the next 3 days? Use the tool."}]
    m3, r3 = chat(q3, tools=tools)
    calls = m3.get("tool_calls") or []
    say("   reasoning:", repr((m3.get("reasoning_content") or "")[:160]))
    say("   content:  ", repr(m3.get("content")), " tool_calls:", json.dumps(calls)[:300])
    check(len(calls) >= 1 and calls[0]["function"]["name"] == "get_weather", "a get_weather call in tool_calls")
    args = json.loads(calls[0]["function"]["arguments"]) if calls else {}
    check(str(args.get("city", "")).lower() == "lyon", "its city is Lyon")
    check(args.get("days") in (3, None), "days is the integer 3 (or left out)")
    check(r3["choices"][0].get("finish_reason") == "tool_calls", "finish_reason is tool_calls")

    say("== 4. the tool's answer")
    if calls:
        q4 = q3 + [{"role": "assistant", "content": m3.get("content") or "", "tool_calls": calls,
                    **({"reasoning_content": m3["reasoning_content"]} if m3.get("reasoning_content") else {})},
                   {"role": "tool", "tool_call_id": calls[0].get("id", "call_0"),
                    "content": json.dumps({"city": "Lyon", "forecast": ["rain, 14C", "cloudy, 16C", "sunny, 19C"]})}]
        m4, _ = chat(q4, tools=tools)
        say("   content:", repr((m4.get("content") or "")[:300]))
        text = (m4.get("content") or "").lower()
        check(any(w in text for w in ("rain", "14", "19", "sunny")), "the reply uses the forecast")
    else:
        check(False, "skipped: no call in 3")

    say("== 5. streaming")
    body = {"model": "laguna-s-2.1-ud-q3_k_xl", "messages": q, "temperature": 0, "max_tokens": 2048, "stream": True}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    chunks, reasoning, content, finish = 0, "", "", None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            for ch in ev.get("choices", []):
                d = ch.get("delta") or {}
                reasoning += d.get("reasoning_content") or ""
                content += d.get("content") or ""
                finish = ch.get("finish_reason") or finish
            chunks += 1
    say(f"   {chunks} chunks; reasoning {len(reasoning)} chars; content {content!r}; finish {finish}")
    check(chunks > 2 and "paris" in content.lower() and bool(reasoning.strip()), "streamed reasoning, then Paris")

    say("== 6. a second turn of 1's conversation")
    q6 = q + [{"role": "assistant", "content": m1.get("content") or "",
               **({"reasoning_content": m1["reasoning_content"]} if m1.get("reasoning_content") else {})},
              {"role": "user", "content": "And of Italy? One word."}]
    m6, r6 = chat(q6)
    say("   content:", repr(m6.get("content")))
    check("rome" in (m6.get("content") or "").lower(), "the answer is Rome")
finally:
    srv.terminate()
    try:
        srv.wait(timeout=90)
    except subprocess.TimeoutExpired:
        srv.kill()
    slog.close()

log_text = pathlib.Path(cfg["log"]).read_text(encoding="utf-8", errors="replace") if pathlib.Path(cfg["log"]).exists() else ""
say("== from the engine's log:")
for line in log_text.splitlines():
    if any(w in line for w in ("laguna:", "loaded in", "strata serve: prompt", "WARNING", "rror")):
        say("  ", line[:220])
say(f"== {'all checks passed' if not failed else f'{len(failed)} FAILED: ' + '; '.join(failed)}")
say(f"== results: {report.name}")
