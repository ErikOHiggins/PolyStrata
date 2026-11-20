"""tools/llama_server_check.py - a llama-server model through Puddle against llama-server alone, greedy, on one
question with a picture (or none).

Puddle renders the chat template, tokenizes the text and sends llama-server the prompt (as token ids, or as text with
llama-server's media marker when there is a picture); llama-server alone does all of that itself through its own
/v1/chat/completions.  With the same file, the same arguments and greedy decoding the two should write the same
answer, and count the same prompt tokens (Puddle plans a picture's tokens by serve/media_tokens.py).

The model must not be running: the check starts Puddle with the run config (on 127.0.0.1, its own port), asks, stops
it, then starts llama-server with the config's own command and environment (the cards in the same order), asks,
stops it, and compares.  One model in VRAM at a time.

    python tools/llama_server_check.py --config strata-nemotron-bolt-24gb.json --image picture.jpg
        [--prompt "What does this picture show? Two sentences."] [--max-tokens 600]

Exit code 0: the same answer; 1: a difference (the first one is printed); 2: something did not start or answer.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(url: str, body: dict, key: str | None = None, timeout: float = 900.0) -> dict:
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_up(url: str, proc: subprocess.Popen, what: str, timeout: float = 900.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if proc.poll() is not None:
            raise RuntimeError(f"{what} exited before it answered (code {proc.returncode})")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1.0)
    raise RuntimeError(f"{what} did not answer {url} within {timeout:.0f} s")


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)


def first_difference(a: str, b: str, around: int = 40) -> str | None:
    """None when equal, else where they part, with some text of each."""
    if a == b:
        return None
    i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return (f"at character {i}:\n    Puddle:       {a[max(0, i - around):i + around]!r}\n"
            f"    llama-server: {b[max(0, i - around):i + around]!r}")


def answer(reply: dict) -> tuple[str, str, int | None]:
    msg = (reply.get("choices") or [{}])[0].get("message") or {}
    return (msg.get("reasoning_content") or ""), (msg.get("content") or ""), (reply.get("usage") or {}).get(
        "prompt_tokens")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, help="the model's run config (\"engine\": \"llama-server\")")
    ap.add_argument("--image", help="a picture to ask about (leave out for text alone)")
    ap.add_argument("--prompt", default="What does this picture show? Two sentences.")
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--log", default="llama_server_check.log", help="both servers' output (default: %(default)s)")
    a = ap.parse_args(argv)
    cfg_path = Path(a.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    if cfg.get("engine") != "llama-server":
        print(f"{cfg_path.name} is not a llama-server model's run config")
        return 2
    content: list | str = a.prompt
    if a.image:
        data = Path(a.image).read_bytes()
        kind = "png" if data[:4] == b"\x89PNG" else "jpeg"
        url = f"data:image/{kind};base64," + base64.b64encode(data).decode()
        content = [{"type": "text", "text": a.prompt}, {"type": "image_url", "image_url": {"url": url}}]
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": a.max_tokens, "temperature": 0,
            "stream": False}
    log = open(a.log, "w", encoding="utf-8")

    # 1. through Puddle (the run config's own settings; greedy by the request)
    port = free_port()
    puddle = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config",
                               str(cfg_path), "--host", "127.0.0.1", "--port", str(port)],
                              cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                              env={k: v for k, v in os.environ.items() if k != "STRATA_API_KEY"})
    try:
        print("starting Puddle with the model ...", flush=True)
        wait_up(f"http://127.0.0.1:{port}/health", puddle, "Puddle")
        key = str(cfg.get("api_key") or "").split(",")[0].strip() or None   # the config's own key, if it has one
        mine = post(f"http://127.0.0.1:{port}/v1/chat/completions", body, key)
    except (RuntimeError, OSError, urllib.error.URLError, ValueError) as e:
        print(f"Puddle: {e} (its output: {a.log})")
        return 2
    finally:
        stop(puddle)
    time.sleep(5)                                       # the cards' memory comes back with the processes

    # 2. llama-server alone: the config's command and environment (serve/server.py's child_env: the same cards, in
    # the same order, and the same libraries)
    from serve import server as S
    env = S.child_env(cfg)
    port = free_port()
    llama = subprocess.Popen([S.engine_exe(cfg), *[str(x) for x in cfg["args"]], "--host", "127.0.0.1", "--port",
                              str(port), "--no-ui"], cwd=cfg.get("cwd") or str(ROOT), stdout=log,
                             stderr=subprocess.STDOUT, env=env)
    try:
        print("starting llama-server alone ...", flush=True)
        wait_up(f"http://127.0.0.1:{port}/health", llama, "llama-server")
        theirs = post(f"http://127.0.0.1:{port}/v1/chat/completions", body)
    except (RuntimeError, OSError, urllib.error.URLError, ValueError) as e:
        print(f"llama-server: {e} (its output: {a.log})")
        return 2
    finally:
        stop(llama)
        log.close()

    r1, c1, n1 = answer(mine)
    r2, c2, n2 = answer(theirs)
    print(f"prompt tokens: Puddle planned {n1}, llama-server counted {n2}" +
          (" (the same)" if n1 == n2 else f" (off by {n1 - n2})" if n1 is not None and n2 is not None else ""))
    same = True
    for what, x, y in (("thinking", r1, r2), ("answer", c1, c2)):
        d = first_difference(x, y)
        if d is None:
            print(f"{what}: the same ({len(x)} characters)")
        else:
            same = False
            print(f"{what}: differs {d}")
    if not same and first_difference(r1 + c1, r2 + c2) is None:
        print("(the whole text is the same: only where the thinking ends differs)")
    print("\nPuddle's answer:\n" + c1)
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
