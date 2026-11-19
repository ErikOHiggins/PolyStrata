#!/usr/bin/env python3
"""add-sampling.py -- the model card's sampling as each strata-poly run config's "sampling" block, so a request that
names no sampling of its own is no longer greedy (strata-poly models get no built-in model-card sampling).

    python3 ~/laguna-deploy/add-sampling.py              # ~/PolyStrata: shows the change, asks, backs up, writes
    python3 ~/laguna-deploy/add-sampling.py --dir DIR --yes

Only the "sampling" key of the listed configs is set; everything else in them stays as it is.  The old files go to
~/polystrata-config-backup-<time>/ first.  Then: sudo systemctl restart polystrata (each model reads its config when
it loads; the server log then says "sampling defaults from the config: ...").

The values (2026-11-16), from each original model's generation_config.json and model card; top_k is capped at 64,
the most the config takes (unset, strata-poly uses 20, where vLLM would use no limit):
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

SAMPLING = {
    # deepseek-ai/DeepSeek-V4-Flash-0731: card "temperature = 1.0, with top_p = 0.95 for agentic scenarios"; no top_k
    "strata-dsv4-ud-iq2_xxs.json": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
    # RecursiveSI/flint: card and generation_config
    "strata-flint-ud-q4_k_xl.json": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
    # google/gemma-4-26B-A4B-it: card "across all use cases"
    "strata-gemma4-qat-ud-q4_k_xl.json": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
    # zai-org/GLM-5.3-Flash: generation_config (and most of the card's evaluations); no top_k
    "strata-glm-3.0bit.json": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
    # Aleph-Alpha/Kolibri-1: card temperature 1.0, top_p 0.97, top_k 128 (capped at 64)
    "strata-kolibri1-q4_k_m.json": {"temperature": 1.0, "top_p": 0.97, "top_k": 64},
    # LiquidAI/LFM2.5-24B-A2B: card and generation_config (the penalty counts over the last 64 tokens)
    "strata-lfm25-q8_0.json": {"temperature": 0.1, "top_k": 50, "repetition_penalty": 1.05},
    # openbmb/MiniCPM-V-4.7-35B-A3B: generation_config (the card's instruct example: 0.7 / 0.95 / 100, rep 1.03)
    "strata-minicpmv-q4_k_m.json": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
    # ornith-ai/Ornith-1.5-35B-A3B: the card's top_p / top_k; temperature 0.8, the user's pick between the card's
    # 0.6 (general tasks) and 1.0 (its benchmarks, and generation_config)
    "strata-ornith-apex-i-quality.json": {"temperature": 0.8, "top_p": 0.95, "top_k": 20},
    # ornith-ai/Ornith-1.5-122B: the card's top_p / top_k; temperature 0.8, the user's pick between the card's
    # 0.6 (precise coding, and generation_config) and 1.0 (general tasks)
    "strata-ornith122-ud-q3_k_xl.json": {"temperature": 0.8, "top_p": 0.95, "top_k": 20},
    # poolside/Laguna-S-2.1: generation_config (temperature 1.0, top_p 1.0, top_k 20, min_p 0); set already when the
    # Laguna configs were written - listed here so that a config without it gets it again
    "strata-laguna-ud-q3_k_xl.json": {"temperature": 1.0, "top_p": 1.0, "top_k": 20, "min_p": 0.0},
    "strata-laguna-ud-q4_k_s.json": {"temperature": 1.0, "top_p": 1.0, "top_k": 20, "min_p": 0.0},
    "strata-laguna-ud-q5_k_s.json": {"temperature": 1.0, "top_p": 1.0, "top_k": 20, "min_p": 0.0},
}


def check(name, s):
    """The server's own limits (sampling_defaults_from_config): a bad value would stop it from starting."""
    for k, v in s.items():
        ok = {"temperature": v >= 0, "top_p": 0 < v <= 1, "min_p": 0 <= v <= 1, "top_k": v == int(v) and 1 <= v <= 64,
              "repetition_penalty": v > 0, "presence_penalty": v >= 0, "frequency_penalty": v >= 0}.get(k)
        if not ok:
            sys.exit(f"{name}: sampling.{k}={v!r} is outside what the server takes")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=str(Path.home() / "PolyStrata"))
    p.add_argument("--yes", action="store_true", help="no question before writing")
    a = p.parse_args()
    d = Path(a.dir).expanduser()
    plan = []
    for name, s in SAMPLING.items():
        check(name, s)
        f = d / name
        if not f.exists():
            print(f"  {name}: not in {d}, skipped")
            continue
        c = json.loads(f.read_text(encoding="utf-8"))
        old = c.get("sampling")
        if old == s:
            print(f"  {name}: already {s}")
            continue
        print(f"  {name}: {old or '(none: greedy unless the request says)'}  ->  {s}")
        plan.append((f, c, s))
    if not plan:
        print("nothing to change")
        return
    if not a.yes and input(f"write these {len(plan)} configs (a backup first)? [y/N] ").strip().lower()[:1] != "y":
        print("nothing written")
        return
    backup = Path.home() / f"polystrata-config-backup-{time.strftime('%Y%m%d-%H%M%S')}"
    backup.mkdir(mode=0o700)
    for f, c, s in plan:
        shutil.copy2(f, backup / f.name)
        c["sampling"] = s
        tmp = f.with_name(f.name + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as out:   # LF on every OS, as the configs are
            out.write(json.dumps(c, indent=1) + "\n")
        json.loads(tmp.read_text(encoding="utf-8"))          # it reads back
        tmp.replace(f)
    print(f"{len(plan)} written; the old ones are in {backup}")
    print("next: sudo systemctl restart polystrata")


if __name__ == "__main__":
    main()
