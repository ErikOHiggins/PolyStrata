#!/usr/bin/env bash
# compare-quants.sh -- run ON ECHO, in tmux: Laguna S 2.1 UD-Q3_K_XL, UD-Q4_K_S and UD-Q5_K_S side by side, on the
# strata-poly of ~/src/PolyStrata (the laguna branch the service runs).
#
#     bash ~/laguna-deploy/compare-quants.sh                # refs, quality, speed: about 2 hours, polystrata stopped
#     bash ~/laguna-deploy/compare-quants.sh speed          # only the stages named
#     bash ~/laguna-deploy/compare-quants.sh install        # the two run configs into ~/PolyStrata (not in the default)
#
# Stages:
#   refs     llama.cpp's answers for UD-Q5_K_S and UD-Q4_K_S, each on its own file, and strata-poly made to follow
#            them: does strata-poly run each new file as llama.cpp does (PR 1 showed it for UD-Q3_K_XL)
#   quality  UD-Q4_K_S and UD-Q3_K_XL on strata-poly, made to follow UD-Q5_K_S's llama.cpp answers: how far each
#            moves from the largest quant here.  UD-Q5_K_S on strata-poly against the same answers (refs) is the floor
#   speed    longctx.py, --kv q8_0, 128 tokens written at 2K to 196K: UD-Q4_K_S, UD-Q5_K_S, then UD-Q3_K_XL again
#            the same day, about 25 minutes each
#   install  strata-laguna-ud-q4_k_s.json and -q5_k_s.json into ~/PolyStrata (they share UD-Q3_K_XL's pack: the same
#            tokenizer and router), then a restart of polystrata: they load by name like UD-Q3_K_XL
# All six parity prompts (6 to about 4,000 tokens, 64 steps each), fp16 KV for parity.  Results: ~/laguna-test/
# quants-<time>/, summary.txt first.  A stage that fails is recorded and the next one runs.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TEST="$HOME/laguna-test"
OUT="$TEST/quants-$(date +%Y%m%d-%H%M%S)"
PY="$HOME/PolyStrata/.venv/bin/python"
LLAMA_BIN="$HOME/llama.cpp/build/bin"
MODELS="$HOME/PolyStrata-data/models"
ALL=(refs quality speed install)
DEFAULT=(refs quality speed)
STAGES=("$@")
[[ ${#STAGES[@]} -gt 0 ]] || STAGES=("${DEFAULT[@]}")
gguf() { echo "$MODELS/laguna-s21-$1/$1/Laguna-S-2.1-$1-00001-of-00003.gguf"; }

mkdir -p "$OUT"
SUM="$OUT/summary.txt"
note() { printf '%s\n' "$*" | tee -a "$SUM"; }
say() { printf '\n== %s  (%s)\n' "$*" "$(date +%H:%M)" | tee -a "$SUM"; }
want() { local s; for s in "${STAGES[@]}"; do [[ $s == "$1" ]] && return 0; done; return 1; }
for s in "${STAGES[@]}"; do
    [[ " ${ALL[*]} " == *" $s "* ]] || { echo "no stage '$s'; the stages: ${ALL[*]}"; exit 2; }
done
[[ -n ${TMUX:-} ]] || echo "(not in tmux: a dropped ssh session would end the run)"

for q in UD-Q3_K_XL UD-Q4_K_S UD-Q5_K_S; do [[ -f $(gguf $q) ]] || { echo "missing: $(gguf $q)"; exit 1; }; done
for q in UD-Q4_K_S UD-Q5_K_S; do
    for i in 1 2 3; do
        f="$MODELS/laguna-s21-$q/$q/Laguna-S-2.1-$q-0000$i-of-00003.gguf"
        [[ -f $f.sha256-ok ]] || { echo "not checked yet: $f (run download-laguna.sh again)"; exit 1; }
    done
done
for f in "$TEST/parity.py" "$TEST/longctx.py" "$LLAMA_BIN/llama-server" "$HOME/src/PolyStrata/build-poly/strata-poly"; do
    [[ -e $f ]] || { echo "missing: $f"; exit 1; }
done

sudo -v || exit 1
( while sleep 120; do sudo -n true 2>/dev/null || exit; done ) &
KEEP=$!
STOPPED=0
finish() {
    kill "$KEEP" 2>/dev/null
    if [[ $STOPPED == 1 ]]; then sudo systemctl start polystrata && echo "polystrata started again"; fi
}
trap finish EXIT
stop_service() {
    if [[ $STOPPED == 0 ]] && systemctl is-active -q polystrata; then
        sudo systemctl stop polystrata && STOPPED=1 && note "polystrata stopped for the GPU stages"
    fi
}

note "compare-quants.sh $(date '+%Y-%m-%d %H:%M'), stages: ${STAGES[*]}"
note "strata-poly: $(git -C "$HOME/src/PolyStrata" log -1 --format='%h %s')"
note "llama.cpp: $("$LLAMA_BIN/llama-server" --version 2>&1 | grep -m1 -i version)"

parity_summary() {   # FILE: the per-prompt lines and the total
    grep -E "^== (short|code|long|doc[0-9]+):|vs llama.cpp: [0-9]+ of|^== all prompts vs llama.cpp" "$1" |
        sed -E 's/; first differs at step ([0-9]+):.*\((near tie|NOT a near tie)\)/; first differs at step \1 (\2)/' |
        cut -c1-170 | sed 's/^/    /' | tee -a "$SUM"
}

# ---------------------------------------------------------------------------------------------------------------
if want refs; then
    say "refs: each new file on llama.cpp, and strata-poly made to follow it"
    stop_service
    for q in UD-Q5_K_S UD-Q4_K_S; do
        t0=$(date +%s)
        "$PY" "$TEST/parity.py" --llama "$LLAMA_BIN" --gguf "$(gguf $q)" > "$OUT/refs-$q.txt" 2>&1
        rc=$?
        txt=$(sed -n 's/^== results: //p' "$OUT/refs-$q.txt" | tail -1)
        [[ -n $txt ]] && echo "${txt%.txt}.json" > "$OUT/ref-$q.path"
        note "  $q (exit $rc, $(( ($(date +%s) - t0) / 60 )) min), llama.cpp's answers in ${txt%.txt}.json:"
        parity_summary "$OUT/refs-$q.txt"
    done
fi

# ---------------------------------------------------------------------------------------------------------------
if want quality; then
    say "quality: strata-poly made to follow UD-Q5_K_S's llama.cpp answers"
    REF5="${REF5:-$(cat "$OUT/ref-UD-Q5_K_S.path" 2>/dev/null)}"
    if [[ -z $REF5 || ! -f $REF5 ]]; then
        note "  SKIPPED: no UD-Q5_K_S reference (run the refs stage, or REF5=<its parity-*.json>)"
    else
        stop_service
        note "  (\"vs llama.cpp\" below is against UD-Q5_K_S on llama.cpp; UD-Q5_K_S's own line is under refs)"
        for q in UD-Q4_K_S UD-Q3_K_XL; do
            "$PY" "$TEST/parity.py" --reuse-ref "$REF5" --gguf "$(gguf $q)" > "$OUT/quality-$q.txt" 2>&1
            note "  $q (exit $?):"
            parity_summary "$OUT/quality-$q.txt"
        done
    fi
fi

# ---------------------------------------------------------------------------------------------------------------
if want speed; then
    say "speed: longctx, q8_0 (each about 25 minutes)"
    stop_service
    for q in UD-Q4_K_S UD-Q5_K_S UD-Q3_K_XL; do
        "$PY" "$TEST/longctx.py" --gguf "$(gguf $q)" --kv q8_0 > "$OUT/speed-$q.txt" 2>&1
        note "  $q (exit $?):"
        grep -E "^ +[0-9,]+ +[0-9,]+ +[0-9]+ +[0-9.]+ |^ +q8_0: 2K|experts in VRAM|routed experts in VRAM" "$OUT/speed-$q.txt" |
            cut -c1-150 | sed 's/^/  /' | tee -a "$SUM"
    done
fi

# ---------------------------------------------------------------------------------------------------------------
if want install; then
    say "install: the two run configs into ~/PolyStrata"
    for q in ud-q4_k_s ud-q5_k_s; do
        cfg="strata-laguna-$q.json"
        [[ -f $HERE/$cfg ]] || { note "  missing: $HERE/$cfg"; continue; }
        if [[ -f $HOME/PolyStrata/$cfg ]] && ! cmp -s "$HERE/$cfg" "$HOME/PolyStrata/$cfg"; then
            cp -p "$HOME/PolyStrata/$cfg" "$OUT/$cfg.before" && note "  the old $cfg kept as $OUT/$cfg.before"
        fi
        cp "$HERE/$cfg" "$HOME/PolyStrata/$cfg" && note "  $cfg installed"
    done
    if [[ $STOPPED == 1 ]]; then sudo systemctl start polystrata && STOPPED=0
    else sudo systemctl restart polystrata; fi
    note "  polystrata (re)started; /v1/models should now list laguna-s-2.1-ud-q4_k_s and laguna-s-2.1-ud-q5_k_s"
fi

say "done: $OUT"
echo "Send back: $SUM"
