#!/usr/bin/env bash
# deploy-laguna.sh -- run ON ECHO as eohiggins, from an interactive ssh session: Laguna S 2.1 into daily use.
#
#     ssh -t eohiggins@echo 'bash ~/laguna-deploy/deploy-laguna.sh'
#
# Next to it: strata-laguna-ud-q3_k_xl.json and polystrata.service.  It shows what it changes and asks before each step:
#   1. checks: the dev checkout (~/src/PolyStrata) has the Laguna commits and its engine is built from them
#   2. the pack: ~/PolyStrata-data/packs/laguna-ud-q3_k_xl/ with the tokenizer and the expert profile from the tests
#   3. the run config: ~/PolyStrata/strata-laguna-ud-q3_k_xl.json (beside the others, so switch_on_request finds it)
#   4. the service: polystrata.service runs the dev checkout's server.py (the installed models stay as they are)
#   5. a restart, Laguna listed, loaded, one short answer, and Swift loaded again
# Everything it replaces goes to ~/polystrata-config-backup-<time>/ first.
set -euo pipefail
shopt -s nullglob

STAGE="$(cd "$(dirname "$0")" && pwd)"
PS="$HOME/PolyStrata"
DEV="$HOME/src/PolyStrata"
TEST="$HOME/laguna-test"
PACK="$HOME/PolyStrata-data/packs/laguna-ud-q3_k_xl"
GGUF="$HOME/PolyStrata-data/models/laguna-s21-UD-Q3_K_XL/UD-Q3_K_XL/Laguna-S-2.1-UD-Q3_K_XL-00001-of-00003.gguf"
CFG=strata-laguna-ud-q3_k_xl.json
UNIT=/etc/systemd/system/polystrata.service
ENV_FILE=/etc/polystrata/polystrata.env
BACKUP="$HOME/polystrata-config-backup-$(date +%Y%m%d-%H%M%S)"
PORT=8080
NEED=1e83014b                       # the dev checkout has at least this commit of the laguna branch
LAGUNA=laguna-s-2.1-ud-q3_k_xl      # the config's model_name
PY="$PS/.venv/bin/python"

say() { printf '\n== %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
ask() { local r; read -r -p "$1 [y/N] " r; [[ $r == [yY]* ]]; }
backup() { mkdir -p -m 700 "$BACKUP"; cp -p "$@" "$BACKUP"/; }

[[ -t 0 ]] || die "needs a terminal: run it with ssh -t"
[[ $(id -un) == eohiggins ]] || die "run this as eohiggins"
for f in "$STAGE/$CFG" "$STAGE/polystrata.service" "$GGUF" "$TEST/tok/tokenizer/chat_template.jinja" \
         "$DEV/serve/server.py" "$DEV/build-poly/strata-poly" "$PY"; do
    [[ -e $f ]] || die "missing: $f"
done

# ---------------------------------------------------------------------------------------------------------------
say "1/5 Checks"
git -C "$DEV" merge-base --is-ancestor "$NEED" HEAD 2>/dev/null ||
    die "$DEV does not have $NEED: cd $DEV && git pull"
echo "  dev checkout: $(git -C "$DEV" log -1 --format='%h %s')"
newer=$(find "$DEV/src" "$DEV/include" "$DEV/CMakeLists.txt" -newer "$DEV/build-poly/strata-poly" -type f | head -3)
[[ -z $newer ]] || die "the dev engine is older than its sources ($newer): build it first:
  cd $DEV && PATH=\$HOME/PolyStrata/.venv/bin:\$PATH cmake --build build-poly --target strata-poly -j 8"
echo "  dev engine: built after its sources"
if [[ -d $PS/.git ]]; then
    inst=$(git -C "$PS" rev-parse HEAD)
    if git -C "$DEV" merge-base --is-ancestor "$inst" HEAD 2>/dev/null; then
        echo "  installed PolyStrata: $(git -C "$PS" log -1 --format='%h %ad' --date=short); the dev checkout has it and more"
    else
        echo "  WARNING: the installed PolyStrata ($(git -C "$PS" log -1 --format='%h %ad' --date=short)) is not in the dev"
        echo "  checkout's history: it may be newer than the dev server, whose protocol its engines then may not match."
        ask "Go on anyway?" || die "stopped; nothing changed"
    fi
else
    echo "  the installation is not a git checkout (its engines' versions are in their logs)"
fi

# ---------------------------------------------------------------------------------------------------------------
say "2/5 The pack: $PACK"
mkdir -p "$PACK"
if [[ -f $PACK/tokenizer/chat_template.jinja ]] && diff -rq "$TEST/tok/tokenizer" "$PACK/tokenizer" >/dev/null; then
    echo "  tokenizer: there already, the same"
else
    [[ -d $PACK/tokenizer ]] && { backup -r "$PACK/tokenizer"; rm -rf "$PACK/tokenizer"; }
    cp -r "$TEST/tok/tokenizer" "$PACK/tokenizer"
    echo "  tokenizer: copied from $TEST/tok/tokenizer"
fi
if [[ -f $PACK/expert-profile.txt ]]; then
    echo "  expert profile: there already, kept (the engine adds to it after every answer)"
elif [[ -f $TEST/expert-profile.txt ]]; then
    cp -p "$TEST/expert-profile.txt" "$PACK/expert-profile.txt"
    echo "  expert profile: the tests' one, copied (it orders the expert cache from the first load)"
else
    echo "  expert profile: none yet; the engine starts one"
fi

# ---------------------------------------------------------------------------------------------------------------
say "3/5 The run config: $PS/$CFG"
if [[ -f $PS/$CFG ]]; then
    diff -u --label "live/$CFG" --label "new/$CFG" "$PS/$CFG" "$STAGE/$CFG" || true
else
    sed 's/^/  /' "$STAGE/$CFG"
fi
if ask "Install it?"; then
    [[ -f $PS/$CFG ]] && backup "$PS/$CFG"
    cp "$STAGE/$CFG" "$PS/$CFG"
    echo "  installed"
else
    echo "  left as it is"
fi

# ---------------------------------------------------------------------------------------------------------------
say "4/5 The service: $UNIT"
diff -u --label "live/polystrata.service" --label "new/polystrata.service" "$UNIT" "$STAGE/polystrata.service" || true
if ask "Install it (the dev checkout's server.py; the installed models' configs and engines as before)?"; then
    backup "$UNIT"
    sudo install -m 644 "$STAGE/polystrata.service" "$UNIT"
    sudo systemctl daemon-reload
    systemd-analyze verify "$UNIT" 2>&1 | sed 's/^/  verify: /' || true
    echo "  installed; the old unit is in $BACKUP"
else
    echo "  left as it is"
fi

# ---------------------------------------------------------------------------------------------------------------
say "5/5 Restart and check"
ask "Restart polystrata now (what it is doing stops)?" || { echo "  later: sudo systemctl restart polystrata"; exit 0; }
sudo systemctl restart polystrata
key=$(sed -n 's/^STRATA_API_KEY=//p' "$ENV_FILE" | cut -d, -f1)
api() {   # METHOD PATH [JSON]: the body back, with the key from the env file (never on a command line)
    if [[ $# -ge 3 ]]; then
        curl -s --max-time 900 -X "$1" -H @<(printf 'Authorization: Bearer %s\nContent-Type: application/json\n' "$key") \
             --data "$3" "http://127.0.0.1:$PORT$2"
    else
        curl -s --max-time 60 -H @<(printf 'Authorization: Bearer %s\n' "$key") "http://127.0.0.1:$PORT$2"
    fi
}
echo "  waiting for the server (up to 5 minutes) ..."
models=""
for _ in $(seq 1 100); do
    models=$(api GET /v1/models 2>/dev/null || true)
    [[ $models == *'"data"'* ]] && break
    sleep 3
done
[[ $models == *'"data"'* ]] || die "no answer from /v1/models: journalctl -u polystrata -n 80 --no-pager"
ids=$("$PY" -c 'import json,sys; print(" ".join(m["id"] for m in json.load(sys.stdin)["data"]))' <<<"$models")
echo "  /v1/models: $ids"
[[ " $ids " == *" $LAGUNA "* ]] || die "$LAGUNA is not listed: is $PS/$CFG there, and switch_on_request on in the Swift config?"
# the text model loaded now: /v1/models lists it first (the image model, when there is one, comes last)
loaded=$("$PY" -c 'import json,sys; d=json.load(sys.stdin)["data"]; print(d[0]["id"] if d else "?")' <<<"$models" 2>/dev/null || echo "?")
echo "  loaded now: $loaded"
if ask "Load Laguna, ask it one short question, then load $loaded again (a minute or two each way)?"; then
    t0=$(date +%s)
    echo "  load: $(api POST /v1/load "{\"model\": \"$LAGUNA\"}" | head -c 300)  ($(( $(date +%s) - t0 )) s)"
    body='{"model": "'"$LAGUNA"'", "messages": [{"role": "user", "content": "What is the capital of France? One word."}], "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": false}}'
    answer=$(api POST /v1/chat/completions "$body")
    echo "  answer: $("$PY" -c 'import json,sys; print(repr(json.load(sys.stdin)["choices"][0]["message"]["content"]))' <<<"$answer" 2>/dev/null || echo "$answer" | head -c 300)"
    if [[ $loaded != "?" && $loaded != "$LAGUNA" ]]; then
        t0=$(date +%s)
        echo "  back to $loaded: $(api POST /v1/load "{\"model\": \"$loaded\"}" | head -c 200)  ($(( $(date +%s) - t0 )) s)"
    fi
fi
unset key

cat <<EOF

Done.  Laguna is "$LAGUNA": a request that names it loads it (switch_on_request), as for the other models.
  - its log:      $PS/strata-laguna-ud-q3_k_xl.log
  - the server:   journalctl -u polystrata -f
  - after a git pull in $DEV: build the engine (cmake --build build-poly --target strata-poly -j 8),
    then sudo systemctl restart polystrata
  - to go back to the installed server: sudo cp $BACKUP/polystrata.service $UNIT &&
    sudo systemctl daemon-reload && sudo systemctl restart polystrata
EOF
