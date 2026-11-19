#!/usr/bin/env bash
# download-laguna.sh -- run ON ECHO as eohiggins. Downloads Unsloth's Laguna-S-2.1 GGUFs from the pinned revision
# the handoff uses (750f92f9): by default UD-Q4_K_S, then UD-Q5_K_S; UD-Q3_K_XL and UD-Q4_K_XL are listed too.
# Partial files are resumed, and every shard's SHA-256 is checked against the one Hugging Face lists for it. Safe to
# run again: done shards are skipped, broken ones are fetched again. A shard lands in
# models/laguna-s21-<QUANT>/<QUANT>/, the layout `hf download --local-dir` gave UD-Q3_K_XL.
#
# In tmux (detach with Ctrl-b d, come back with: tmux attach -t laguna-dl):
#   tmux new -s laguna-dl
#   bash ~/laguna-download/download-laguna.sh 2>&1 | tee -a ~/laguna-download.log
# Other quants:   QUANTS="UD-Q4_K_XL" bash ~/laguna-download/download-laguna.sh
set -euo pipefail

REV=750f92f90cf54159c4d7a610cb7b3e74498e75c6
BASE="https://huggingface.co/unsloth/Laguna-S-2.1-GGUF/resolve/$REV"
DEST="${DEST:-/home/eohiggins/PolyStrata-data/models}"
QUANTS="${QUANTS:-UD-Q4_K_S UD-Q5_K_S}"

# quant  shard  bytes  sha256 (Hugging Face's LFS ids at 750f92f9)
SHARDS=(
"UD-Q3_K_XL 1 3683648     39de6a5ff3b9813b923e1bf8684d8ff6dc786d6337f75fa245f21d1e6f2c586b"
"UD-Q3_K_XL 2 49699878368 0e8bc5a30c755c5faac6e6cdef3cc8c8c2425bc7468e31ecb282bbbbf6558c19"
"UD-Q3_K_XL 3 4390258592  25c8d2c8e9e1c6c059b1b7dfbeaa02e29caeadc85974c4971dea5e7878665d2d"
"UD-Q4_K_XL 1 3683648     0cfaf46917260d253773e5e2fab64329fa5c9c60fdf0db0f59f31205b5f5dd32"
"UD-Q4_K_XL 2 49971821312 2296102462b02edca70163121ac62bacf7a82078c0eafc91625c8822850769bf"
"UD-Q4_K_XL 3 23419667040 9150e2338f7690af29685b6a2ca621a8fda7ecf9724678266c4b04b7c6dd0ef3"
"UD-Q4_K_S 1 3683648     3e8d20d4aab3497d32568010b9b4b29fc0143d4d5e716ab958264e5a083301fb"
"UD-Q4_K_S 2 49587832416 c3a48e4dd559ac20bd1c0e78ac6064aa95fd35b38d981f60ed6653093a0934ab"
"UD-Q4_K_S 3 18997819168 2354ad27a19ea411cc85e1982e2209c21c1e672d6d397c8c76d77fb01419d2c5"
"UD-Q5_K_S 1 3683648     426ec039077f7f9c0ce8e44551b2037b6a3ab3f1f5d33a2ad16d2197f47a6673"
"UD-Q5_K_S 2 49502923552 f026ec1e20133d3ad81de936cc0a1c9a85b899f8e53d3c1c72310d04f8cdbce1"
"UD-Q5_K_S 3 33149588064 11192189e91501aeb4aacea43fc77e0037d4705452f0607b5a00a24409bb3658"
)

log() { printf '%s  %s\n' "$(date '+%F %T')" "$*"; }
size_of() { stat -c %s "$1" 2>/dev/null || echo 0; }
# a line every 10 minutes while a shard comes in: how much is there, and the pace since the last line
progress() {
    local f=$1 total=$2 prev now
    prev=$(size_of "$f")
    while sleep 600; do
        now=$(size_of "$f")
        log "  ... $(( now / 1024**2 )) of $(( total / 1024**2 )) MiB, $(( (now - prev) / 600 / 1024 )) KiB/s"
        prev=$now
    done
}
PROGRESS=""
trap '[[ -n $PROGRESS ]] && kill "$PROGRESS" 2>/dev/null; true' EXIT

# --- room on the disk: what is still to come, plus 5 GiB
need=0
for row in "${SHARDS[@]}"; do
    read -r q i bytes sha <<<"$row"
    [[ " $QUANTS " == *" $q "* ]] || continue
    f="$DEST/laguna-s21-$q/$q/Laguna-S-2.1-$q-0000$i-of-00003.gguf"
    [[ -f $f.sha256-ok ]] && continue
    have=$(( $(size_of "$f") + $(size_of "$f.part") ))
    need=$(( need + bytes - have ))
done
mkdir -p "$DEST"
avail=$(df -B1 --output=avail "$DEST" | tail -1)
log "to download: $(( need / 1024**3 )) GiB; free on $(df --output=target "$DEST" | tail -1): $(( avail / 1024**3 )) GiB"
(( avail > need + 5 * 1024**3 )) || { log "ERROR: not enough free space (need $(( need / 1024**3 )) GiB + 5)"; exit 1; }

# --- the shards
for row in "${SHARDS[@]}"; do
    read -r q i bytes sha <<<"$row"
    [[ " $QUANTS " == *" $q "* ]] || continue
    dir="$DEST/laguna-s21-$q/$q"; name="Laguna-S-2.1-$q-0000$i-of-00003.gguf"; f="$dir/$name"
    mkdir -p "$dir"
    if [[ -f $f.sha256-ok ]]; then log "ok (checked before): $q/$name"; continue; fi
    if [[ ! -f $f ]]; then
        log "downloading $q/$name ($(( bytes / 1024**2 )) MiB; $(( $(size_of "$f.part") / 1024**2 )) MiB there already) ..."
        t0=$(date +%s)
        progress "$f.part" "$bytes" &
        PROGRESS=$!
        for attempt in $(seq 1 50); do                    # each try resumes from what .part holds now
            [[ $(size_of "$f.part") == "$bytes" ]] && break
            curl -L --fail --no-progress-meter -C - -o "$f.part" "$BASE/$q/$name" && break
            log "  try $attempt stopped at $(( $(size_of "$f.part") / 1024**2 )) MiB; again in 30 s"
            sleep 30
        done
        kill "$PROGRESS" 2>/dev/null || true
        PROGRESS=""
        dt=$(( $(date +%s) - t0 )); (( dt > 0 )) || dt=1
        log "  got $(( $(size_of "$f.part") / 1024**2 )) MiB in ${dt}s ($(( $(size_of "$f.part") / 1024**2 / dt )) MiB/s)"
        [[ $(size_of "$f.part") == "$bytes" ]] || { log "ERROR: $name is $(size_of "$f.part") bytes, not $bytes: run again to resume"; exit 1; }
        mv "$f.part" "$f"
    fi
    log "checking SHA-256 of $q/$name ..."
    got=$(sha256sum "$f" | cut -d' ' -f1)
    if [[ $got == "$sha" ]]; then
        touch "$f.sha256-ok"; log "  ok"
    else
        log "ERROR: $name: SHA-256 $got, expected $sha - removed; run again to fetch it anew"
        rm -f "$f"; exit 1
    fi
done
log "done:"
for q in $QUANTS; do ls -l --block-size=M "$DEST/laguna-s21-$q/$q/"*.gguf | sed 's/^/    /'; done
