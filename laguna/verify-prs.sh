#!/usr/bin/env bash
# verify-prs.sh -- run ON ECHO, in tmux: the evidence for the three Laguna pull requests.
#
#     bash ~/laguna-deploy/verify-prs.sh                 # every stage, about 2 to 3 hours unattended
#     bash ~/laguna-deploy/verify-prs.sh parity sass     # only the stages named
#
# Stages, in this order (the GPU ones first, with polystrata stopped; it is started again after "daily"):
#   setup    worktrees in ~/src/pr/{base,engine,decode,serve} at 67dde645 and the fork's pr/laguna-* branches
#   cuda     CUDA builds of base, engine (PR 1) and decode (PR 3)
#   tests    poly_attend_test and poly_sink_attend_test on all three (their output must not change), and
#            poly_laguna_attend_test on decode
#   sass     every kernel of base is in engine with the same machine code, and every kernel of engine in decode
#   parity   against llama.cpp's saved answers: engine and decode, fp16 and q8_0
#   longctx  decode speed by depth to 196K, q8_0: engine, then decode (same day, same machine)
#   daily    ~/src/PolyStrata to the fork's new laguna branch (PR 1 + 2 + 3), rebuilt; polystrata started again
#   serve    every server test on the PR 2 branch (Linux)
#   hip      HIP builds (gfx1100) of engine and decode in Docker (rocm/dev-ubuntu-24.04:7.2.4-complete)
#   gfx906   the gfx906 build of engine and decode in Docker (mixa3607/rocm-gfx906:7.14-complete)
# Results: ~/laguna-test/verify-<time>/, summary.txt first.  A stage that fails is recorded and the next one runs.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HOME/src/PolyStrata"
WT="$HOME/src/pr"
BASE=67dde645
TEST="$HOME/laguna-test"
OUT="$TEST/verify-$(date +%Y%m%d-%H%M%S)"
PY="$HOME/PolyStrata/.venv/bin/python"
LLAMA="$HOME/PolyStrata/third_party/llama.cpp"
NVCC=/usr/local/cuda-13.4/bin/nvcc
REF="$TEST/parity-20261114-065039.json"
HIP_IMAGE=rocm/dev-ubuntu-24.04:7.2.4-complete
GFX906_IMAGE=mixa3607/rocm-gfx906:7.14-complete
ALL=(setup cuda tests sass parity longctx daily serve hip gfx906)
STAGES=("$@")
[[ ${#STAGES[@]} -gt 0 ]] || STAGES=("${ALL[@]}")
export PATH="$HOME/PolyStrata/.venv/bin:/usr/local/cuda-13.4/bin:$PATH"

mkdir -p "$OUT"
SUM="$OUT/summary.txt"
note() { printf '%s\n' "$*" | tee -a "$SUM"; }
say() { printf '\n== %s  (%s)\n' "$*" "$(date +%H:%M)" | tee -a "$SUM"; }
want() { local s; for s in "${STAGES[@]}"; do [[ $s == "$1" ]] && return 0; done; return 1; }
for s in "${STAGES[@]}"; do
    [[ " ${ALL[*]} " == *" $s "* ]] || { echo "no stage '$s'; the stages: ${ALL[*]}"; exit 2; }
done
[[ -n ${TMUX:-} ]] || echo "(not in tmux: a dropped ssh session would end the run)"

# sudo once, kept fresh for the hours this runs (stopping and starting polystrata, docker if it needs sudo)
sudo -v || exit 1
( while sleep 120; do sudo -n true 2>/dev/null || exit; done ) &
KEEP=$!
STOPPED=0
finish() {
    kill "$KEEP" 2>/dev/null
    if [[ $STOPPED == 1 ]]; then sudo systemctl start polystrata && echo "polystrata started again"; fi
}
trap finish EXIT

note "verify-prs.sh $(date '+%Y-%m-%d %H:%M'), stages: ${STAGES[*]}"
note "machine: $(hostname), $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | paste -sd ';'), $(lscpu | sed -n 's/^Model name: *//p'), $(free -g | awk '/Mem:/ {print $2}') GiB"
note "nvcc: $($NVCC --version | tail -1)"

FORK=$(git -C "$SRC" remote -v | awk '/ErikOHiggins\/PolyStrata/ && /\(fetch\)/ {print $1; exit}')
if [[ -z $FORK ]]; then
    git -C "$SRC" remote add fork https://github.com/ErikOHiggins/PolyStrata.git
    FORK=fork
fi
git -C "$SRC" fetch -q "$FORK" || { note "ERROR: git fetch $FORK failed"; exit 1; }
declare -A REFS=([base]=$BASE [engine]=$FORK/pr/laguna-engine [decode]=$FORK/pr/laguna-decode [serve]=$FORK/pr/laguna-serve)

stop_service() {
    if [[ $STOPPED == 0 ]] && systemctl is-active -q polystrata; then
        sudo systemctl stop polystrata && STOPPED=1 && note "polystrata stopped for the GPU stages"
    fi
}

# ---------------------------------------------------------------------------------------------------------------
if want setup; then
    say "setup"
    mkdir -p "$WT"
    for b in base engine decode serve; do
        if [[ -d $WT/$b ]]; then git -C "$WT/$b" checkout -q --detach "${REFS[$b]}"
        else git -C "$SRC" worktree add -q --detach "$WT/$b" "${REFS[$b]}"; fi
        note "  $b: $(git -C "$WT/$b" log -1 --format='%h %s')"
    done
    cp "$HERE/parity.py" "$HERE/longctx.py" "$TEST/" && note "  parity.py and longctx.py (with --tools) copied to $TEST"
fi

# ---------------------------------------------------------------------------------------------------------------
cuda_build() {   # NAME TARGET...: configure once, then build
    local b=$1; shift
    if [[ ! -f $WT/$b/build-poly/build.ninja ]]; then
        cmake -G Ninja -S "$WT/$b" -B "$WT/$b/build-poly" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
            -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PORTABLE=OFF -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_COMPILER="$NVCC" \
            -DSTRATA_GGML_DIR="$LLAMA" > "$OUT/cuda-$b.log" 2>&1 || return 1
    fi
    cmake --build "$WT/$b/build-poly" --target "$@" -j 8 >> "$OUT/cuda-$b.log" 2>&1
}
if want cuda; then
    say "cuda builds (sm_120)"
    for b in base engine decode; do
        t0=$(date +%s)
        targets=(strata-poly poly_attend_test poly_sink_attend_test)
        [[ $b == decode ]] && targets+=(poly_laguna_attend_test)
        if cuda_build "$b" "${targets[@]}"; then note "  $b: built in $(( $(date +%s) - t0 )) s"
        else note "  $b: BUILD FAILED, see cuda-$b.log:"; grep -m5 -E "error|Error" "$OUT/cuda-$b.log" | sed 's/^/    /' | tee -a "$SUM"; fi
    done
fi

# ---------------------------------------------------------------------------------------------------------------
if want tests; then
    say "tests"
    stop_service
    for t in poly_attend_test poly_sink_attend_test; do
        for b in base engine decode; do
            "$WT/$b/build-poly/$t" > "$OUT/$t-$b.txt" 2>&1; echo "exit $?" >> "$OUT/$t-$b.txt"
        done
        note "  $t: base $(tail -2 "$OUT/$t-base.txt" | paste -sd ' ')"
        for b in engine decode; do
            if cmp -s "$OUT/$t-base.txt" "$OUT/$t-$b.txt"; then note "    $b: the same output as base, line for line"
            else note "    $b: OUTPUT DIFFERS from base:"; diff "$OUT/$t-base.txt" "$OUT/$t-$b.txt" | head -8 | sed 's/^/      /' | tee -a "$SUM"; fi
        done
    done
    "$WT/decode/build-poly/poly_laguna_attend_test" > "$OUT/poly_laguna_attend_test.txt" 2>&1
    note "  poly_laguna_attend_test (decode), exit $?:"
    grep -E "a token over|passed|failed|FAIL" "$OUT/poly_laguna_attend_test.txt" | sed 's/^/    /' | tee -a "$SUM"
fi

# ---------------------------------------------------------------------------------------------------------------
if want sass; then
    say "sass"
    "$PY" "$HERE/same_sass.py" "$WT/base/build-poly/strata-poly" "$WT/engine/build-poly/strata-poly" > "$OUT/sass-base-engine.txt" 2>&1
    note "  base -> engine (exit $?):"; sed 's/^/    /' "$OUT/sass-base-engine.txt" | head -30 | tee -a "$SUM"
    "$PY" "$HERE/same_sass.py" "$WT/engine/build-poly/strata-poly" "$WT/decode/build-poly/strata-poly" > "$OUT/sass-engine-decode.txt" 2>&1
    note "  engine -> decode (exit $?):"; sed 's/^/    /' "$OUT/sass-engine-decode.txt" | head -30 | tee -a "$SUM"
fi

# ---------------------------------------------------------------------------------------------------------------
if want parity; then
    say "parity against llama.cpp's saved answers ($REF)"
    stop_service
    for b in engine decode; do
        for kv in fp16 q8_0; do
            args=(--src "$WT/$b" --tools "$WT/serve/tools" --reuse-ref "$REF" --only short,code,doc600,long --against-ours "$REF")
            [[ $kv == q8_0 ]] && args+=(--kv q8_0)
            "$PY" "$TEST/parity.py" "${args[@]}" > "$OUT/parity-$b-$kv.txt" 2>&1
            note "  $b $kv (exit $?):"
            grep -E "^   [a-z0-9]+: [0-9]+ of [0-9]+ agree|^== all prompts|^   all: same" "$OUT/parity-$b-$kv.txt" | cut -c1-150 | sed 's/^/  /' | tee -a "$SUM"
        done
    done
fi

# ---------------------------------------------------------------------------------------------------------------
if want longctx; then
    say "longctx, q8_0 (each about 25 minutes)"
    stop_service
    for b in engine decode; do
        "$PY" "$TEST/longctx.py" --src "$WT/$b" --tools "$WT/serve/tools" --kv q8_0 > "$OUT/longctx-$b.txt" 2>&1
        note "  $b (exit $?):"
        grep -E "^ +[0-9,]+ +[0-9,]+ +[0-9]+ +[0-9.]+ |^ +q8_0: 2K" "$OUT/longctx-$b.txt" | sed 's/^/  /' | tee -a "$SUM"
    done
fi

# ---------------------------------------------------------------------------------------------------------------
if want daily; then
    say "daily: ~/src/PolyStrata to $FORK/laguna"
    if [[ -n $(git -C "$SRC" status --porcelain --untracked-files=no) ]]; then
        note "  SKIPPED: $SRC has changes of its own (git status); nothing touched"
    else
        git -C "$SRC" checkout -q laguna && git -C "$SRC" reset -q --hard "$FORK/laguna"
        note "  $(git -C "$SRC" log -1 --format='%h %s')"
        if cmake --build "$SRC/build-poly" --target strata-poly -j 8 > "$OUT/daily-build.log" 2>&1; then note "  built"
        else note "  BUILD FAILED (daily-build.log); polystrata is started on the old binary"; fi
    fi
    if [[ $STOPPED == 1 ]]; then sudo systemctl start polystrata && STOPPED=0 && note "  polystrata started again"; fi
fi

# ---------------------------------------------------------------------------------------------------------------
if want serve; then
    say "serve tests on the PR 2 branch"
    ( cd "$WT/serve" && "$PY" -m unittest discover -s serve -p "test_*.py" -t . ) > "$OUT/serve-tests.txt" 2>&1
    note "  exit $?: $(grep -E '^Ran [0-9]+ tests' "$OUT/serve-tests.txt") $(tail -1 "$OUT/serve-tests.txt")"
    grep -E "^(FAIL|ERROR):" "$OUT/serve-tests.txt" | sed 's/^/    /' | tee -a "$SUM"
fi

# ---------------------------------------------------------------------------------------------------------------
DOCKER=""
docker_ready() {   # Docker found, and room for an image (asked before each stage: the images are large)
    if [[ -z $DOCKER ]]; then
        if docker info >/dev/null 2>&1; then DOCKER=docker
        elif sudo -n docker info >/dev/null 2>&1; then DOCKER="sudo docker"
        else note "  SKIPPED: no Docker (Ubuntu: sudo apt install docker.io && sudo usermod -aG docker \$USER, log in again)"; return 1; fi
    fi
    local root avail
    root=$($DOCKER info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)
    avail=$(sudo df -BG --output=avail "$root" 2>/dev/null | tail -1 | tr -dc 0-9)
    note "  docker: $($DOCKER --version), $root has ${avail:-?} GB free"
    if [[ -n $avail && $avail -lt 45 ]]; then note "  SKIPPED: under 45 GB free for the ROCm image"; return 1; fi
}
hip_build() {   # NAME IMAGE CMAKE-DEFS TARGETS: engine and decode, in a container of IMAGE with cmake and ninja added
    local name=$1 image=$2 defs=$3 targets=$4 tag="strata-verify-$1"
    if ! $DOCKER image inspect "$tag" >/dev/null 2>&1; then
        note "  pulling $image and adding cmake and ninja (once; large) ..."
        printf 'FROM %s\nRUN apt-get update && apt-get install -y --no-install-recommends cmake ninja-build git && rm -rf /var/lib/apt/lists/*\n' \
            "$image" | $DOCKER build -q -t "$tag" - > "$OUT/$name-image.log" 2>&1 ||
            { note "  IMAGE FAILED ($name-image.log)"; return 1; }
    fi
    for b in engine decode; do
        local t0 tg=$targets
        t0=$(date +%s)
        [[ $b == engine ]] && tg=${targets/ poly_laguna_attend_test/}
        $DOCKER run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$WT/$b:/src" -v "$LLAMA:/llama:ro" "$tag" bash -c "
            set -e; R=/opt/rocm
            BC=\$(ls -d \$R/lib/llvm/amdgcn/bitcode \$R/amdgcn/bitcode 2>/dev/null | head -1)
            export HIP_PLATFORM=amd ROCM_PATH=\$R HIP_PATH=\$R PATH=\$R/bin:\$R/llvm/bin:\$PATH
            hipcc --version | head -2
            cmake -S /src -B /src/build-$name -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_BUILD_TESTS=OFF \
                -DSTRATA_PORTABLE=OFF -DSTRATA_GGML_DIR=/llama $defs
            cmake --build /src/build-$name --target $tg -j 8" > "$OUT/$name-$b.log" 2>&1
        local rc=$?
        if [[ $rc == 0 ]]; then note "  $b: built ($tg) in $(( $(date +%s) - t0 )) s, $(sed -n 1,2p "$OUT/$name-$b.log" | paste -sd ' ' | cut -c1-80)"
        else note "  $b: BUILD FAILED ($name-$b.log):"; grep -m6 -E "error" "$OUT/$name-$b.log" | cut -c1-200 | sed 's/^/    /' | tee -a "$SUM"; fi
    done
}
if want hip; then
    say "hip (gfx1100, $HIP_IMAGE)"
    docker_ready && hip_build hip "$HIP_IMAGE" \
        "-DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_HIP_COMPILER=\$R/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=\$R -DCMAKE_PREFIX_PATH=\$R \"-DCMAKE_HIP_FLAGS=--rocm-path=\$R --rocm-device-lib-path=\$BC\"" \
        "strata-poly poly_laguna_attend_test"
fi
if want gfx906; then
    say "gfx906 ($GFX906_IMAGE)"
    docker_ready && hip_build gfx906 "$GFX906_IMAGE" \
        "-DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 -DCMAKE_C_COMPILER=\$R/llvm/bin/clang -DCMAKE_CXX_COMPILER=\$R/llvm/bin/clang++" \
        "strata-poly poly_laguna_attend_test"
fi

say "done: $OUT"
echo "Send back: $SUM"
