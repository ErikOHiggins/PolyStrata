# Laguna-S-2.1 on echo: getting the files (2026-11-13)

Targets: Unsloth `UD-Q3_K_XL` (54.1 GB, the expected daily driver) and `UD-Q4_K_XL` (73.4 GB, the quality
reference), both at revision 750f92f9. Details and VRAM estimates are in `../HANDOFF-laguna-polystrata.md`
(section 1, "Target files", and section 7.4).

`download-laguna.sh` runs on echo. It fetches Q3 first, then Q4, into
`/home/eohiggins/PolyStrata-data/models/laguna-s21-<quant>/`. It checks the free space first, resumes partial
files, and checks every shard's SHA-256. Running it again skips the shards that are done.

PowerShell on this workstation:

```powershell
# 1) free space where the models go (needs about 119 GiB, plus 5)
ssh eohiggins@echo 'df -h /home/eohiggins/PolyStrata-data/models'

# 2) copy the script over
ssh eohiggins@echo 'mkdir -p ~/laguna-download'
Set-Location V:\common\configs\Ploystrata\laguna
scp download-laguna.sh eohiggins@echo:laguna-download/

# 3) start it detached: it keeps going after the ssh session ends
ssh eohiggins@echo 'setsid nohup bash ~/laguna-download/download-laguna.sh > ~/laguna-download.log 2>&1 < /dev/null &'

# 4) progress, any time (.part files grow; "ok" lines mean a checked shard)
ssh eohiggins@echo 'tail -n 8 ~/laguna-download.log; ls -l --block-size=M ~/PolyStrata-data/models/laguna-s21-*/'
```

To fetch only one of the two, put it in front of the command in step 3:
`QUANTS=UD-Q3_K_XL setsid nohup bash ...`.

## Daily use (2026-11-14)

`deploy-laguna.sh` (run on echo) installs `strata-laguna-ud-q3_k_xl.json` into `~/PolyStrata`, the tokenizer and
the expert profile into `~/PolyStrata-data/packs/laguna-ud-q3_k_xl/`, and `polystrata.service` with ExecStart on
the dev checkout's `server.py`. It asks before each step and backs up what it replaces.

```powershell
ssh eohiggins@echo 'mkdir -p ~/laguna-deploy'
Set-Location V:\common\configs\Ploystrata\laguna
scp deploy-laguna.sh strata-laguna-ud-q3_k_xl.json polystrata.service eohiggins@echo:laguna-deploy/
```

On echo, in tmux:

```bash
cd ~/src/PolyStrata && git pull
export PATH="$HOME/PolyStrata/.venv/bin:$PATH"
cmake --build build-poly --target strata-poly -j 8 2>&1 | tail -3
bash ~/laguna-deploy/deploy-laguna.sh
```

## The PR evidence (2026-11-14)

`verify-prs.sh` (on echo, in tmux) builds main, PR 1 and PR 3 for CUDA and runs the attention tests on all three.
It compares their kernels' machine code (`same_sass.py`), runs parity and the long-context speed for PR 1 and PR 3,
moves `~/src/PolyStrata` to the new `laguna` branch and restarts polystrata, runs the server tests on PR 2, and
builds PR 1 and PR 3 for HIP (gfx1100, gfx906) in Docker. Results go to `~/laguna-test/verify-<time>/summary.txt`.

```powershell
Set-Location V:\common\configs\Ploystrata\laguna
scp verify-prs.sh same_sass.py parity.py longctx.py eohiggins@echo:laguna-deploy/
```

```bash
bash ~/laguna-deploy/verify-prs.sh
```
