<h1 align="center">Puddle</h1>

<p align="center"><b>Large AI models on your own PC</b><br>
Laguna-S 2.1, Laguna-XS 2.1, Occamy 1.0 and Gemma 4 26B-A4B · free and open source</p>

Puddle runs mixture-of-experts models that usually need a server on a normal PC: one or two graphics cards and the
PC's RAM. They chat, write code and work with your apps and coding agents over an OpenAI- and Anthropic-compatible
API on `http://127.0.0.1:8080`. Nothing leaves your PC.

Puddle is a fork of [PolyStrata](https://github.com/simplexai-labs/PolyStrata) by VecSzn, which is built on
[Strata](https://github.com/Niko1221/Strata) by Niko1221. It keeps PolyStrata's second engine, **strata-poly**, with
its server, web app and installer, and adds poolside's Laguna. Strata's Qwen3.8 engine and PolyStrata's other models
are not part of Puddle.

## The models

| Model | By | Sizes setup offers | Reads pictures |
| --- | --- | --- | --- |
| **[Laguna-S 2.1](https://huggingface.co/poolside/Laguna-S-2.1)** | poolside; quantized by Unsloth | UD-Q3_K_XL (54 GB, the default), UD-Q4_K_XL (73 GB), poolside's own Q4_K_M (96 GB) | no |
| **[Laguna-XS 2.1](https://huggingface.co/poolside/Laguna-XS-2.1)** | poolside | Q4_K_M (20 GB) | no |
| **[Occamy 1.0 35B-A3B](https://huggingface.co/Accio-Lab/occamy-1.0-GGUF)** | Accio-Lab (APEX quantization) | APEX-Quality (23 GB, the default), APEX-I-Compact (17 GB), APEX-Balanced (25 GB) | yes |
| **[Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it)** | Google; quantized by Unsloth | UD-Q3_K_M (13 GB, the default), UD-Q2_K_XL, UD-Q4_K_M, QAT-UD-Q4_K_XL | yes |

All four are experimental. poolside's Laguna-S Q4_K_M keeps 8 of its 47 expert layers in BF16, which run on the
processor every token, and it has a 1M-token rope where Unsloth's files have 262K; it has not been run yet.

## How fast is it?

A token is about ¾ of a word. Measured so far:

**Laguna-S 2.1 UD-Q3_K_XL** on two RTX 5060 Ti 16 GB (PCIe 3.0 x8 and x4), an i9-9900K and 128 GB of DDR4-3200,
Linux, with the KV cache in q8_0 and prompt-lookup guesses off (2026-11-14):

| Context | 2K | 32K | 64K | 128K | 168K | 196K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Writes, tokens/s | 37.2 | 29.0 | 27.5 | 22.3 | 19.4 | 19.4 |
| Reads the prompt, tokens/s | 258-269 | about 700 | about 470 | about 310 | about 225 | about 190 |

Against llama.cpp on the same file (six prompts, 64 steps each, llama.cpp choosing each token), strata-poly's first
choice was the same at 95.3% of the 384 steps, and in every prompt the first difference was a near tie.

**Gemma 4 26B-A4B UD-Q3_K_M**, measured by PolyStrata: writes 92-93 tokens/s and reads a prompt at 2,240 on a laptop
with an RTX 4070 Laptop (8 GB), a Core i7-14700HX and 32 GB of RAM; 353-356 and 8,420-8,480 on an RTX 5090.

Laguna-XS and Occamy have not been measured here yet. `./bench.sh` (`BENCH.bat`) measures your own PC and writes a
report ([how](docs/COMMUNITY_BENCHMARKS.md)).

## What you need

| Model | Graphics card | RAM, by setup's estimate | Disk |
| --- | --- | --- | --- |
| Laguna-S 2.1 | NVIDIA RTX 20 series or newer; it has run on two 16 GB cards | about 51 GB beside a 16 GB card (UD-Q3_K_XL) | 54 to 96 GB |
| Laguna-XS 2.1 | same | about 17 GB beside a 16 GB card | 20 GB |
| Occamy 1.0 | same | about 12 to 21 GB beside a 16 GB card | 17 to 25 GB |
| Gemma 4 26B-A4B | NVIDIA RTX 20 series or newer, 8 GB or more | about 15 to 21 GB beside an 8 GB card | 11 to 17 GB |

A larger card needs less RAM, and two or three cards can share a model ([multi-GPU](docs/MULTI_GPU.md)). Laguna
and Occamy have run on Linux only so far; setup says so and asks before it installs them on Windows. AMD cards
(Linux and Windows) and Pascal / Volta cards (with CUDA 12) are experimental: [AMD](docs/AMD_HIP.md),
[older cards](docs/OLDER_GPUS.md). On a Mac (Apple Silicon), setup builds strata-poly's Metal engine; Laguna does not
run on it yet, and Occamy and Gemma 4 have not been run there ([docs/MAC.md](docs/MAC.md)).

## Install

### Let your AI set it up

Paste this into your AI coding assistant (Claude Code, Cursor, Codex, ...):

```text
Set up Puddle on this PC for me: https://github.com/ErikOHiggins/Puddle - follow docs/AI_SETUP.md in that repository.
```

AI tools can also install, start and stop Puddle through its [MCP server](docs/MCP_SERVER.md).

### Or do it yourself

[Download Puddle](https://github.com/ErikOHiggins/Puddle/archive/refs/heads/main.zip) and unzip it, or `git clone` it.
**Windows:** double-click **`START-HERE.bat`**. **Linux and Mac:** run **`./setup.sh`** in the Puddle folder.

Setup checks your PC, then asks which model, which size, how much context and whether the model should read pictures.
Press Enter for the recommended answer. Then it installs the Python packages, compiles strata-poly for your card (it
installs the build tools first, and asks; the first compile takes 10 to 20 minutes), downloads the model and starts
it. If the download stops, run it again: it continues where it stopped. Your browser opens the app at
`http://127.0.0.1:8080`.

`./setup.sh --family laguna-s` (or `laguna-xs`, `occamy`, `gemma4`) goes straight to one model, and `--model SIZE`
to one size. Every option: [docs/INSTALL.md](docs/INSTALL.md).

**Next time**, run `START-HERE.bat` (or `./setup.sh`) again: it starts right away and downloads nothing twice. Close
its window to stop the model. `UPDATE.bat` (`./update.sh`) updates Puddle without starting it.

## Using it

- **In the browser:** `http://127.0.0.1:8080` has **Chat**, a live **Monitor** of the model and your GPU, CPU and
  RAM, and **About** with the settings and addresses.
- **Your apps and coding agents:** an "OpenAI-compatible" provider with the base URL **`http://127.0.0.1:8080/v1`**.
  Apps that use Anthropic's API: `http://127.0.0.1:8080/v1/messages` (Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`).
  The OpenAI Responses API (Codex CLI): `/v1/responses`. All of it: [docs/API.md](docs/API.md).
- **Thinking:** off, low, medium or high, in the chat menu or your app's "reasoning effort".
- **Pictures** (Occamy, Gemma 4): say yes to "Images?" in setup, then attach pictures in the chat or your app.
- **Making pictures** (optional): `./setup.sh --setup --images flux2-klein` adds FLUX.2 klein 4B in
  stable-diffusion.cpp's own web UI, at `/sd/` ([docs/IMAGES.md](docs/IMAGES.md)).
- **From your phone or another PC:** `./setup.sh --setup --host 0.0.0.0 --api-key <secret>`. Always set a key.

## Something went wrong?

[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) has the common problems and their fixes. Still stuck? Open an
[issue](https://github.com/ErikOHiggins/Puddle/issues) and attach `strata-<model>.log` from the Puddle folder. A
security problem: [SECURITY.md](SECURITY.md).

## How it works

A mixture-of-experts model uses only a few of its experts for each token. strata-poly reads the model file in place:
the layers every token needs and as many experts as fit go to the graphics card, and the rest stay in RAM (the
system's file cache), where the processor computes them when a token needs them. The engine counts which experts
are used and keeps the counts for its next start, when the most-used ones go to the card first. How the engine and its backends work:
[docs/BACKENDS.md](docs/BACKENDS.md).

## Credits and license

Laguna-S 2.1 and Laguna-XS 2.1 are by [poolside](https://huggingface.co/poolside), under the OpenMDW-1.1 license;
Laguna-S in [Unsloth](https://huggingface.co/unsloth)'s quantization and poolside's own. Occamy 1.0 is by
[Accio-Lab](https://huggingface.co/Accio-Lab), under Apache 2.0. Gemma 4 is by Google, under its own terms, in
Unsloth's quantization.

Puddle is built on [PolyStrata](https://github.com/simplexai-labs/PolyStrata) by VecSzn and its contributors, which is
built on [Strata](https://github.com/Niko1221/Strata) by Niko1221 and its contributors (how Strata's engine works:
[the Strata paper](https://github.com/simplexai-labs/PolyStrata/blob/67dde645dd94190733cca8ad78d2faa68b930e2a/docs/paper/Strata-Paper.pdf)).
It uses parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp), and image generation uses
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp). Puddle is open source under the
[MIT License](LICENSE); every model has its own license.
