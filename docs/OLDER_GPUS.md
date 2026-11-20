# Older GPUs (experimental)

Puddle's supported cards are NVIDIA RTX 20 / 30 / 40 / 50 and the AMD cards in [AMD_HIP.md](AMD_HIP.md). Pascal and
Volta NVIDIA cards (Tesla P100, P40 / P4, GTX 10 series, Tesla V100, Titan V) run through an **opt-in, experimental**
path: strata-poly compiled with CUDA 12. Nobody has run Puddle's models on one yet.

## NVIDIA: strata-poly with CUDA 12

CUDA 13 dropped Pascal and Volta: it cannot compile for them. So setup compiles strata-poly with a CUDA 12.x toolkit
for a model that runs on such a card, with `-DSTRATA_EXPERIMENTAL_SM60=ON`, into `build-poly-cuda12` (and the image
encoder into `build-vision-poly-cuda12`). One engine runs per model, so the choice is made per model, by the oldest
card that model runs on, and kept in its config (`"cuda": 12`).

An older card is used only when you choose it; a PC with a newer card keeps recommending the newer one.

| You | Setup |
| --- | --- |
| have only Pascal / Volta NVIDIA cards (and no AMD card it can use) | uses them, with CUDA 12 |
| name the card: `./setup.sh --setup --gpu 1`, or `--gpus 0,1` with a newer card | uses it, with CUDA 12 |
| `--cuda 12` (or `STRATA_CUDA=12`) | CUDA 12 for this model, on any card |
| `--cuda 13` | CUDA 13 even with an older card (a warning: it has no code for that card) |
| `STRATA_EXPERIMENTAL_SM60=1` | the older cards are listed as usable |

Setup looks for the newest CUDA 12.x toolkit; `STRATA_NVCC=<path to nvcc>` picks one. An RTX 50 card needs 12.8 or
newer.

`--cuda 12` is also the way to run Puddle with an NVIDIA driver older than 580: CUDA 12 needs 528 or newer on Windows
and 525 on Linux; setup's "driver too old" stop says so. Such old drivers were not tested.

What PolyStrata measured with strata-poly built this way: on two Tesla V100-PCIE-32GB, with CUDA 12.9.1, Qwen3.6
UD-Q3_K_M split over both cards wrote 157 tokens/s in a chat and read 4K and 16K prompts at 2,615 and 2,778 tokens/s
([PolyStrata issue 88](https://github.com/simplexai-labs/PolyStrata/issues/88)). Built as compute_61 PTX only and run on
an RTX 3080, which takes the Pascal code paths, Qwen3.6's greedy answer matched llama.cpp's for 60 of 64 tokens (the
sm_86 build: 64) and differed after that by a near tie. On an RTX 5090 with driver 570, strata-poly built with CUDA
12.8 read 31,516-token prompts at 6,200 to 6,700 tokens/s. On Windows, with CUDA 12.9 and Visual Studio's 14.44
compiler, it builds for sm_61 and sm_89; the sm_89 build wrote the same 200 tokens of Qwen3.6 UD-Q2_K_XL as the CUDA 13
one on an RTX 4070 Laptop GPU, at 95.6 tokens/s against 93.2.

## AMD

The RX 5600 XT / 5700 / 5700 XT (gfx1010) and the RX 6000 series (gfx1030, gfx1031, gfx1032) are in setup's list and
unvalidated ([AMD_HIP.md](AMD_HIP.md)). Other older AMD cards (gfx1012, gfx906) are not offered.

## Reports welcome

These paths stay experimental until more people run them. A report with the card, the driver version, the model and
the engine log (`strata-<model>.log`) in an issue helps; [COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md) has the
format for measurements.
