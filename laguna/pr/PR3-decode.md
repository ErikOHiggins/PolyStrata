poly: a decode attention kernel for Laguna's full layers

## Title
Issue: #16 (the model vote). Depends on #166: review the last commit.

## Summary
Laguna's 12 full-attention layers (heads of 128, groups of 6) used `attend()`'s generic score, softmax and value
kernels, which read a long KV cache well below the card's bandwidth. `attend_group()` is a split-K kernel for this
shape, for fp16 and q8_0 caches. `attend()` itself is unchanged, and only Laguna calls the new function.

## What changed
- `src/poly/kernels.cu`, `include/strata/poly/kernels.hpp`: `attend_group()` and `attn_split_group_kernel`, on
  `attn_split_kernel`'s pattern.
  - Blocks of 256 rows, with 4 rows' reads in flight per warp.
  - Any other call falls back to `attend()`.
  - `attn_join_kernel` takes the block size as a template parameter. Its default, `SPLIT_KEYS`, keeps the existing
    instantiations the same.
- `src/poly/laguna.cpp`: calls `attend_group()`, and sizes its scratch with `attend_group_floats()`.
- `src/poly/laguna_attend_test.cu` (new, run by hand) and its CMake target.

## How it was checked
2× RTX 5060 Ti 16 GB (PCIe 3.0 ×8 and ×4), i9-9900K, 128 GB DDR4-3200, CUDA 13.4; UD-Q3_K_XL with `--kv q8_0`.

Decode speed in tok/s (128 tokens written at each depth of one growing conversation; same day, prompt lookup off):

| context | 2K | 32K | 64K | 128K | 168K | 196K |
|---|---|---|---|---|---|---|
| #166 | 35.1 | 24.1 | 17.7 | 12.1 | 9.5 | 8.9 |
| this PR | 34.5 | 30.1 | 28.1 | 23.0 | 18.7 | 18.8 |

An earlier run of the same kernel gave 19.4 at 196K; the share of experts found in VRAM varies from run to run
(75-78% here at that depth).

- `poly_laguna_attend_test`:
  - results within 2e-5 of `attend()`;
  - a token alone, a token with the call's other tokens, and the position read from the card give the same bits;
  - without room in the scratch it gives `attend()`'s bits exactly;
  - one token over 70K rows: fp16 0.71 ms (402 GB/s of the cache) against `attend()`'s 1.58 ms; q8_0 0.58 ms
    (263 GB/s) against 1.87 ms.
- Parity against llama.cpp is as in #166: fp16 94.5% same top token (KL median 1.9e-3), q8_0 95.7% (1.9e-3).
- `cuobjdump -sass`: all 2,561 kernels of #166's build are in this build with the same machine code.
- Builds: CUDA sm_120; HIP gfx1100 (ROCm 7.2.4 in Docker), the test included.
  - gfx906: not applicable. CMake leaves strata-poly out of that build (`NOT STRATA_HIP_GFX906`), and it compiles
    none of these files.
  - Metal: not built (no Mac); `kernels.hpp` only gains declarations there.
  - SYCL: no file touched.
