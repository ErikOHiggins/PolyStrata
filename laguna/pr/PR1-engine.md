poly: run Laguna S 2.1 (llama.cpp architecture `laguna`)

## Title
Issue: #16 (the model vote)

## Summary
strata-poly runs poolside's Laguna S 2.1 (118B-A8B) GGUFs. The model code follows `flint.cpp`, which has the same
3-of-4 sliding-window layout and sigmoid routing: `git diff --no-index src/poly/flint.cpp src/poly/laguna.cpp` is 20
hunks. Other models are unaffected. The chat format is a separate PR (#167), and so is a faster decode kernel.

## What changed
- `src/poly/laguna.{hpp,cpp}` (new): the model. Unlike Flint: query heads per layer (48 full, 72 window), YaRN on
  the full layers, a softplus output gate, no attention sinks.
- `src/poly/kernels.cu`, `include/strata/poly/kernels.hpp`: `head_norm_rope_yarn()`, as `head_norm_rope_kernel<true>`.
  The `<false>` instantiation is the old kernel body unchanged. Also `mul_softplus()`.
- `src/poly/engine.cpp`: the factory entry (not on Metal), and `laguna` in the `--kv q8_0` gate.
- `CMakeLists.txt`: `laguna.cpp` in the three targets that list the models.
- `docs/Q8_KV.md`: `laguna` in the list of models.

## How it was checked
2× RTX 5060 Ti 16 GB, i9-9900K, 128 GB DDR4-3200, Linux, CUDA 13.4 (driver 580.196.07);
`Laguna-S-2.1-UD-Q3_K_XL` (Unsloth).

- **Against llama.cpp** (build 12242, same file, greedy): strata-poly follows llama.cpp's answer and its 10 likeliest
  tokens are compared at each step. 4 prompts of 6 to 2,410 tokens, 64 steps each.

  | KV cache | same top token | KL median | KL mean |
  |---|---|---|---|
  | fp16 | 95.7% | 2.3e-3 | 4.0e-2 |
  | q8_0 | 95.7% | 1.7e-3 | 3.6e-2 |

  For scale, strata-poly's own q8_0 and fp16 runs differ by as much: 95.3% same top token, KL median 2.3e-3.
- **Default path unchanged:**
  - `cuobjdump -sass`: all 2,559 kernels of main's `strata-poly` are in this build with the same machine code.
    `head_norm_rope_kernel` is now named `head_norm_rope_kernel<false>`, with the same instructions.
  - `poly_attend_test` and `poly_sink_attend_test` print the same output as on main.
  - On the host side, other models see one change: `laguna` added to the `--kv q8_0` refusal message.
- **Builds:** CUDA sm_120; HIP gfx1100 (ROCm 7.2.4, `rocm/dev-ubuntu-24.04:7.2.4-complete` in Docker).
  - gfx906: not applicable. CMake leaves strata-poly out of that build (`NOT STRATA_HIP_GFX906`), and it compiles
    none of these files.
  - Metal: not built (no Mac). Laguna is left out of the Metal build by `#if`. Metal compiles `kernels.hpp`, which
    only gains declarations; nothing on Metal calls them.
  - SYCL: no file touched.

## Extra Notes
- Not run on an AMD card. Not tested: UD-Q4_K_XL, a single GPU.
- Decode on this machine with `--kv q8_0`: 35 tok/s at 2K of context, 8.9 at 196K. The decode PR addresses the
  long-context number.
