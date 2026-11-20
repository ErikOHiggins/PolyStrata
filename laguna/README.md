# laguna/: Laguna's bring-up and measurement tools

The tools used to port poolside's Laguna (Laguna-S 2.1, Laguna-XS 2.1) to strata-poly and to measure it. Each one runs
the engine of a checkout (`--src`, this one by default; its `build-poly/strata-poly` must be built) on a GGUF you name
(`--gguf`), and writes its results to `--out` (`laguna-test/` in the current folder by default). They need the GPUs to
themselves: stop any server that uses them first (`--service NAME` refuses to run while that systemd service is active).

| Tool | What it does |
|---|---|
| `smoke.py` | The first run: the tokenizer from the GGUF, three prompts through the engine's own line protocol, their speed, and the engine's rope line. `--llama DIR` compares the prompt ids with llama.cpp's `llama-tokenize`. |
| `parity.py` | Token-by-token agreement with llama.cpp on the same GGUF: llama.cpp answers greedily, strata-poly is made to follow that answer, and the first disagreement and the KL divergence over llama.cpp's 10 likeliest tokens are reported per prompt. Needs a Laguna-capable llama.cpp (`--llama DIR`). |
| `longctx.py` | Prompt and decode speed as the context grows (2K to 200K by default), per KV format (`--kv fp16,q8_0`), with the share of routed experts found in VRAM at each depth. |
| `chat_smoke.py` | Laguna through this checkout's server: thinking on and off, a tool call and its answer, streaming, and how much of a second turn's prompt is read again. |
| `gguf_types.py` | The tensor types of Unsloth's Laguna-S GGUFs by role (routed experts, shared expert, attention, ...), read from the shards' headers over HTTP range requests; the `gguf-types-*.json` files here are its output for UD-Q3_K_XL, UD-Q4_K_S and UD-Q5_K_S. |
| `same_sass.py` | Whether every GPU kernel of one build is in another, instruction for instruction (their SASS, by `cuobjdump`): shows a change left the kernels it should not touch as they were. |

    python laguna/smoke.py --gguf MODEL.gguf
    python laguna/parity.py --gguf MODEL.gguf --llama LLAMA_CPP/build/bin --kv q8_0
    python laguna/longctx.py --gguf MODEL.gguf --kv q8_0 --depths 2048,32768,131072,200704

Their long prompts are the repository's own text, read as it was at a fixed git revision (`--corpus`, `ac62d979` by
default, the one every run so far used), so a run stays comparable with the earlier ones although the docs have
changed since.

`reference/` holds what the port was checked against: the GGUF metadata of poolside's and Unsloth's files, the
Hugging Face configs and chat templates, llama.cpp's `laguna.cpp`, and PolyStrata's Flint code the port started from.
`pr/` holds the write-ups of the pull requests the port was sent upstream in.
