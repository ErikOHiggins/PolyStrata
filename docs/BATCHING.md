# Several requests at once (batch slots)

By default Puddle serves **one request at a time**: the others wait in the server's queue. With `"parallel": N` (the
engine's `--batch N`, also spelled `--slots N`) strata-poly keeps up to N conversations open and writes them
**together**, a token of each at a time, so the dense weights and every routed expert two conversations share are read
once for all of them. It is opt-in and changes nothing when the option is absent. Not measured with Puddle's models
yet; the numbers below are PolyStrata's.

## How it works

`"parallel": N` in the model's config (`strata-<model>.json`), or `--parallel N` when you run setup, up to 8. The
server drives it like this:

- A request alone writes on its own and guesses ahead as always, so one user sees no change.
- When a second one comes, the first goes on in a batch slot from where it was, nothing read again, and the two
  write together, a token each at a time, without guesses. Each slot has its own position, recurrent state and
  attention rows; the dense weights and every routed expert two requests both use are read once for both.
- A request's prompt is read the usual way, with the conversation cache, and copied into its slot on the card.
  While a prompt is read the slots go on writing, for half the time each piece of the prompt took, and a long read
  gives way to a much shorter request at its next piece (`BYIELD`).
- Left alone again, a request goes back to writing on its own with guesses, also without reading anything again.

Where it pays depends on the card. PolyStrata measured: on an RTX 4070 laptop (8 GB) most experts of Qwen3.6 run on the CPU, and
different requests route to different experts, so the CPU does each request's own: three chats of 150 tokens with
UD-Q3_K_M came at 105 tokens/s together, and at 79 one after the other. A card that holds every expert of the model
gains more. On an RTX 5090 with Qwen3.6 UD-Q4_K_M, chats of 300 tokens through the server:

| Chats at once | Tokens/s, all together |
| ---: | ---: |
| 1 (with guesses) | 312-333 |
| 2 | 315-324 |
| 3 | 381-389 |
| 4 | 428-430 |

Each slot's attention rows take VRAM from the expert cache as its conversation grows, as the first conversation's
do. A model split across cards (`"layer_split"`) runs its slots card after card.

Check it on your PC with the engine alone: `strata-poly --model M.gguf --batch 3 --check-batch 3 --tokens ID,ID,...`
answers three prompts (the given one, and it without its first one and two tokens) one after the other and then
together, and says for each how many tokens came out the same. To compare token by token, keep the expert cache
fixed and leave guesses and the tail skip out: `--expert-swaps 0 --cpu-share 0 --window 1 --tail-skip 0`. The
check gives the attention caches their VRAM at load (`--kv-whole`): otherwise the sequences growing together take
VRAM from the expert cache, an expert that moves to the CPU rounds differently, and long prompts part for that
reason alone ([PolyStrata issue 55](https://github.com/simplexai-labs/PolyStrata/issues/55)). On the
laptop every token matched for Qwen3.6, Gemma 4 and LFM2.5 with 2 to 4 sequences and prompts of 19 to 3,000 tokens,
and on the RTX 5090 for Qwen3.6, MiniCPM-V, LFM2.5, DeepSeek V4 Flash and GLM 5.3 Flash with 3 sequences after
prompts of 64 and 3,000 tokens, and for Flint after prompts of 15 and 3,000 tokens, with every expert on the card
and with the card held to 8 GB. Through the server on the laptop, with the same four options in the run config,
three chats at once gave the text each gave alone, to the character.

Without those options the text of a chat answered together can part from the one it gets alone after a few dozen
words, and so can two runs of the same chat alone. On the laptop that happened with the run config as setup writes
it, and with the expert cache fixed but guesses on. The tail skip (Qwen3.6's default 6) leaves out an expert outside
VRAM only where no token of the call ranks it high, so with it a request's text also depends on the others written
with it.

### Guesses inside the slots (`--batch-mtp`)

A model whose file carries a block that guesses the next token (upstream's Qwen3.6, Ornith 1.5 and Tiel Coder; Laguna
has none) uses it alone as always. With `--batch-mtp` in the config's `"args"` (or `STRATA_BATCH_MTP=1` in the
server's environment) the slots that write together use it too: before each step every slot's
block guesses the token after its next one, all slots in one pass, and the step takes each slot's token and its
guess together. Where the guess is what the model picks, the slot gets two tokens from that step; where it is not,
the slot goes back to the state after its first token. The answers are the ones each request gets alone.

Measured with `--check-batch` (the same tokens as each prompt alone, a Python prompt of 47 tokens, 160 tokens
written, 94 to 96% of the guesses right), tokens/s of all slots together:

| Machine, model | slots | without | with `--batch-mtp` |
| --- | ---: | ---: | ---: |
| RTX PRO 6000 (Linux), Qwen3.6 UD-Q3_K_M, every expert on the card | 2 | 295-333 | 412-472 |
| same | 3 | 428-429 | 442-446 |
| same, the card held to 9 GB (most experts on the CPU) | 2 | 144-154 | 160-164 |
| M5 Ultra (Metal), Qwen3.6 UD-Q2_K_XL | 2 | 140 | 173 |

It pays most with two slots. With three a step reads the experts of six rows instead of three, and that takes back
most of what the guesses save. It is off by default. Two chats at once through the server gave each the same text
as alone (the RTX PRO 6000, greedy, the expert cache held still).

## Engine protocol (`--serve`)

On top of `GEN` / `GENI`:

| Line | Direction | Meaning |
| --- | --- | --- |
| `BGEN <slot> <max_new> [keys] <ids>` | in | read the prompt (as `GEN 1`), then continue in `<slot>` |
| `BGENI <slot> <max_new> [keys] <file> <ids>` | in | the same with images |
| `BADM <slot> <1/0>` | out | after the admission's `DONE`: 1 = it continues in the slot, 0 = it ended |
| `BT <slot> <id>` | out | a token of that slot |
| `BDONE <slot> <generated> <stop/length/cancel> <ms>` | out | the slot is free again (it keeps its conversation) |
| `BSTOP <slot>` | in | end that slot at its next window |
| `BYIELD <slot>` | in | the prompt being read gives way at its next chunk boundary; its part read waits in `<slot>` (the admission's own, or a free slot for a solo request) |
| `YIELDED <slot> <tokens>` | out | before the `DONE cancel` of a read that gave way: the request is sent again later and goes on from there |
| `INFO ... batch_slots=N` | out | the slots the engine runs (only with `--batch`) |

`tools/batch_test.py` drives the engine directly: the same prompts alone, then together, compared token by token,
and the aggregate rate.
