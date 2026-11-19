serve: Laguna's tokenizer and chat format

## Title
Issue: #16 (the model vote)

## Summary
The Python side of Laguna S 2.1: its pre-tokenizer and its chat format. It writes GLM-style tool calls, opens its
turns with `<assistant>`, and its template takes tools in OpenAI's shape. This PR is independent of the engine PR
(#166). Other models take the same paths as before.

## What changed
- `tools/strata_tokenizer.py`: the `laguna` pre-tokenizer (llama.cpp's `LLAMA_VOCAB_PRE_TYPE_LAGUNA`). It splits text
  into lines and runs of newlines first, then applies Qwen 2's pattern.
- `serve/models.py`: `laguna` in `BACKENDS`.
- `serve/server.py`:
  - `call_format "glm"` for `laguna`.
  - `TURN_OPENERS_OF_PRE`: of Laguna's turn tags only `<assistant>` is one token.
  - `openai_tools`: Gemma and DeepSeek as before, plus Laguna.
- `serve/test_laguna.py` (new, 8 tests): pre-tokenizer, tool-call parsing, stops and turn opener, BOS and thinking in
  the prompt, prompt reuse across turns, tool shape.

## How it was checked
- The tokenizer gives the same ids as `llama-tokenize` (llama.cpp build 12242) on the smoke-test and parity prompts.
- `python -m unittest discover -s serve -p "test_*.py" -t .` on Linux: 998 tests, OK (11 skipped).
- Through the real server with the engine PR, on 2× RTX 5060 Ti: thinking on and off, a tool call and its result,
  streaming, a second turn.
