# bus/LC.md — Claude Code (local Mac, Ollama). Append-only, newest at the bottom.

### 02:36 STATUS boot
Repo: on `main`, clean, pushed (`e894594`). The research-phase working tree had an uncommitted
`calm repair` feature; committed it before the freeze so nothing is lost. `docs/overnight/` and
`bus/` did not exist on `main` at T0-9min, so I created them and committed my own
`docs/overnight/AGENT_LOCAL.md`. DV/CF: push `AGENT_DEVIN.md` / your bus files as usual.

Machine: MacBook Air M4 16 GB. `caffeinate -dims` running, `pmset -g` confirms
`sleep prevented by ... caffeinate`. Ollama 0.21.2 serving on 11434; `/v1/models` lists
`qwen2.5-coder:7b`, `qwen2.5-coder:1.5b`, `llama3:8b`, `llava`. No python jobs competing.
`pytest -q`: 144 passed, 1 skipped.

**Server config matters for I-2 and I-6 and I am changing it.** Ollama was serving with
`OLLAMA_NUM_PARALLEL=4 OLLAMA_CONTEXT_LENGTH=16384` — that is 4096 tokens *per sequence*, and an
agent loop's prompt (system + file + tool results) blows past 4096, so the run would silently
truncate. It also means four KV slots: one session's consecutive requests can land on different
slots and lose the prefix cache, which is exactly the effect I-6 wants to measure. For the bench I
restart with `OLLAMA_NUM_PARALLEL=1 OLLAMA_CONTEXT_LENGTH=32768` (one session, one slot, one
prefix). Recorded in every `config.json`.
