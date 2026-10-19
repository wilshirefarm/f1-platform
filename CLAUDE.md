# CLAUDE.md

F1 real-time ML platform: replay past races as a live stream, predict finishing order lap by lap,
validate the stream, and parse race control messages with a fine-tuned model.

- **Source of truth:** `PLAN.md` (stages, goals, deliverables). Don't overwrite it; propose edits.
- **Where we are:** `PROGRESS.md` (stage checklists).

## Session start / end
- Start: read `PROGRESS.md` and `docs/decisions.md`; summarize where things stand in 3–5 lines,
  then propose the next checklist item.
- End: append a dated 3–5 line entry to `docs/session-log.md`.

## Working style
Claude Code writes the code; the user reviews it and must understand every part.
- Use plan mode for any change touching more than one file. Wait for approval.
- Before implementing, explain the *why* of each design choice in 1–2 sentences.
- When two approaches are reasonable, name both, recommend one, say why.
- After each step, say what to run to verify it and what "working" looks like.
- After finishing a module, explain it in plain English in under 10 sentences, including one
  thing that could go wrong.
- At the end of each stage, ask the user to answer the stage's "you understand it when" question
  before checking it off.

## Conventions
- Python 3.12+, `uv`, `ruff`, `pytest`, Pydantic, type hints everywhere.
- All code in `src/f1platform/`; each stage gets a `docs/stages/` writeup and a `make` target.
- Add a stage's dependencies when that stage starts, not before.
- Config via environment variables; commit `.env.example`, never `.env` or real keys.
- Calls to Claude go through the existing gateway (`GATEWAY_URL`).
- Every event and prediction carries an event time; never join on "latest" values when building
  training data.
- Keep `docs/decisions.md`, `docs/lessons-claude-code.md`, `docs/gaps.md` current. Append only.
- Tests run offline from `tests/fixtures/`; CI must never hit the network.
- One small commit per checklist item.

## Progress tracking
- Check items off in `PROGRESS.md` only when verified working.

## Avoid
- LLM frameworks (LangChain, LlamaIndex). Unsloth/TRL are fine.
- Unsloth Studio (the UI) for Stage 5 training. Stage 5 runs from code in the repo so every
  setting is reviewable; Studio is for the Stage 4b warm-up only.
- Random train/test splits. Always split by race.
- Any test-set message appearing in training data.
- Skipping writeups: a stage isn't done until its `docs/stages/` writeup is.
- Committing `data/`, FastF1 cache, model weights, adapters, or full labeled sets.

## Commands
```
uv sync            # install
make lint          # ruff check + format check
make fmt           # auto-fix
make test          # pytest
make up / down     # Redpanda (Kafka on localhost:19092, Console on :8080)
make replay RACE=2024-monza SPEED=60   # Stage 2
```

## Machines
Mac: everything except training/serving. Windows PC (RTX 5080, WSL2): local model warm-up (4b),
CUDA sim (3b), QLoRA fine-tuning and model serving (5).
