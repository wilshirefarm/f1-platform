# Decisions

Append-only. Each entry: date, decision, why, alternatives considered.

## 2026-10-16: Python 3.12 via uv
Pinned with `.python-version`; `requires-python = ">=3.12"`. Numba/CUDA and fine-tuning wheels lag new Python releases, so 3.12 is the safest floor. Alternative: 3.13 (system Python), rejected for later wheel risk.

## 2026-10-16: src layout
Package lives in `src/f1platform/` so tests import the installed package, not stray files in the working directory. Alternative: flat layout, simpler but can hide packaging mistakes.

## 2026-10-16: Add dependencies per stage
Core install is only `pydantic`; each stage adds its own deps when it starts. Keeps CI fast and avoids CUDA/Unsloth packages that cannot install on the Mac or in CI. Alternative: everything up front, rejected.

## 2026-10-16: Pinned Redpanda single node
`redpanda:v24.2.7` in dev-container mode, Kafka on `localhost:19092`, Console on `:8080`. Pinned so replays stay reproducible.

## 2026-10-16: CI uses `uv sync --locked`
CI fails if `uv.lock` is stale, so local and CI environments match.

## 2026-10-18: Plan update: Stage 4b and quantization check
PLAN.md added an optional Stage 4b (serve Qwen3.8-27B with `llama-server`, toy Unsloth Studio fine-tune), a fourth 5a baseline (the 27B, prompted), and a 5b quantization comparison. Setup changes: `LOCAL_LLM_URL` kept separate from `MODEL_SERVER_URL` so the off-the-shelf and fine-tuned models can't be confused; Stage 5 training stays in code, Studio only for 4b.
