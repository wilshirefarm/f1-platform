# F1 Real-Time ML Platform: Learning Plan

Oct 2, 2026

## What it does

The system is a live race predictor, not a chatbot. While a race plays out, it re-predicts every driver's finishing position each lap, the way a broadcast win-probability graphic updates during a game.

Example output during a replay (made-up numbers):

```
LAP 18 / 53                         LAP 19 / 53  (Safety Car deployed)
Predicted finish                    Predicted finish
 1  VER   (running P1)               1  VER
 2  LEC   (running P2)               2  NOR   <- pits cheaply under Safety Car
 3  NOR   (running P4)               3  LEC
 4  HAM   (running P3)               4  HAM

LAP 31: "5 SECOND TIME PENALTY FOR CAR 16 (LEC)"
 -> LEC's predicted finish drops from P3 to P5
```

**How the pieces fit:**

- **Replay engine (Stage 2)** feeds a past race into the system event by event, as if live.
- **Data quality checks (Stage 4)** stop bad, late, or duplicate events before they reach the models.
- **Race control parser (Stage 5)** is an upstream model. It turns the officials' text messages into structured events, such as a 5-second penalty for LEC. Features like `penalty_seconds_pending` and `safety_car_active` are derived from those events.
- **Finishing-order predictor (Stage 3)** is the main, downstream model. It uses features computed only from what is known at each lap.
- **Race simulator (optional Stage 3b)** runs thousands of simulations of the remaining laps on the GPU and turns the prediction into win and podium probabilities.

Parser errors flow downstream into wrong predictions, so the parser is evaluated on its own before it is plugged in.

## Overview

Build one F1 system in five stages, plus one optional CUDA stage, each teaching one platform concept on historical race data. The end result replays any past race as a live stream, predicts the finishing order lap by lap, validates the data as it flows, and parses official race messages with a fine-tuned model running on the 5080.

The stages follow your interests in ranked order:

1. **Real-time systems:** a replay engine streams historical races as if they were live (Stage 2).
2. **Point-in-time correctness:** predictions use only what was known at each lap, proven by breaking it on purpose first (Stage 3).
3. **Fine-tuning:** a small local model learns to read race control messages, first as one-word labels, then as full JSON (Stages 5a and 5b).
4. **Data quality:** checks catch bad, late, and duplicate events in the stream (Stage 4).

Optional detour: **GPU programming**, a hand-written CUDA kernel for a Monte Carlo race simulator (Stage 3b).

Why F1 historical data: every race has a known outcome to check against, the FastF1 library loads a whole race in a few lines, and nothing depends on waiting for a race weekend.

Working style: Claude Code writes the code. You review every change and must be able to explain how each part works before moving to the next stage.

## How this maps to the old plan

This plan replaces P1–P5 as the main track. Most of the useful habits carry over; the generic projects are paused, not deleted.

| Old project | Status | What happens to it |
| --- | --- | --- |
| P0 Claude Code | Continues | Still the build tool. Keep `CLAUDE.md`, `docs/lessons-claude-code.md`, and `docs/decisions.md`. |
| P1 Kubernetes | Deferred | Everything runs locally with Docker Compose. Revisit only if you want to deploy. |
| P2 Gateway | Reused | The thin Claude gateway labels training data in Stage 5a. No new gateway work. |
| P3 Agentic RAG | Paused | Could return later as an F1 race engineer chatbot (see Later extensions). |
| P4 Evals | Folded in | Eval habits drive Stages 3, 5a, and 5b: held-out test sets, per-field scoring, error analysis. |
| P5 Self-hosting | Folded in | Stage 4b serves an off-the-shelf local model; the 5080 then trains and serves the fine-tuned model in Stage 5. |

## Setup and environment

The Mac runs everything except model training and serving, which run on the 5080 PC.

| Machine | Runs |
| --- | --- |
| Mac | Claude Code, data exploration, replay engine, stream, predictions, data quality checks, evals |
| Windows PC (RTX 5080, 16 GB) | Local model warm-up in Stage 4b; CUDA race simulator kernel in Stage 3b; QLoRA fine-tuning under WSL2; serving the fine-tuned model on the LAN in Stage 5b |

**Stack:** Python 3.12+, `uv`, `ruff`, `pytest`, Pydantic. FastF1 for race data, including race control messages. DuckDB and Parquet for stored data. Redpanda (a Kafka-compatible stream in one Docker container) from Stage 2. Numba (CUDA 12.8+) for the Stage 3b kernel. Unsloth (the code library) or Hugging Face TRL for fine-tuning; llama.cpp's `llama-server`, Ollama or vLLM for serving. Unsloth Studio (the web UI) for the Stage 4b warm-up only.

**Repo layout:** one public GitHub repo built around one Python package, because the stages share code. Each stage's writeup lives in `docs/stages/`, so the repo reads as one system with a learning log beside it.

```
f1-platform/
  README.md              front page: what it does, demo GIF, architecture, results tables
  PLAN.md                this plan, exported to Markdown
  CLAUDE.md              short instructions for Claude Code (under 100 lines)
  PROGRESS.md            stage checklists
  pyproject.toml         uv project config
  uv.lock
  Makefile               replay, test, lint, train, simulate targets
  docker-compose.yml     Redpanda
  .env.example
  .gitignore
  .github/workflows/
    ci.yml               ruff + pytest on every push
  src/f1platform/
    events/              event schemas (Pydantic)
    replay/              Stage 2
    features/            Stage 3 feature functions, shared by training and live
    predict/             Stage 3 models and live predictor
    sim/                 Stage 3b NumPy reference and CUDA simulator
    dq/                  Stage 4 checks and quarantine
    finetune/            Stage 5 labeling, training, evaluation
  notebooks/             Stage 1 exploration
  tests/
    fixtures/            a tiny slice of one race, so tests run offline
  docs/
    stages/              01-explore.md through 05b-json.md (plus 04b-local-models.md)
    decisions.md
    lessons-claude-code.md
    gaps.md
    session-log.md
  data/                  gitignored
```

**Kept out of git from the first commit:**

- `.env` and any API keys. A committed key must be rotated; deleting the file does not remove it from history.
- `data/` and the FastF1 cache. The code fetches the data, so anyone can reproduce it.
- Model weights and LoRA adapters. Share adapters on Hugging Face Hub and link them from the README.
- Full labeled training sets. Commit a 50-example sample to show the format; a script generates the rest.

**Data scope:** start with the 2023–2025 seasons, about 70 races. Enable FastF1's local cache on day one, because downloads are slow and the API rate-limits.

## Stage 1: Explore the data

Goal: know what one race looks like as data before building anything on top of it.

**Build:** load one race you remember well with FastF1. Inspect the laps table (lap times, positions, tyre compound, tyre age, pit in/out times), the results, the weather, and the race control messages. Plot each driver's position by lap and mark the pit stops and Safety Car periods.

**You understand it when** you can point at the chart and explain why each position change happened, using only the data.

**Deliverable:** `docs/stages/01-explore.md` describing each table, its grain (one row per what?), its timestamps, and anything surprising: missing laps, odd values, inconsistent names.

- [ ] FastF1 installed with caching enabled
- [ ] One race loaded; every table inspected
- [ ] Position-by-lap chart with pit stops and Safety Car marked
- [ ] Stage writeup with table descriptions and data surprises

## Stage 2: Replay engine

Goal: turn a finished race back into a live stream, so you can develop and test real-time logic any day of the week.

**Build:**

1. Convert a race's tables into one ordered list of events: lap completed, pit in, pit out, race control message, weather update. Each event has an event time and a type-specific payload.
2. Publish the events to a Redpanda topic in event-time order, at a configurable speed (1x real time, 60x, or as fast as possible).
3. Write a consumer that keeps a live race state (current order, gaps, tyre age) and prints it each lap.
4. Add fault injection switches for later: drop, duplicate, delay, or corrupt a chosen share of events.

**You understand it when** you can explain the difference between event time (when it happened on track) and processing time (when your system saw it), and why the stream must be ordered by event time.

**Deliverable:** `make replay RACE=2024-monza SPEED=60` streams a race; the consumer's final state matches the official result.

- [ ] Event schema defined with Pydantic
- [ ] Race tables converted to an ordered event list
- [ ] Redpanda running in Docker; producer publishes at configurable speed
- [ ] Consumer maintains live race state
- [ ] Final consumer state matches the official result for 3 races
- [ ] Fault injection switches (drop, duplicate, delay, corrupt)

## Stage 3: Point-in-time predictions

Goal: predict each driver's finishing position at every lap, using only information available at that lap, and see what happens when you don't.

**Build:**

1. **Features as of each lap:** position, gap to leader, tyre compound and age, pit stops so far, recent pace versus the field. One feature function serves both training and the live replay, so there is no training/serving skew.
2. **Training set:** one row per driver per lap, built from the 2023–2024 seasons. Store it in Parquet with the lap and event time on every row.
3. **Model:** a gradient-boosted model (LightGBM or XGBoost). Keep it simple; the lesson is the data, not the model.
4. **Leakage experiment:** train a second model that adds two leaked features, total pit stops in the race and final tyre compound. Compare both models offline, then on the replay.
5. **Split by race, not by row:** test on 2025 races only. Rows from the same race in both train and test are another form of leakage.
6. **Live predictions:** run the honest model inside the Stage 2 consumer and print predicted finishing order each lap.

**You understand it when** you can explain why the leaky model scored better offline and worse on the replay, in two sentences.

**Deliverable:** `docs/stages/03-predict.md` with both models' offline scores, their replay scores, and the explanation. This is your best interview story about point-in-time correctness.

- [ ] Feature function shared between training and live replay
- [ ] Training set built and stored with event times
- [ ] Honest model trained and evaluated on held-out 2025 races
- [ ] Leaky model trained; offline vs replay gap documented
- [ ] Live predictions running inside the replay consumer
- [ ] Stage writeup with results and explanation

## Stage 3b (optional): CUDA race simulator

Goal: write GPU code by hand for the first time, by turning the predicted order into probabilities with a Monte Carlo simulation that runs on the 5080.

**What it does:** at any lap, take the current race state (positions, gaps, tyre age, each driver's pace from Stage 3) and simulate the rest of the race many times. Each simulation adds random lap-time noise, pit stop time loss, and a chance of a Safety Car. Counting outcomes across all simulations gives each driver's chance of winning, finishing on the podium, or scoring points.

```
LAP 19 / 53  (Safety Car)
VER   P(win) 61%   P(podium) 94%
NOR   P(win) 22%   P(podium) 81%
```

Every simulation is independent, so each one runs on its own GPU thread. That makes this a clean first CUDA task.

**Build:**

1. **CPU reference in NumPy** on the Mac. Keep it simple and correct; it is what the GPU results are checked against.
2. **CUDA kernel with Numba** (`@cuda.jit`) on the 5080: threads, blocks and grids; copying data between CPU and GPU memory; one random number generator per thread.
3. **Check correctness:** GPU finishing distributions match the CPU version within sampling noise.
4. **Benchmark CPU vs GPU** at 1k, 10k, 100k, and 1M simulations, and find the crossover point.
5. **One optimization:** keep race state on the GPU between laps instead of copying it back and forth each lap, and measure the difference.
6. **Wire it into the replay:** print win and podium probabilities alongside the predicted order each lap.

**Setup notes:** CUDA runs only on the 5080 under WSL2; the Mac has no NVIDIA GPU, so the NumPy reference is written there and the kernel on the PC. The 5080 is a Blackwell card and needs CUDA 12.8 or newer and recent Numba builds; older versions may not recognize it. To write real CUDA C++ instead of Python, put the kernel in a `.cu` file and launch it with CuPy's `RawKernel`. The concepts are the same.

**You understand it when** you can explain why the GPU loses at 1,000 simulations and wins at 1,000,000.

**Deliverable:** `docs/stages/03b-simulate.md` with the benchmark table (CPU vs GPU time at each simulation count), the crossover point, the effect of the optimization, and a replay showing probabilities shift at a Safety Car.

| Simulations | CPU (NumPy) | GPU (CUDA) | Speedup |
| --- | --- | --- | --- |
| 1,000 |  |  |  |
| 10,000 |  |  |  |
| 100,000 |  |  |  |
| 1,000,000 |  |  |  |

- [ ] NumPy reference simulator on the Mac
- [ ] CUDA toolkit and Numba working on the 5080 under WSL2
- [ ] CUDA kernel written; results match the CPU version
- [ ] Benchmark table filled in; crossover point found
- [ ] State-on-GPU optimization measured
- [ ] Probabilities printed during replay
- [ ] Stage writeup complete

## Stage 4: Data quality on the stream

Goal: catch bad data before it reaches the predictions, using the fault injection switches from Stage 2 to prove each check works.

**Checks to build:**

| Check | Example of what it catches |
| --- | --- |
| Schema | A lap event missing its driver number |
| Range | A lap time of 12 seconds or 900 seconds |
| Sequence | Driver jumps from lap 14 to lap 16 |
| Duplicate | The same lap event arriving twice |
| Late arrival | A lap 10 event arriving after lap 15 has been processed |
| Consistency | Two drivers reported in the same position |

**Build:** a validation step between the stream and the predictor. Bad events go to a quarantine topic with the reason attached; good events continue. Track counts per check, per race.

**Decide and document:** what happens to a late event? Options are drop it, reprocess affected laps, or correct the prediction going forward. Record the choice in `docs/decisions.md`.

**You understand it when** you can inject each fault type and predict which check fires before you run it.

**Deliverable:** a replay with 5% injected faults where every fault lands in quarantine and the predictions match a clean replay.

- [ ] Validation step with all six checks
- [ ] Quarantine topic with failure reasons
- [ ] Per-check counts reported after each replay
- [ ] Late-event policy decided and documented
- [ ] Faulty replay produces the same predictions as a clean one

## Stage 4b (optional, any time before Stage 5): Local model warm-up

Goal: learn how running a model locally works (quantization, VRAM limits, context size, tokens per second) before your own fine-tuned model depends on it. The 5080 is idle during Stages 1–4, so this can run in parallel. Timebox it to a weekend.

**Build:**

1. **Serve a strong local model:** run Qwen3.8-27B with `llama-server`, using an Unsloth GGUF that fits fully on the GPU (Q3_K_M, ~13.5 GB, or IQ4_XS, ~15 GB with a short context). Start at 16k context.
2. **Call it from the Mac** over the LAN through its OpenAI-compatible API. This is a rehearsal of the Stage 5b serving setup.
3. **Measure:** tokens per second, VRAM use (`nvidia-smi`), and the largest context that still fits. Turn off "CUDA Sysmem Fallback Policy" in the NVIDIA Control Panel so a model that doesn't fit errors out instead of silently slowing down.
4. **Try one toy fine-tune in Unsloth Studio:** pick a small model and a tiny dataset, run a QLoRA fine-tune through the UI, export it to GGUF, and load it in `llama-server`. The goal is to see the whole workflow once, end to end. Stage 5's real fine-tunes are written as code in the repo, not done in the UI.

**You understand it when** you can explain why a ~16 GB file doesn't fit on a 16 GB card, and why a dense model slows down sharply when part of it spills into system RAM while a MoE model doesn't.

**Deliverable:** `docs/stages/04b-local-models.md` with a small table (model, quant, file size, context, VRAM used, tokens per second) and notes on what the Studio fine-tune did at each step.

- [ ] Qwen3.8-27B served with `llama-server`, fully on the GPU
- [ ] Called from the Mac over the LAN
- [ ] Speed, VRAM and max context recorded
- [ ] Toy fine-tune run and exported in Unsloth Studio
- [ ] Stage writeup complete

## Stage 5a: Fine-tune a message classifier

Goal: learn the full fine-tuning workflow on the simplest possible task. The model reads one race control message and outputs one label.

**Example:** input `FIA STEWARDS: 5 SECOND TIME PENALTY FOR CAR 16 (LEC) - CAUSING A COLLISION`, output `penalty`.

**Build:**

1. **Fix the label set first:** for example `penalty`, `investigation`, `track_limits`, `flag`, `safety_car`, `drs`, `other`. Write one sentence defining each. Changing labels later means relabeling everything.
2. **Collect messages** from FastF1 for 2023–2025.
3. **Label them with Claude** through your existing gateway, then hand-check a random 200. Record the agreement rate.
4. **Split by race:** hold out every message from about 10 races as the test set. The model never trains on them.
5. **Baselines:** score four things on the test set. A rules-based classifier (regex), the base 7–8B model with a prompt, Qwen3.8-27B with the same prompt (served locally as in Stage 4b), and Claude.
6. **Fine-tune** the same 7–8B model with QLoRA on the 5080, using the Unsloth library in code under `src/f1platform/finetune/`, then score it on the same test set. Don't try to fine-tune the 27B: QLoRA training needs far more memory than running a model, and it won't fit in 16 GB.

**Results table to fill in:**

| Approach | Accuracy | Cost per 1,000 messages | Latency per message |
| --- | --- | --- | --- |
| Regex rules |  |  |  |
| Base 8B model, prompted |  |  |  |
| Qwen3.8-27B, prompted (local) |  |  |  |
| Fine-tuned 8B model |  |  |  |
| Claude |  |  |  |

**You understand it when** you can explain what changed inside the model during training, whether fine-tuning beat the regex baseline, and whether a fine-tuned 8B beat a 27B model three times its size that wasn't fine-tuned. If the regex wins, that is a valid and useful finding: it shows when fine-tuning is worth it.

**Deliverable:** the results table filled in, plus `docs/stages/05a-classifier.md` explaining the training data format, the training settings, and the errors the fine-tuned model still makes.

- [ ] Label set and definitions written down
- [ ] Messages collected and labeled; 200 hand-checked
- [ ] Test set held out by race
- [ ] Four baselines scored
- [ ] QLoRA fine-tune runs on the 5080
- [ ] Results table and stage writeup complete

## Stage 5b: Fine-tune full JSON extraction

Goal: extend the model from one label to a full structured record, then plug it into the live system so penalties change the predictions.

**Example output** for the same penalty message:

```json
{"type": "penalty", "driver": "LEC", "penalty_seconds": 5, "reason": "causing_collision", "lap": null}
```

**Build:**

1. **Fix the JSON schema first** as a Pydantic model. Reuse the 5a labels as the `type` field.
2. **Extend the labels** with Claude for the new fields, and hand-check 200 again.
3. **Use the same held-out test races as 5a** so results are directly comparable.
4. **Fine-tune** the 8B model on the JSON task.
5. **Score per field:** valid JSON rate, then accuracy for each field separately.
6. **Compare `type` accuracy with the 5a model.** Did asking for more make the simple part worse?
7. **Quantization check:** export the fine-tuned model to GGUF at Q4_K_M, Q8_0 and 16-bit, and score each on the held-out races. Record valid-JSON rate, per-field accuracy, file size and tokens per second, then pick the one to serve and note why in `docs/decisions.md`.
8. **Serve the model** on the 5080 over the LAN (`llama-server` or Ollama first, vLLM if you want to compare).
9. **Wire it into the replay:** race control messages pass through the model, and a time penalty adjusts that driver's predicted finishing position.

**You understand it when** you can name the field the model gets wrong most often and explain why, and say how much accuracy each quantization level cost.

| Quant | File size | Valid JSON | Worst field accuracy | Tokens/s |
| --- | --- | --- | --- | --- |
| 16-bit |  |  |  |  |
| Q8_0 |  |  |  |  |
| Q4_K_M |  |  |  |  |

**Deliverable:** a replay of a race with a known penalty where the predicted order shifts at the moment the penalty is announced. Write the per-field results and error analysis in `docs/stages/05b-json.md`.

- [ ] JSON schema defined; labels extended and hand-checked
- [ ] JSON fine-tune trained on the 5080
- [ ] Valid-JSON rate and per-field accuracy measured
- [ ] `type` accuracy compared with the 5a model
- [ ] Quant levels compared; choice recorded in decisions.md
- [ ] Model served on the LAN
- [ ] Penalties adjust live predictions during replay

## Later extensions

Pick these up only after Stage 5b, and only the ones that still interest you.

| Extension | What it adds | Builds on |
| --- | --- | --- |
| Live races | Swap the replay source for live timing data (check OpenF1's current access terms) | Stages 2–4 |
| Team radio | Transcribe radio with Whisper; fine-tune a classifier for tyre complaints, damage, strategy calls | Stage 5 |
| Race engineer chatbot | The paused P3 reborn: answers fan questions from live state; its traces feed a data flywheel | Stages 3, 5, old P3/P4 |
| ATC + ADS-B | Same patterns on aviation: transcribed controller instructions joined to aircraft positions | Everything |
| NHTSA defect early warning | Classify vehicle complaints; detect spikes before recalls; the most startup-shaped option | Stages 3–5 |
| Kubernetes | Deploy the whole system if a role you target expects it | Old P1 |

## Getting started with Claude Code

Commit this file as `PLAN.md` at the repo root; it is the source of truth Claude Code reads. `CLAUDE.md` holds only a condensed version of the next section plus a pointer to `PLAN.md` and `PROGRESS.md`.

**First-session prompt:**

> Read PLAN.md. Propose a CLAUDE.md under 100 lines from its "Instructions for Claude Code" section, create PROGRESS.md from the stage checklists, and set up the repo skeleton from the Setup section: pyproject.toml, Makefile, docker-compose.yml, .gitignore, .env.example, and a CI workflow running ruff and pytest. Use plan mode and wait for my approval before writing files.

**Every later session:** "Read PROGRESS.md and summarize where we are, then propose the next checklist item."

**GitHub habits:**

- One small commit per checklist item, with a clear message.
- CI runs `ruff` and `pytest` on every push, using offline test fixtures only.
- Keep the top-level `README.md` current: a demo GIF of a race replaying with live predictions, the Stage 3 leakage results, and the Stage 5a results table.

## Instructions for Claude Code

Claude Code writes the code; the user reviews it and must understand every part. Read this section at the start of every session.

**Working style**

- Use plan mode for any change touching more than one file. Show the plan and wait for approval.
- Before implementing, explain the *why* of each design choice in one or two sentences.
- When two approaches are reasonable, name both, recommend one, and say why.
- After each step, say what to run to verify it and what "working" looks like.
- After finishing a module, explain how it works in plain English in under 10 sentences, including one thing that could go wrong with it.
- At the end of each stage, ask the user to explain the stage's "you understand it when" question back before checking it off.

**Conventions**

- Python 3.12+, `uv`, `ruff`, `pytest`, Pydantic, type hints everywhere.
- All code lives in the `src/f1platform/` package; each stage gets a writeup in `docs/stages/` and a `make` target to run it.
- Config via environment variables; commit `.env.example`, never `.env` or real keys.
- Calls to Claude go through the existing gateway.
- Every event and prediction carries an event time; never join on "latest" values when building training data.
- Keep `docs/decisions.md`, `docs/lessons-claude-code.md`, and `docs/gaps.md` up to date. Append, don't rewrite.

**Progress tracking**

- Maintain `PROGRESS.md` with the checklists from this plan. Check items off only when verified working.
- At session start, read `PROGRESS.md` and `docs/decisions.md` and summarize where things stand in 3–5 lines.
- At session end, append a dated 3–5 line entry to `docs/session-log.md`.

**Things to avoid**

- No LLM frameworks (LangChain, LlamaIndex). Fine-tuning libraries (Unsloth, TRL) are fine.
- Stage 5 training runs from code in the repo, not from Unsloth Studio's UI, so every setting is reviewable and reproducible. Studio is for the Stage 4b warm-up only.
- No random train/test splits. Always split by race.
- Never let a test-set message appear in training data.
- Don't skip the writeups. A stage isn't done until its `docs/stages/` writeup is.
- Don't overwrite this plan or `CLAUDE.md` wholesale; propose edits.
