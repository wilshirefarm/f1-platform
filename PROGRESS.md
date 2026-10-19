# Progress

Check items off only when verified working. Checklists copied from PLAN.md.

## Setup

- [x] Repo skeleton: pyproject, Makefile, docker-compose, CI, package layout
- [x] CLAUDE.md and PROGRESS.md
- [ ] Git identity configured; first commit made
- [ ] GitHub repo created; CI green

## Stage 1: Explore the data

- [ ] FastF1 installed with caching enabled
- [ ] One race loaded; every table inspected
- [ ] Position-by-lap chart with pit stops and Safety Car marked
- [ ] Stage writeup with table descriptions and data surprises

## Stage 2: Replay engine

- [ ] Event schema defined with Pydantic
- [ ] Race tables converted to an ordered event list
- [ ] Redpanda running in Docker; producer publishes at configurable speed
- [ ] Consumer maintains live race state
- [ ] Final consumer state matches the official result for 3 races
- [ ] Fault injection switches (drop, duplicate, delay, corrupt)

## Stage 3: Point-in-time predictions

- [ ] Feature function shared between training and live replay
- [ ] Training set built and stored with event times
- [ ] Honest model trained and evaluated on held-out 2025 races
- [ ] Leaky model trained; offline vs replay gap documented
- [ ] Live predictions running inside the replay consumer
- [ ] Stage writeup with results and explanation

## Stage 3b (optional): CUDA race simulator

- [ ] NumPy reference simulator on the Mac
- [ ] CUDA toolkit and Numba working on the 5080 under WSL2
- [ ] CUDA kernel written; results match the CPU version
- [ ] Benchmark table filled in; crossover point found
- [ ] State-on-GPU optimization measured
- [ ] Probabilities printed during replay
- [ ] Stage writeup complete

## Stage 4: Data quality on the stream

- [ ] Validation step with all six checks
- [ ] Quarantine topic with failure reasons
- [ ] Per-check counts reported after each replay
- [ ] Late-event policy decided and documented
- [ ] Faulty replay produces the same predictions as a clean one

## Stage 4b (optional, any time before Stage 5): Local model warm-up

- [ ] Qwen3.8-27B served with `llama-server`, fully on the GPU
- [ ] Called from the Mac over the LAN
- [ ] Speed, VRAM and max context recorded
- [ ] Toy fine-tune run and exported in Unsloth Studio
- [ ] Stage writeup complete

## Stage 5a: Fine-tune a message classifier

- [ ] Label set and definitions written down
- [ ] Messages collected and labeled; 200 hand-checked
- [ ] Test set held out by race
- [ ] Four baselines scored
- [ ] QLoRA fine-tune runs on the 5080
- [ ] Results table and stage writeup complete

## Stage 5b: Fine-tune full JSON extraction

- [ ] JSON schema defined; labels extended and hand-checked
- [ ] JSON fine-tune trained on the 5080
- [ ] Valid-JSON rate and per-field accuracy measured
- [ ] `type` accuracy compared with the 5a model
- [ ] Quant levels compared; choice recorded in decisions.md
- [ ] Model served on the LAN
- [ ] Penalties adjust live predictions during replay
