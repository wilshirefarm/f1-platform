# f1-platform

A live race predictor for Formula 1. It replays any past race as a live event stream and
re-predicts every driver's finishing position each lap, like a broadcast win-probability graphic.

- **Replay engine:** streams a historical race event by event through Redpanda.
- **Data quality checks:** stop bad, late, or duplicate events before they reach the models.
- **Race control parser:** a fine-tuned local model turns officials' messages into structured events.
- **Finishing-order predictor:** uses only what was known at each lap.
- **Race simulator (optional):** CUDA Monte Carlo for win and podium probabilities.

See [PLAN.md](PLAN.md) for the full plan and [PROGRESS.md](PROGRESS.md) for status.

## Quickstart

```
uv sync
make test
make up        # Redpanda on localhost:19092, Console on http://localhost:8080
```
