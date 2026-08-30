# L3 — LLM agents in the maze

This layer replays the L2 pheromone world with real LLM agents:
the shared memory (`memory.py`) is the pheromone field, retrieval top-k is a
candidate-set truncation (the model's effective follow sharpness β_eff is
MEASURED by `calib_beta.py`, not assumed), per-episode evaporation is the
evaporation operator, and `metrics.py` implements the five operational
criteria of Reinforcement Rot.

The maze (`maze.py`) is a CONTROLLED-topology 15x15 grid: a designed shortest
path (>= 30 steps), one detour route longer by exactly delta (delta EVEN —
bipartite parity), and one dangling single-neck ring (entrance == exit; the
start cell is never in the ring). Cyclomatic number == 2, all invariants
asserted at construction and re-checked by `experiments/validate_mazes.py`.

## Quick start

```bash
cd llm
cp config.example.yaml config.yaml
# edit config.yaml: model.name / api_key / base_url  (any OpenAI-compatible API)
#   — or set name: "mock" for a free offline smoke test of the full pipeline
cd ..
python -m llm.run_b1       # capability gate (single agent, no memory, 85-90%)
python -m llm.calib_beta   # beta_eff model card (cheap; run before B2)
python -m llm.run_b2       # existence experiment (resumable; per-episode JSONL)
```

Dependencies: PyYAML only (optional — config.yaml also accepts JSON).
Every run writes `<output>.manifest.json` (git SHA, full config, seeds, timing,
output hash) and supports clean interruption via the `max_calls` cost fuse.

## Experiment map

| ID | Question | Script | Status |
|----|----------|--------|--------|
| B1 | Can one agent solve the mazes WITHOUT memory? (gate: 85–90%) | `run_b1.py` | runnable |
| β | How sharply does the model follow memory weight? (β_eff + CI) | `calib_beta.py` | runnable |
| B2 | Does a shared loop emerge with memory + no evaporation? | `run_b2.py` | runnable |
| B3 | Perturbation injection dose-response δ ∈ {2,4,6} (the heart) | planned | — |
| B4 | Evaporation dose-response (per-round retention grid) | `run_b2.py` (`b2.round_retentions`) | runnable |
| B5 | Write-rule ablation (2×2: mechanical/authored × success-gated/universal) | `run_b2.py` (`b2.write_rule`) | runnable |
| B6 | N ∈ {1,2,4,8} | `run_b2.py` (`b2.n_agents`) | runnable |
| B7 | Persistence over T=20 rounds | `run_b2.py` (`b2.n_rounds`) | runnable |

B3–B7 reuse the B2 engine via config; dedicated scripts land as the campaign
progresses. C1 (cross-model probes) = rerun with a different `model` block.
