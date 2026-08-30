# L3 — LLM agents in the maze

This layer replays the L2 pheromone world with real LLM agents:
the shared memory (`memory.py`) is the pheromone field, retrieval top-k is the
follow rule, `lam` is evaporation, and `metrics.py` implements the five
operational criteria of Reinforcement Rot.

## Quick start

```bash
cd llm
cp config.example.yaml config.yaml
# edit config.yaml: model.name / api_key / base_url  (any OpenAI-compatible API)
#   — or set name: "mock" for a free offline smoke test of the full pipeline
cd ..
python -m llm.run_b1     # capability gate (single agent, no memory)
python -m llm.run_b2     # existence experiment (shared memory, lam contrast)
```

No dependencies beyond PyYAML (optional: config.yaml also accepts JSON).

## Experiment map

| ID | Question | Script | Status |
|----|----------|--------|--------|
| B1 | Can one agent solve the mazes WITHOUT memory? (gate: 85–90%) | `run_b1.py` | runnable |
| B2 | Does a shared loop emerge with memory + no evaporation? | `run_b2.py` | runnable |
| B3 | Perturbation injection dose-response (the heart) | planned | — |
| B4 | Evaporation dose-response λ ∈ {1, .99, .97, .95, .9} | `run_b2.py` (`b2.lams`) | runnable |
| B5 | Write-rule ablation (universal / success-only / efficient-success-only) | `run_b2.py` (`b2.write_rule`) | runnable |
| B6 | N ∈ {1,2,4,8} | `run_b2.py` (`b2.n_agents`) | runnable |
| B7 | Persistence over T=20 rounds | `run_b2.py` (`b2.n_rounds`) | runnable |

B3–B7 reuse the B2 engine via config; dedicated scripts land as the campaign
progresses. C1 (cross-model probes) = rerun with a different `model` block.
