"""B1 — capability gate.

Question: WITHOUT any shared memory, can a single agent solve these mazes?
The gate for proceeding to B2+ is a no-memory success rate of ~85-90%
(mazes too easy -> no room for rot; too hard -> death is attributable to
capability, not reinforcement).

Run:  python -m llm.run_b1            (from the repo root)
Uses: llm/config.yaml (copy config.example.yaml and fill in model + API)
"""
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # tiny fallback: accept JSON in config.yaml
    yaml = None

from llm.maze import Maze
from llm.client import from_config
from llm.agents import SolverAgent


def load_config():
    p = Path(__file__).parent / 'config.yaml'
    if not p.exists():
        sys.exit("llm/config.yaml not found — copy config.example.yaml and fill in "
                 "your model name, api_key and base_url.")
    text = p.read_text()
    if yaml:
        return yaml.safe_load(text)
    return json.loads(text)


def main():
    cfg = load_config()
    mz, b1 = cfg['maze'], cfg['b1']
    client = from_config(cfg)
    results = []
    for k in range(b1['n_mazes']):
        maze = Maze(size=mz['size'], detour_openings=mz['detour_openings'],
                    ring=mz.get('ring', True), seed=mz.get('seed', 0) + k)
        agent = SolverAgent(client, maze, memory=None,
                            breadcrumbs=b1.get('breadcrumbs', True),
                            write_rule='none', agent_id=0)
        traj = agent.run(max_steps=b1['max_steps'])
        results.append(traj)
        print(f"maze {k}: success={traj['success']} steps={traj['steps']} "
              f"shortest={traj['shortest']} illegal={traj['illegal']}", flush=True)
    sr = sum(t['success'] for t in results) / len(results)
    print(f"\nB1 no-memory success rate: {sr:.1%} over {len(results)} mazes")
    print(f"API calls: {client.n_calls}, tokens: {client.n_tokens}")
    if 0.85 <= sr <= 0.90:
        print("GATE PASSED (85-90%): difficulty is calibrated, proceed to B2.")
    else:
        print("GATE NOT MET: adjust maze.size / detour_openings / max_steps "
              "(or breadcrumbs) and rerun.")
    out = Path(__file__).parent / 'b1_results.json'
    out.write_text(json.dumps(results, indent=1))
    print("trajectories saved to", out)


if __name__ == '__main__':
    main()
