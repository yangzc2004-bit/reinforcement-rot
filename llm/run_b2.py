"""B2 — existence of Reinforcement Rot with real LLM agents.

Protocol: n_agents solver agents run SEQUENTIALLY on the SAME maze, sharing one
memory (the pheromone analog). After each round, memory evaporates by lam.
We track the five criteria across rounds and look for the signature: a shared
loop (mill) emerging WITHOUT any illegal action, ending in budget-exhaustion
death — zero progress, zero errors.

Run:  python -m llm.run_b2            (from the repo root)
Uses: llm/config.yaml (b2 section)
"""
import json
import sys
from pathlib import Path

from llm.run_b1 import load_config
from llm.maze import Maze
from llm.client import from_config
from llm.memory import SharedMemory
from llm.agents import SolverAgent
from llm.metrics import episode_stats, population_stats


def main():
    cfg = load_config()
    mz, b2 = cfg['maze'], cfg['b2']
    client = from_config(cfg)
    all_trajs = []
    for k in range(b2['n_mazes']):
        maze = Maze(size=mz['size'], detour_openings=mz['detour_openings'],
                    ring=mz.get('ring', True), seed=mz.get('seed', 0) + k)
        for lam in b2['lams']:
            mem = SharedMemory(lam=lam, top_k=b2.get('top_k', 5))
            trajs = []
            for r in range(b2['n_rounds']):
                for a in range(b2['n_agents']):
                    agent = SolverAgent(client, maze, memory=mem,
                                        breadcrumbs=b2.get('breadcrumbs', True),
                                        write_rule=b2.get('write_rule', 'success_only'),
                                        agent_id=a)
                    trajs.append(agent.run(max_steps=b2['max_steps'], round_id=r))
                mem.evaporate()
            stats = population_stats(trajs)
            print(f"maze {k} lam={lam}: cycle_rate={stats['cycle_rate']:.2f} "
                  f"mill_rate={stats['mill_rate']:.2f} death={stats['death_rate']:.2f} "
                  f"silence={stats['silence']} escape={stats['escape_time']}",
                  flush=True)
            all_trajs.append(dict(maze=k, lam=lam, stats=stats, trajs=trajs))
    out = Path(__file__).parent / 'b2_results.json'
    out.write_text(json.dumps(all_trajs, indent=1))
    print(f"\nAPI calls: {client.n_calls}, tokens: {client.n_tokens}")
    print("saved to", out)


if __name__ == '__main__':
    main()
