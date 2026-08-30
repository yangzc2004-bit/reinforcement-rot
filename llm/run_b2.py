"""B2 — existence of Reinforcement Rot with real LLM agents.

Protocol: n_agents solver agents run SEQUENTIALLY on the SAME maze, sharing one
memory (the pheromone analog). Memory evaporates after EVERY EPISODE
(lam is per-episode retention). Agent execution order ROTATES each round so no
agent is systematically first/last.

Evaporation dose is specified per ROUND for interpretability:
    round_retention = lam_episode ** n_agents   ->   lam_episode = rr ** (1/n_agents)
Config gives `round_retentions`; the derived per-episode lam is logged.

Checkpointing: every finished episode is appended to llm/b2_episodes.jsonl.
Resume granularity is one (maze, round_retention) CONDITION, not one episode:
memory state is path-dependent (later agents read earlier agents' writes), so a
partially-finished condition is re-run from a fresh memory; completed conditions
are skipped entirely. If the client's max_calls fuse trips, everything completed
is kept and the manifest is stamped interrupted=true.

Run:  python -m llm.run_b2            (from the repo root)
Uses: llm/config.yaml (b2 section)
"""
import json
import sys
from pathlib import Path

from llm.run_b1 import load_config
from llm.maze import Maze
from llm.client import from_config, CallLimitExceeded
from llm.memory import SharedMemory
from llm.agents import SolverAgent
from llm.metrics import population_stats
from llm.manifest import new_manifest, write_manifest

EPISODES_LOG = Path(__file__).parent / 'b2_episodes.jsonl'
RESULTS = Path(__file__).parent / 'b2_results.json'


def _episode_key(e):
    return (e['maze_seed'], e['round_retention'], e['round'], e['agent'])


def _load_done():
    done = {}
    if EPISODES_LOG.exists():
        for line in EPISODES_LOG.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                done[_episode_key(e)] = e
    return done


def main():
    cfg = load_config()
    mz, b2 = cfg['maze'], cfg['b2']
    client = from_config(cfg)
    man = new_manifest('b2', cfg)
    n_agents = b2['n_agents']
    mazes = {k: Maze(size=mz['size'], delta=mz.get('delta', 4),
                     ring=mz.get('ring', True), seed=mz.get('seed', 0) + k,
                     min_shortest=mz.get('min_shortest', 30))
             for k in range(b2['n_mazes'])}

    done = _load_done()
    log = open(EPISODES_LOG, 'a')
    interrupted = False
    try:
        for k, maze in mazes.items():
            seed = mz.get('seed', 0) + k
            for rr in b2['round_retentions']:
                needed = {(seed, rr, r, a) for r in range(b2['n_rounds'])
                          for a in range(n_agents)}
                if needed and all(key in done for key in needed):
                    print(f"maze {k} rr={rr}: complete, skipping", flush=True)
                    continue  # resume at condition granularity (see docstring)
                lam = rr ** (1.0 / n_agents)  # per-episode retention
                mem = SharedMemory(lam=lam, top_k=b2.get('top_k', 5))
                for r in range(b2['n_rounds']):
                    order = [(r + i) % n_agents for i in range(n_agents)]  # rotate
                    for pos, a in enumerate(order):
                        agent = SolverAgent(client, maze, memory=mem,
                                            breadcrumbs=b2.get('breadcrumbs', True),
                                            write_rule=b2.get('write_rule', 'success_only'),
                                            agent_id=a)
                        traj = agent.run(max_steps=b2['max_steps'], round_id=r,
                                         maze_seed=seed)
                        traj['round_retention'] = rr
                        traj['lam_episode'] = lam
                        traj['order'] = pos
                        rec = dict(maze_seed=seed, round_retention=rr,
                                   round=r, agent=a, traj=traj)
                        log.write(json.dumps(rec) + '\n')
                        log.flush()
                        done[(seed, rr, r, a)] = rec
                        mem.evaporate()  # per-EPISODE evaporation
                        print(f"maze {k} rr={rr} round {r} agent {a}: "
                              f"success={traj['success']} steps={traj['steps']}",
                              flush=True)
    except CallLimitExceeded as e:
        interrupted = True
        print(f"\nCOST FUSE: {e} — checkpoint saved, rerun to resume.", flush=True)
    finally:
        log.close()
        client.close()

    # ---- aggregate per (maze, round_retention) from ALL episodes on disk ----
    episodes = list(_load_done().values())
    results = []
    for k in mazes:
        seed = mz.get('seed', 0) + k
        for rr in b2['round_retentions']:
            trajs = [e['traj'] for e in episodes
                     if e['maze_seed'] == seed and e['round_retention'] == rr]
            if not trajs:
                continue
            stats = population_stats(trajs)
            print(f"maze {k} rr={rr}: cycle_rate={stats['cycle_rate']:.2f} "
                  f"mill_rate={stats['mill_rate']:.2f} "
                  f"indiv_cycle={stats['individual_cycle_rate']:.2f} "
                  f"death={stats['death_rate']:.2f} silence={stats['silence']} "
                  f"escape={stats['escape_time']}",
                  flush=True)
            results.append(dict(maze=k, maze_seed=seed, round_retention=rr,
                                lam_episode=rr ** (1.0 / n_agents),
                                stats=stats, trajs=trajs))
    RESULTS.write_text(json.dumps(results, indent=1))
    man['interrupted'] = interrupted
    mpath = write_manifest(man, RESULTS, client=client)
    print(f"\nAPI calls: {client.n_calls}, tokens: {client.n_tokens}")
    print("saved to", RESULTS, "| manifest:", mpath)


if __name__ == '__main__':
    main()
