"""B2 — existence of Reinforcement Rot with real LLM agents.

Protocol: n_agents solver agents run SEQUENTIALLY on the SAME maze. Every maze
has a paired no-memory control plus shared-memory conditions (the pheromone
analog). Memory evaporates after EVERY EPISODE (lam is per-episode retention).
Agent execution order ROTATES each round so no agent is systematically
first/last.

Evaporation dose is specified per ROUND for interpretability:
    round_retention = lam_episode ** n_agents   ->   lam_episode = rr ** (1/n_agents)
Config gives `round_retentions`; the derived per-episode lam is logged.

Checkpointing: every finished episode is appended to llm/b2_episodes.jsonl.
Resume granularity is one (maze, memory_condition, round_retention) CONDITION,
not one episode:
memory state is path-dependent (later agents read earlier agents' writes), so a
partially-finished condition is re-run from a fresh memory; completed conditions
are skipped entirely. If the client's max_calls fuse trips, everything completed
is kept and the manifest is stamped interrupted=true.

Run:  python -m llm.run_b2            (from the repo root)
Uses: llm/config.yaml (b2 section)
"""
import json
import os
import sys
from pathlib import Path

from llm.run_b1 import load_config
from llm.maze import Maze
from llm.client import from_config, CallLimitExceeded
from llm.memory import SharedMemory
from llm.agents import SolverAgent
from llm.metrics import population_stats
from llm.manifest import new_manifest, write_manifest

EPISODES_LOG = Path(os.environ.get(
    'LLM_B2_EPISODES', Path(__file__).parent / 'b2_episodes.jsonl'))
RESULTS = Path(os.environ.get(
    'LLM_B2_RESULTS', Path(__file__).parent / 'b2_results.json'))


def _episode_key(e):
    return (e['maze_seed'], e.get('memory_condition', 'shared'),
            e.get('round_retention'), e['round'], e['agent'])


def _conditions(b2):
    conditions = []
    if b2.get('include_no_memory_control', True):
        conditions.append(dict(memory_condition='none', round_retention=None))
    conditions.extend(
        dict(memory_condition='shared', round_retention=rr)
        for rr in b2['round_retentions']
    )
    return conditions


def _load_done():
    """Parse the JSONL episode log, tolerating ONE corrupt TAIL line.

    The log is append-only with flush after every record, so the only realistic
    corruption is a truncated last line from a crash/kill mid-write. Anything
    else (a corrupt line in the MIDDLE) means the file was damaged by something
    we don't understand — refuse to continue rather than silently drop data.
    """
    done = {}
    if not EPISODES_LOG.exists():
        return done
    raw_lines = EPISODES_LOG.read_bytes().splitlines(keepends=True)
    nonempty = [(i, line) for i, line in enumerate(raw_lines) if line.strip()]
    for pos, (i, raw_line) in enumerate(nonempty):
        try:
            e = json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            is_last_nonempty = pos == len(nonempty) - 1
            has_line_terminator = raw_line.endswith((b'\n', b'\r'))
            if is_last_nonempty and not has_line_terminator:
                truncate_at = sum(len(line) for line in raw_lines[:i])
                with EPISODES_LOG.open('r+b') as f:
                    f.truncate(truncate_at)
                print(f"WARNING: dropping truncated last line of {EPISODES_LOG} "
                      f"({len(raw_line)} bytes) — that episode will be re-run",
                      flush=True)
                break
            raise ValueError(
                f"corrupt JSONL line {i + 1} of {len(raw_lines)} in {EPISODES_LOG}; "
                f"only a truncated LAST line is recoverable — inspect the file "
                f"manually before rerunning") from exc
        done[_episode_key(e)] = e
    return done


def main():
    cfg = load_config()
    mz, b2 = cfg['maze'], cfg['b2']
    client = from_config(cfg)
    man = new_manifest('b2', cfg)
    n_agents = b2['n_agents']
    conditions = _conditions(b2)
    mazes = {k: Maze(size=mz['size'], delta=mz.get('delta', 4),
                     ring=mz.get('ring', True), seed=mz.get('seed', 0) + k,
                     min_shortest=mz.get('min_shortest', 30),
                     max_backbone=mz.get('max_backbone', 60),
                     neck_min_dist=mz.get('neck_min_dist', 6),
                     max_corridor=mz.get('max_corridor', 4),
                     deceptive_fill_roots=mz.get('deceptive_fill_roots', 0),
                     ring_entry_bias=mz.get('ring_entry_bias', 'any'))
             for k in range(b2['n_mazes'])}

    done = _load_done()
    log = open(EPISODES_LOG, 'a')
    interrupted = False
    error_message = None
    fatal_error = None
    try:
        for k, maze in mazes.items():
            seed = mz.get('seed', 0) + k
            for condition in conditions:
                memory_condition = condition['memory_condition']
                rr = condition['round_retention']
                label = memory_condition if rr is None else f'{memory_condition}@{rr}'
                needed = {(seed, memory_condition, rr, r, a)
                          for r in range(b2['n_rounds'])
                          for a in range(n_agents)}
                if needed and all(key in done for key in needed):
                    print(f"maze {k} {label}: complete, skipping", flush=True)
                    continue  # resume at condition granularity (see docstring)
                lam = (rr ** (1.0 / n_agents)
                       if memory_condition == 'shared' else None)
                mem = (SharedMemory(lam=lam, top_k=b2.get('top_k', 5))
                       if memory_condition == 'shared' else None)
                for r in range(b2['n_rounds']):
                    order = [(r + i) % n_agents for i in range(n_agents)]  # rotate
                    for pos, a in enumerate(order):
                        agent = SolverAgent(client, maze, memory=mem,
                                            breadcrumbs=b2.get('breadcrumbs', True),
                                            write_rule=(
                                                b2.get('write_rule', 'success_only')
                                                if mem is not None else 'none'),
                                            agent_id=a,
                                            navigation_ledger=b2.get(
                                                'navigation_ledger', False),
                                            navigation_guard=b2.get(
                                                'navigation_guard', False))
                        traj = agent.run(max_steps=b2['max_steps'], round_id=r,
                                         maze_seed=seed)
                        traj['round_retention'] = rr
                        traj['lam_episode'] = lam
                        traj['memory_condition'] = memory_condition
                        traj['order'] = pos
                        rec = dict(maze_seed=seed,
                                   memory_condition=memory_condition,
                                   round_retention=rr,
                                   round=r, agent=a, traj=traj)
                        log.write(json.dumps(rec) + '\n')
                        log.flush()
                        done[(seed, memory_condition, rr, r, a)] = rec
                        if mem is not None:
                            mem.evaporate()  # per-EPISODE evaporation
                        print(f"maze {k} {label} round {r} agent {a}: "
                              f"success={traj['success']} steps={traj['steps']}",
                              flush=True)
    except CallLimitExceeded as e:
        interrupted = True
        error_message = str(e)
        print(f"\nCOST FUSE: {e} — checkpoint saved, rerun to resume.", flush=True)
    except Exception as e:
        interrupted = True
        fatal_error = e
        error_message = f'{type(e).__name__}: {e}'
        print(f"\nRUN ERROR: {error_message} — checkpoint saved.", flush=True)
    finally:
        log.close()
        client.close()

    # Aggregate per (maze, memory condition, round retention) from all episodes.
    episodes = list(_load_done().values())
    results = []
    partial_conditions = []
    expected_per_condition = b2['n_rounds'] * n_agents
    for k in mazes:
        seed = mz.get('seed', 0) + k
        for condition in conditions:
            memory_condition = condition['memory_condition']
            rr = condition['round_retention']
            label = memory_condition if rr is None else f'{memory_condition}@{rr}'
            trajs = [e['traj'] for e in episodes
                     if e['maze_seed'] == seed
                     and e.get('memory_condition', 'shared') == memory_condition
                     and e.get('round_retention') == rr]
            if not trajs:
                continue
            observed = len(trajs)
            if observed != expected_per_condition:
                partial_conditions.append(dict(
                    maze=k,
                    maze_seed=seed,
                    memory_condition=memory_condition,
                    round_retention=rr,
                    n_expected=expected_per_condition,
                    n_observed=observed,
                ))
                print(f"maze {k} {label}: incomplete condition "
                      f"({observed}/{expected_per_condition}), omitted from results",
                      flush=True)
                continue
            stats = population_stats(trajs)
            print(f"maze {k} {label}: cycle_rate={stats['cycle_rate']:.2f} "
                  f"oscillation={stats['oscillation_rate']:.2f} "
                  f"mill_rate={stats['mill_rate']:.2f} "
                  f"indiv_cycle={stats['individual_cycle_rate']:.2f} "
                  f"death={stats['death_rate']:.2f} silence={stats['silence']} "
                  f"escape={stats['escape_time']}",
                  flush=True)
            results.append(dict(maze=k, maze_seed=seed,
                                memory_condition=memory_condition,
                                round_retention=rr,
                                lam_episode=(rr ** (1.0 / n_agents)
                                             if rr is not None else None),
                                complete=True,
                                n_expected=expected_per_condition,
                                n_observed=observed,
                                stats=stats, trajs=trajs))
    RESULTS.write_text(json.dumps(results, indent=1))
    man['interrupted'] = interrupted or bool(partial_conditions)
    man['error'] = error_message
    man['expected_episodes_per_condition'] = expected_per_condition
    man['conditions'] = conditions
    man['partial_conditions'] = partial_conditions
    mpath = write_manifest(man, RESULTS, client=client)
    print(f"\nAPI calls: {client.n_calls}, tokens: {client.n_tokens}")
    print("saved to", RESULTS, "| manifest:", mpath)
    if fatal_error is not None:
        raise fatal_error


if __name__ == '__main__':
    main()
