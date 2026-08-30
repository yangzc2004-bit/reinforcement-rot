"""Metrics: the five operational criteria of Reinforcement Rot, plus helpers.

From a set of trajectories (list of dicts from SolverAgent.run):
  cycle_rate   : fraction of steps spent inside detected loops
  excess_steps : steps beyond the planted shortest path (per episode)
  mill_rate    : fraction of agents whose dominant loop is shared by >= 2 agents
                 (the collective "death circle")
  escape_time  : for agents that enter a loop and later leave it, median steps
                 from loop entry to loop exit (persistence)
  death_rate   : fraction of episodes that exhaust the step budget without
                 reaching the goal;  silence = the run died with ZERO illegal
                 actions (it looped "correctly", no error ever fired)
"""
from statistics import median


def find_loops(path):
    """Return list of (start_idx, end_idx) loop segments via first-repeat scan."""
    loops = []
    first = {}
    for i, cell in enumerate(path):
        if cell in first:
            j = first[cell]
            if i - j >= 3:  # ignore immediate back-and-forth (A->B->A)
                loops.append((j, i))
            first = {}
        if cell not in first:
            first[cell] = i
    return loops


def _canon(loop_cells):
    """Canonical form of a loop (rotation- and reversal-invariant) for sharing tests."""
    seq = list(loop_cells)
    rots = [tuple(seq[i:] + seq[:i]) for i in range(len(seq))]
    rev = list(reversed(seq))
    rots += [tuple(rev[i:] + rev[:i]) for i in range(len(rev))]
    return min(rots)


def episode_stats(traj):
    path = traj['path']
    loops = find_loops(path)
    in_loop = set()
    for a, b in loops:
        in_loop.update(range(a, b))
    cycle_rate = len(in_loop) / max(1, len(path) - 1)
    excess = traj['steps'] - traj['shortest'] if traj['success'] else traj['steps']
    esc = []
    for a, b in loops:
        after = [i for i in range(b + 1, len(path)) if path[i] not in path[a:b]]
        if after:
            esc.append(after[0] - a)
    return dict(cycle_rate=cycle_rate, excess_steps=excess,
                loops=[_canon(path[a:b]) for a, b in loops],
                escape_time=median(esc) if esc else None,
                dead=not traj['success'], illegal=traj['illegal'])


def population_stats(trajs):
    eps = [episode_stats(t) for t in trajs]
    from collections import Counter
    loop_count = Counter(l for e in eps for l in e['loops'])
    shared = {l for l, n in loop_count.items() if n >= 2}
    n_agents = max(1, len({t['agent'] for t in trajs}))
    milling_agents = {t['agent'] for t, e in zip(trajs, eps)
                      if any(l in shared for l in e['loops'])}
    deaths = [e for e in eps if e['dead']]
    return dict(
        cycle_rate=sum(e['cycle_rate'] for e in eps) / len(eps),
        excess_steps=sum(e['excess_steps'] for e in eps) / len(eps),
        mill_rate=len(milling_agents) / n_agents,
        escape_time=median([e['escape_time'] for e in eps if e['escape_time'] is not None])
                    if any(e['escape_time'] is not None for e in eps) else None,
        death_rate=len(deaths) / len(eps),
        silence=all(e['illegal'] == 0 for e in deaths) if deaths else None,
    )
