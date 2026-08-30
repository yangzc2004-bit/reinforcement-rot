"""Behavioral metrics for L3 trajectories.

Loop detection is LOOP-ERASURE (the trajectory analog of cycle counting): scan
the path, and whenever the walk revisits a cell, the segment between the two
visits is one loop traversal; erase it and continue. This decomposes any walk
into a loop-free skeleton plus an ordered list of loop traversals.

The five operational criteria of Reinforcement Rot (population level):
  cycle_rate            fraction of episodes containing >= 1 loop
  excess_over_shortest  (steps - shortest) / shortest, by success/failure
  mill_rate             fraction of (maze, round) with >= 2 DISTINCT agents
                        trapped on the SAME canonical loop IN THE SAME ROUND
  escape_time           steps from first trap entry to final loop exit
                        (trap = >= 2 CONSECUTIVE traversals of one loop)
  death_rate            budget exhausted without reaching the goal
  silence               fraction of failed episodes with zero memory writes

A canonical loop is invariant under rotation and reversal (cyclic rotations of
both the cell sequence and its reverse; lexicographic minimum), so two agents
circling the same ring in opposite directions still match.
"""
from collections import deque

MIN_LOOP = 3  # a 2-step back-and-forth (u,v,u) is hesitation, not circling


def _canon(cyc):
    """Canonical form of a closed loop given as an OPEN cell sequence."""
    rots = [tuple(cyc[i:] + cyc[:i]) for i in range(len(cyc))]
    rev = list(reversed(cyc))
    rots += [tuple(rev[i:] + rev[:i]) for i in range(len(rev))]
    return min(rots)


def extract_loops(path):
    """Loop-erased decomposition. Returns a list of dicts with keys
    canon / cells / t_entry / t_close in detection order."""
    path = [tuple(c) for c in path]  # JSON reload gives lists
    loops = []
    stack = []
    first_seen = {}
    for t, cell in enumerate(path):
        if cell in first_seen:
            i = first_seen[cell]
            cyc = stack[i:]
            if len(cyc) >= MIN_LOOP:
                loops.append(dict(canon=_canon(cyc), cells=cyc,
                                  t_entry=i, t_close=t))
            del stack[i:]
            first_seen = {c: k for k, c in enumerate(stack)}
        stack.append(cell)
        first_seen[cell] = len(stack) - 1
    return loops


def _trap_episode(loops, path):
    """First canonical loop with a CONSECUTIVE repeat (run length >= 2) in
    detection order — an interleaved A,B,A pattern is NOT a trap."""
    run_start = None
    for k in range(1, len(loops) + 1):
        cont = k < len(loops) and loops[k]['canon'] == loops[k - 1]['canon']
        if cont and run_start is None:
            run_start = k - 1
        if not cont and run_start is not None:
            group = loops[run_start:k]
            entry = group[0]['t_entry']
            last_close = group[-1]['t_close']
            loop_cells = set(group[0]['cells'])
            exit_idx = None
            for j in range(last_close + 1, len(path)):
                if path[j] not in loop_cells:
                    exit_idx = j
                    break
            return entry, exit_idx, group[0]['canon']
    return None, None, None


def episode_stats(traj):
    path = traj['path']
    loops = extract_loops(path)
    entry, exit_idx, trap = _trap_episode(loops, path)
    return dict(
        loops=loops,
        n_loops=len(loops),
        trapped=trap is not None,
        trap_loop=trap,
        escaped=exit_idx is not None,
        escape_time=None if exit_idx is None else exit_idx - entry,
        excess=(traj['steps'] - traj['shortest']) / max(traj['shortest'], 1),
        success=traj['success'],
        dead=not traj['success'],
        silent=(not traj['success']) and traj.get('n_writes', 0) == 0,
        maze_seed=traj.get('maze_seed'),
        agent=traj.get('agent'),
        round=traj.get('round'),
    )


def population_stats(trajs):
    eps = [episode_stats(t) for t in trajs]
    n = len(eps)
    succ = [e for e in eps if e['success']]
    fail = [e for e in eps if not e['success']]

    # mill: >= 2 DISTINCT agents, same maze, same round, same canonical loop
    groups = {}
    for e in eps:
        if e['trapped']:
            key = (e.get('maze_seed'), e['round'], e['trap_loop'])
            groups.setdefault(key, set()).add(e['agent'])
    mill_episodes = sum(1 for e in eps if e['trapped'] and
                        len(groups.get((e.get('maze_seed'), e['round'],
                                        e['trap_loop']), ())) >= 2)
    rounds = {(e.get('maze_seed'), e['round']) for e in eps}

    # escape_rate split into early / late halves of the rounds run
    rs = sorted({e['round'] for e in eps if e['round'] is not None})
    med = rs[len(rs) // 2] if rs else None
    esc = {}
    for name, pred in [('early', lambda r: r <= med), ('late', lambda r: r > med)]:
        sub = [e for e in eps if e['trapped'] and e['round'] is not None
               and pred(e['round'])]
        esc[f'escape_rate_{name}'] = (sum(e['escaped'] for e in sub) / len(sub)
                                      if sub else None)

    et = [e['escape_time'] for e in eps if e['escape_time'] is not None]
    return dict(
        n_episodes=n,
        cycle_rate=sum(e['n_loops'] > 0 for e in eps) / n,
        individual_cycle_rate=sum(e['trapped'] for e in eps) / n,
        mill_rate=mill_episodes / n,
        mill_groups=sum(1 for g in groups.values() if len(g) >= 2),
        n_rounds=len(rounds),
        excess_over_shortest_success=(sum(e['excess'] for e in succ) / len(succ)
                                      if succ else None),
        excess_over_shortest_failure=(sum(e['excess'] for e in fail) / len(fail)
                                      if fail else None),
        escape_time=(sum(et) / len(et) if et else None),
        death_rate=sum(e['dead'] for e in eps) / n,
        silence=sum(e['silent'] for e in eps) / max(len(fail), 1) if fail else 0.0,
        success_rate=len(succ) / n,
        **esc,
    )
