"""Metrics: the five operational criteria of Reinforcement Rot, plus helpers.

Loop detection uses LOOP-ERASED decomposition on the trajectory sequence
(replaces the old first-repeat heuristic, which could not handle nested or
overlapping loops): walk the path maintaining a stack; whenever the next cell
is already on the stack, the stack segment above it is one traversal of a loop
— record it (with its time interval) and erase it. On a grid all true loops
have length >= 4; length-2 "loops" are immediate backtracks and are ignored.

Definitions (operational, frozen 2026-08):
  loop            : a closed cell segment of length >= 3 detected by loop erasure.
  trap            : the FIRST canonical loop detected >= 2 times (consecutively in
                    detection order). Entry = entry index of its first traversal.
  escape          : the first time index after the trap's last traversal on which
                    the path visits a cell OUTSIDE the trap's loop cells.
  escape_time     : escape_index - entry_index; None if never trapped or never escaped.
  cycle_rate      : fraction of step indices lying inside any detected loop segment.
  oscillation_rate: fraction of steps inside a sustained two-cell alternation
                    A-B-A-B-A... of at least four moves (two full periods).
                    This is reported separately from loop/mill metrics because
                    an edge backtrack is not a topological ring.
  excess_over_shortest : successful episodes only — steps minus the planted shortest.
  budget_exhaustion    : failed episodes only — steps burned before the budget died.
  mill_rate       : fraction of (maze, agent) pairs that share a canonical loop
                    IN THE SAME ROUND with >= 1 OTHER agent (collective mill).
                    Grouping key is (maze_seed, round, canonical_loop); a single
                    agent repeating its own loop across rounds does NOT count.
  individual_cycle_rate : fraction of (maze, agent) pairs repeating some canonical
                    loop across >= 2 of their own episodes — private habit, not a mill.
  death_rate      : fraction of episodes that exhaust the step budget without
                    reaching the goal;  silence = the run died with ZERO illegal
                    actions (it looped "correctly", no error ever fired).
  escape_rate_early / escape_rate_late : P(escape | trapped) for episodes in the
                    first / second half of the rounds — tests whether rot deepens
                    with use.
"""
from collections import Counter, defaultdict
from statistics import median

MIN_LOOP = 3  # grid is bipartite, so real loops have length >= 4; 3 is a safe floor


def extract_loops(path):
    """Loop-erased decomposition.

    Returns a list of dicts(canon, cells, t_entry, t_close) in detection order:
      cells   : the loop body (tuple of cells, no repeated endpoint)
      t_entry : path index where this traversal entered the loop (== first cell)
      t_close : path index where this traversal closed the loop (== re-entry)
    """
    path = [tuple(c) for c in path]  # JSON round-trips yield lists; normalize
    stack, pos, loops = [], {}, []
    for t, cell in enumerate(path):
        if cell in pos:
            i = pos[cell]
            body = tuple(stack[i:])
            if len(body) >= MIN_LOOP:
                loops.append(dict(canon=_canon(body), cells=body,
                                  t_entry=t - len(body), t_close=t))
            for c in stack[i:]:
                del pos[c]
            del stack[i:]
        pos[cell] = len(stack)
        stack.append(cell)
    return loops


def _canon(loop_cells):
    """Canonical form of a loop (rotation- and reversal-invariant) for sharing tests."""
    seq = list(loop_cells)
    rots = [tuple(seq[i:] + seq[:i]) for i in range(len(seq))]
    rev = list(reversed(seq))
    rots += [tuple(rev[i:] + rev[:i]) for i in range(len(rev))]
    return min(rots)


def _trap_episode(loops, path):
    """First canonical loop with a CONSECUTIVE repeat (run length >= 2) in
    detection order — an interleaved A,B,A pattern is NOT a trap.
    Returns (entry_idx, exit_idx|None, canon)."""
    run_start = None  # index in loops where the current run began
    for k in range(1, len(loops) + 1):
        cont = k < len(loops) and loops[k]['canon'] == loops[k - 1]['canon']
        if cont and run_start is None:
            run_start = k - 1
        if not cont and run_start is not None:
            # run loops[run_start .. k-1] of one canon, length >= 2: a trap
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


def _two_cell_oscillation(path):
    """Return (step_fraction, longest_run_steps) for sustained A-B alternation.

    A qualifying run contains at least five cells / four moves:
    A-B-A-B-A. A single immediate backtrack A-B-A is exploration, not a
    sustained oscillation.
    """
    path = [tuple(c) for c in path]
    marked_steps = set()
    longest = 0
    i = 0
    while i + 4 < len(path):
        if (path[i] == path[i + 2] == path[i + 4]
                and path[i + 1] == path[i + 3]
                and path[i] != path[i + 1]):
            end = i + 4
            while end + 1 < len(path) and path[end + 1] == path[end - 1]:
                end += 1
            marked_steps.update(range(i, end))
            longest = max(longest, end - i)
            i = end
        else:
            i += 1
    return len(marked_steps) / max(1, len(path) - 1), longest


def episode_stats(traj):
    path = traj['path']
    loops = extract_loops(path)
    in_loop = set()
    for l in loops:
        in_loop.update(range(l['t_entry'], l['t_close']))
    cycle_rate = len(in_loop) / max(1, len(path) - 1)
    oscillation_rate, longest_oscillation = _two_cell_oscillation(path)
    entry, exit_idx, trap_canon = _trap_episode(loops, path)
    return dict(
        cycle_rate=cycle_rate,
        oscillation_rate=oscillation_rate,
        longest_oscillation=longest_oscillation,
        loop_canons=[l['canon'] for l in loops],
        excess_over_shortest=(traj['steps'] - traj['shortest']) if traj['success'] else None,
        budget_exhaustion=None if traj['success'] else traj['steps'],
        trapped=trap_canon is not None,
        escaped=exit_idx is not None,
        escape_time=(exit_idx - entry) if exit_idx is not None else None,
        dead=not traj['success'],
        illegal=traj['illegal'],
    )


def population_stats(trajs):
    eps = [episode_stats(t) for t in trajs]

    # --- collective mill: same canonical loop, SAME maze, SAME round, >= 2 agents
    groups = defaultdict(set)
    for t, e in zip(trajs, eps):
        for c in set(e['loop_canons']):
            groups[(t.get('maze_seed'), t['round'], c)].add(t['agent'])
    shared = {k for k, agents in groups.items() if len(agents) >= 2}
    milling = {(t.get('maze_seed'), t['agent']) for t, e in zip(trajs, eps)
               if any((t.get('maze_seed'), t['round'], c) in shared
                      for c in set(e['loop_canons']))}
    pairs = {(t.get('maze_seed'), t['agent']) for t in trajs}
    n_pairs = max(1, len(pairs))

    # --- individual cycling: one agent repeats a loop across its own episodes
    own = defaultdict(Counter)
    for t, e in zip(trajs, eps):
        for c in set(e['loop_canons']):
            own[(t.get('maze_seed'), t['agent'])][c] += 1
    cyclers = {k for k, cnt in own.items() if any(n >= 2 for n in cnt.values())}

    # --- escape: overall and early-vs-late rounds
    rounds = sorted({t['round'] for t in trajs})
    mid = rounds[len(rounds) // 2] if rounds else 0
    trapped = [(t, e) for t, e in zip(trajs, eps) if e['trapped']]
    esc = [e['escape_time'] for t, e in trapped if e['escaped']]
    early = [e for t, e in trapped if t['round'] < mid]
    late = [e for t, e in trapped if t['round'] >= mid]

    succ = [e['excess_over_shortest'] for e in eps if e['excess_over_shortest'] is not None]
    fail = [e['budget_exhaustion'] for e in eps if e['budget_exhaustion'] is not None]
    deaths = [e for e in eps if e['dead']]
    return dict(
        n_episodes=len(eps),
        cycle_rate=sum(e['cycle_rate'] for e in eps) / len(eps),
        oscillation_rate=sum(e['oscillation_rate'] for e in eps) / len(eps),
        oscillation_episode_rate=(
            sum(e['longest_oscillation'] > 0 for e in eps) / len(eps)
        ),
        longest_oscillation=max(
            (e['longest_oscillation'] for e in eps), default=0),
        excess_over_shortest=sum(succ) / len(succ) if succ else None,
        budget_exhaustion=sum(fail) / len(fail) if fail else None,
        mill_rate=len(milling) / n_pairs,
        individual_cycle_rate=len(cyclers) / n_pairs,
        n_shared_loops=len(shared),
        escape_time=median(esc) if esc else None,
        escape_rate=(len(esc) / len(trapped)) if trapped else None,
        escape_rate_early=(sum(e['escaped'] for e in early) / len(early)) if early else None,
        escape_rate_late=(sum(e['escaped'] for e in late) / len(late)) if late else None,
        death_rate=len(deaths) / len(eps),
        silence=all(e['illegal'] == 0 for e in deaths) if deaths else None,
    )
