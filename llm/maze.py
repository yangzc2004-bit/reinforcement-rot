"""L3 maze generator: CONTROLLED-TOPOLOGY 15x15 maze (replaces random-DFS version).

Why controlled: the random-DFS version had three construct-validity defects —
  (a) the ring was always the 2x2 block AT the start cell (agents spawned inside
      the loop, so circling it was a topological artifact, not a memory effect);
  (b) the ring had 2-4 external openings, not L2's single-neck capture geometry;
  (c) detours were random wall removals, so delta (the perturbation dose) and
      the shortest-path length were uncontrolled.

Construction (exactly two cycles — cyclomatic number 2):
  1. BACKBONE: a simple path S -> G of length >= min_shortest, built from the
     monotone path by random plaquette bumps (+2 steps each). In the tree fill
     this remains the UNIQUE S-G route, hence the BFS shortest path.
  2. DETOUR: between two backbone cells A, B (backbone distance d_ab), a second
     internally-disjoint route of length EXACTLY d_ab + delta — the long lane,
     the L3 analog of L2's delta. NOTE: the grid is bipartite, so two routes
     between the SAME endpoints always differ by an EVEN number of steps —
     delta must be even (delta in {2, 4, 6}, mirroring L2's L1 = L0 + 2*delta).
     Perturbation injection (B3) writes onto maze.detour_cells.
  3. RING: a dangling rectangular loop attached to the backbone at the single
     neck cell N via a short corridor. Entrance == exit == the neck. The start
     cell is NEVER part of the ring assembly.
  4. FILL: each remaining free-space component is attached to the controlled
     graph by EXACTLY ONE root edge, then spanned internally by randomized
     Prim. By default roots prefer a step away from the goal. The explicit
     `deceptive_fill_roots` calibration parameter can instead place a fixed
     number of roots toward the goal. This keeps filler difficulty controlled
     rather than seed-dependent.

Structural invariants (asserted at construction; experiments/validate_mazes.py
re-checks them over 1000 seeds):
  connected; BFS shortest == len(backbone) - 1; detour A-B route == d_ab + delta;
  cyclomatic number == (delta > 0) + ring; start/goal not in ring; exactly one
  edge between the ring assembly and the rest of the maze.

Representation: cells (r, c); walls[(r,c)] = set of blocked directions.
"""
import random
from collections import deque

DIRS = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1)}
OPP = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
PERP = {'N': ('E', 'W'), 'S': ('E', 'W'), 'E': ('N', 'S'), 'W': ('N', 'S')}


class MazeConstructionError(RuntimeError):
    pass


class Maze:
    def __init__(self, size=15, delta=4, ring=True, seed=0, min_shortest=30,
                 max_backbone=60, neck_min_dist=6, max_corridor=4,
                 deceptive_fill_roots=0, ring_entry_bias='any',
                 max_attempts=300):
        if delta % 2 != 0:
            raise ValueError('delta must be EVEN: on a bipartite grid two routes '
                             'between the same endpoints differ by an even number')
        if deceptive_fill_roots not in (0, 1):
            raise ValueError('deceptive_fill_roots currently supports only 0 or 1')
        if ring_entry_bias not in ('any', 'toward', 'away'):
            raise ValueError("ring_entry_bias must be 'any', 'toward', or 'away'")
        self.size = size
        self.delta = delta
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        rng = random.Random(seed)
        for _ in range(max_attempts):
            try:
                self._build(rng, delta, ring, min_shortest, max_backbone,
                            neck_min_dist, max_corridor, deceptive_fill_roots,
                            ring_entry_bias)
                self._validate(delta, ring, min_shortest, deceptive_fill_roots,
                               ring_entry_bias)
                return
            except MazeConstructionError:
                continue
        raise MazeConstructionError(f'seed {seed}: no valid maze in {max_attempts} attempts')

    # --------------------------------------------------------- path growing
    def _bump_path(self, rng, blocked, src, dst, target_len, max_restarts=40,
                   max_tries=4000):
        """Simple path src -> dst of EXACTLY target_len steps, avoiding `blocked`
        (src/dst exempt). Grows a BFS shortest path by random plaquette bumps
        (+2 steps each); parity of (target_len - manhattan) must be even."""
        md = abs(src[0] - dst[0]) + abs(src[1] - dst[1])
        if target_len < md or (target_len - md) % 2 != 0:
            return None
        for _ in range(max_restarts):
            path = self._bfs_path(src, dst, blocked)
            if path is None:
                return None
            if (target_len - (len(path) - 1)) % 2 != 0:
                return None
            cells = set(path)
            tries = 0
            while len(path) - 1 < target_len and tries < max_tries:
                tries += 1
                i = rng.randrange(len(path) - 1)
                u, v = path[i], path[i + 1]
                d = self._dir_between(u, v)
                pd = rng.choice(PERP[d])
                dr, dc = DIRS[pd]
                a = (u[0] + dr, u[1] + dc)
                b = (v[0] + dr, v[1] + dc)
                if not self._usable(a, blocked, cells) or not self._usable(b, blocked, cells):
                    continue
                path[i + 1:i + 1] = [a, b]
                cells.add(a)
                cells.add(b)
            if len(path) - 1 == target_len:
                return path
        return None

    def _usable(self, cell, blocked, cells):
        r, c = cell
        return (0 <= r < self.size and 0 <= c < self.size
                and cell not in blocked and cell not in cells)

    def _bfs_path(self, src, dst, blocked):
        dist = {src: 0}
        prev = {}
        q = deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                path = [dst]
                while path[-1] != src:
                    path.append(prev[path[-1]])
                return path[::-1]
            for d, (dr, dc) in DIRS.items():
                v = (u[0] + dr, u[1] + dc)
                if (0 <= v[0] < self.size and 0 <= v[1] < self.size
                        and v not in dist and (v not in blocked or v == dst)):
                    dist[v] = dist[u] + 1
                    prev[v] = u
                    q.append(v)
        return None

    # ------------------------------------------------------------------ build
    def _build(self, rng, delta, ring, min_shortest, max_backbone,
               neck_min_dist, max_corridor, deceptive_fill_roots,
               ring_entry_bias):
        size = self.size
        self.walls = {(r, c): set(DIRS) for r in range(size) for c in range(size)}

        # 1. backbone S -> G (even lengths only: bipartite parity)
        lo = min_shortest + (min_shortest % 2)
        hi = max_backbone - (max_backbone % 2)
        L = rng.choice(list(range(lo, hi + 1, 2)))
        bb = self._bump_path(rng, frozenset(), self.start, self.goal, L)
        if bb is None:
            raise MazeConstructionError('backbone failed')
        self.backbone = bb
        self._carve_path(bb)
        occupied = set(bb)

        # 2. detour: A, B on backbone, alternative route longer by exactly delta
        self.detour_cells, self.detour_endpoints = [], None
        if delta > 0:
            n = len(bb)
            for _ in range(80):
                if n >= 15:
                    i = rng.randrange(2, n - 12)
                    j = rng.randrange(i + 8, min(n - 2, i + 26))
                else:
                    # Small calibration mazes have shorter backbones. Keep
                    # one-cell endpoint margins and a meaningful A-B segment
                    # instead of assuming the 15x15 experiment geometry.
                    min_span = 3
                    i_hi = n - min_span - 2
                    if i_hi < 1:
                        raise MazeConstructionError('backbone too short for detour')
                    i = rng.randint(1, i_hi)
                    j = rng.randint(i + min_span, n - 2)
                A, B = bb[i], bb[j]
                det = self._bump_path(rng, frozenset(occupied - {A, B}),
                                      A, B, (j - i) + delta)
                if det is not None:
                    self._carve_path(det)
                    self.detour_cells = det[1:-1]
                    self.detour_endpoints = (A, B)
                    self._detour_d_ab = j - i
                    occupied.update(det)
                    break
            if self.detour_endpoints is None:
                raise MazeConstructionError('detour failed')

        # 3. dangling single-neck ring (rectangular loop + short corridor)
        self.ring_cells, self.neck, self.corridor_cells = [], None, []
        self.ring_interior = []
        self.ring_entry_goal_delta = None
        if ring:
            neck_lo = max(neck_min_dist, 2)
            neck_hi = len(bb) - neck_min_dist
            for _ in range(120):
                ni = rng.randrange(neck_lo, neck_hi)
                N = bb[ni]
                a, b = rng.randint(2, 3), rng.randint(2, 4)
                r0 = rng.randrange(0, size - a)
                c0 = rng.randrange(0, size - b)
                # perimeter cells IN CYCLIC ORDER (top L->R, right down,
                # bottom R->L, left up): consecutive entries must be adjacent
                perim = ([(r0, c) for c in range(c0, c0 + b)]
                         + [(r, c0 + b - 1) for r in range(r0 + 1, r0 + a)]
                         + [(r0 + a - 1, c) for c in range(c0 + b - 2, c0 - 1, -1)]
                         + [(r, c0) for r in range(r0 + a - 2, r0, -1)])
                if any(p in occupied for p in perim):
                    continue
                E = perim[0]  # entry == exit corner
                # corridor: any valid length 1..max_corridor with right parity
                cor = None
                for ncor in rng.sample(range(1, max_corridor + 1), max_corridor):
                    candidate = self._bump_path(
                        rng, frozenset(occupied | (set(perim) - {E})),
                        N, E, ncor, max_restarts=6, max_tries=400)
                    if candidate is None:
                        continue
                    first = candidate[1]
                    entry_delta = (
                        abs(first[0] - self.goal[0])
                        + abs(first[1] - self.goal[1])
                        - abs(N[0] - self.goal[0])
                        - abs(N[1] - self.goal[1])
                    )
                    if ring_entry_bias == 'toward' and entry_delta >= 0:
                        continue
                    if ring_entry_bias == 'away' and entry_delta <= 0:
                        continue
                    cor = candidate
                    if cor is not None:
                        break
                if cor is None:
                    continue
                self._carve_path(cor)
                for u, v in zip(perim, perim[1:] + perim[:1]):
                    self._open(u, v, self._dir_between(u, v))
                self.ring_cells = list(perim)
                self.neck = N
                self.corridor_cells = cor[1:-1]
                self.ring_entry_goal_delta = entry_delta
                # rectangle interior becomes part of the ring assembly: it is
                # reachable only THROUGH the ring, so assembly-external edges
                # stay exactly one (the neck). Interior cells are connected by
                # a mini-Prim pass anchored at the assembly.
                interior = [(r, c) for r in range(r0 + 1, r0 + a - 1)
                            for c in range(c0 + 1, c0 + b - 1)]
                self.ring_interior = interior
                assembly_now = set(self.ring_cells) | set(self.corridor_cells) | {N}
                for cell in interior:
                    for d, (dr, dc) in DIRS.items():
                        v = (cell[0] + dr, cell[1] + dc)
                        if v in assembly_now:
                            self._open(cell, v, d)
                            break
                    assembly_now.add(cell)
                occupied.update(cor)
                occupied.update(perim)
                occupied.update(interior)
                break
            if self.neck is None:
                raise MazeConstructionError('ring failed')

        # 4. Component-rooted fill. The old multi-source Prim exposed many
        # random filler entrances directly on the backbone. Those entrances
        # dominated B1 failures even though filler topology is not a scientific
        # variable. Each free-space component now gets exactly one root edge.
        backbone_cells = set(self.backbone)
        anchors = backbone_cells | set(self.detour_cells)
        if ring:
            anchors -= set(self.corridor_cells)
            anchors.discard(self.neck)
        remaining = ({(r, c) for r in range(size) for c in range(size)}
                     - set(occupied))
        components = []
        while remaining:
            seed = min(remaining)
            comp = {seed}
            q = deque([seed])
            remaining.remove(seed)
            while q:
                u = q.popleft()
                for dr, dc in DIRS.values():
                    v = (u[0] + dr, u[1] + dc)
                    if v in remaining:
                        remaining.remove(v)
                        comp.add(v)
                        q.append(v)
            components.append(comp)

        self.fill_components = [sorted(comp) for comp in components]
        self.fill_roots = []
        specs = []
        for comp in components:
            options = []
            for v in sorted(comp):
                for d_from_v, (dr, dc) in DIRS.items():
                    u = (v[0] + dr, v[1] + dc)
                    if u not in anchors:
                        continue
                    d = OPP[d_from_v]  # direction anchor u -> filler root v
                    goal_delta = (
                        abs(v[0] - self.goal[0]) + abs(v[1] - self.goal[1])
                        - abs(u[0] - self.goal[0]) - abs(u[1] - self.goal[1])
                    )
                    # Prefer a root step away from G; then prefer detour over
                    # backbone so filler is less likely to distract the main
                    # capability path.
                    score = (goal_delta, int(u not in backbone_cells))
                    options.append((score, u, v, d, goal_delta))
            if not options:
                raise MazeConstructionError('fill component has no controlled anchor')
            specs.append((comp, options))

        eligible = [
            i for i, (_, options) in enumerate(specs)
            if any(goal_delta < 0 and anchor in backbone_cells
                   for _, anchor, _, _, goal_delta in options)
        ]
        if len(eligible) < deceptive_fill_roots:
            raise MazeConstructionError('not enough toward-goal fill roots')
        deceptive_components = set(
            rng.sample(eligible, deceptive_fill_roots))

        for i, (comp, options) in enumerate(specs):
            deceptive = i in deceptive_components
            if deceptive:
                candidates = [
                    o for o in options
                    if o[4] < 0 and o[1] in backbone_cells
                ]
                target_delta = min(o[4] for o in candidates)
                choices = [o for o in candidates if o[4] == target_delta]
            else:
                best = max(score for score, *_ in options)
                choices = [o for o in options if o[0] == best]
            _, anchor, root, d, goal_delta = rng.choice(choices)
            self._open(anchor, root, d)
            self.fill_roots.append(dict(anchor=anchor, root=root,
                                        goal_delta=goal_delta,
                                        deceptive=deceptive, size=len(comp)))

            grown = {root}
            frontier = []

            def add_component_frontier(cell):
                for direction, (dr, dc) in DIRS.items():
                    nxt = (cell[0] + dr, cell[1] + dc)
                    if nxt in comp and nxt not in grown:
                        frontier.append((nxt, cell, direction))

            add_component_frontier(root)
            while frontier:
                k = rng.randrange(len(frontier))
                v, u, direction = frontier[k]
                frontier[k] = frontier[-1]
                frontier.pop()
                if v in grown:
                    continue
                self._open(u, v, direction)
                grown.add(v)
                add_component_frontier(v)
            if grown != comp:
                raise MazeConstructionError('fill component incomplete')

    @staticmethod
    def _dir_between(u, v):
        dr, dc = v[0] - u[0], v[1] - u[1]
        for d, (ar, ac) in DIRS.items():
            if (dr, dc) == (ar, ac):
                return d
        raise MazeConstructionError(f'non-adjacent cells {u}->{v}')

    def _carve_path(self, cells):
        for u, v in zip(cells, cells[1:]):
            self._open(u, v, self._dir_between(u, v))

    def _open(self, u, v, d):
        self.walls[u].discard(d)
        self.walls[v].discard(OPP[d])

    # ------------------------------------------------------------- validate
    def _validate(self, delta, ring, min_shortest, deceptive_fill_roots,
                  ring_entry_bias):
        size = self.size
        # connectivity + edge count -> cyclomatic number
        n_edges = sum(len(self.open_dirs(c)) for c in self.walls) // 2
        cyc = n_edges - size * size + 1
        want = (1 if delta > 0 else 0) + (1 if ring else 0)
        assert cyc == want, f'cyclomatic {cyc} != {want}'
        # full reachability
        seen = {self.start}
        q = deque([self.start])
        while q:
            u = q.popleft()
            for d in self.open_dirs(u):
                v = self.move(u, d)
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        assert len(seen) == size * size, 'maze not fully connected'
        # designed shortest == actual BFS shortest
        sp = self.shortest_path_len()
        assert sp == len(self.backbone) - 1, f'shortest {sp} != designed {len(self.backbone) - 1}'
        assert sp >= min_shortest
        # detour route length == d_ab + delta (with the backbone segment blocked)
        if delta > 0:
            A, B = self.detour_endpoints
            blocked = {frozenset(e) for e in zip(self.backbone, self.backbone[1:])}
            d = self._bfs_blocked(A, B, blocked)
            assert d == self._detour_d_ab + delta, f'detour {d} != {self._detour_d_ab}+{delta}'
        # ring geometry: start/goal outside, exactly one external edge
        if ring:
            assert self.start not in self.ring_cells and self.goal not in self.ring_cells
            assembly = (set(self.ring_cells) | set(self.corridor_cells)
                        | set(self.ring_interior))
            ext = sum(1 for c in assembly for d in self.open_dirs(c)
                      if self.move(c, d) not in assembly)
            assert ext == 1, f'ring assembly has {ext} external edges, want 1'
            if ring_entry_bias == 'toward':
                assert self.ring_entry_goal_delta < 0
            elif ring_entry_bias == 'away':
                assert self.ring_entry_goal_delta > 0
        assert sum(root['deceptive'] for root in self.fill_roots) == deceptive_fill_roots

    def _bfs_blocked(self, src, dst, blocked):
        dist = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                return dist[u]
            for d in self.open_dirs(u):
                v = self.move(u, d)
                if v not in dist and frozenset((u, v)) not in blocked:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return -1

    # ----------------------------------------------------------------- API
    def open_dirs(self, cell):
        return [d for d in 'NSEW' if d not in self.walls[cell]]

    def move(self, cell, d):
        if d in self.walls[cell]:
            return None
        dr, dc = DIRS[d]
        return (cell[0] + dr, cell[1] + dc)

    def shortest_path_len(self):
        dist = {self.start: 0}
        q = deque([self.start])
        while q:
            u = q.popleft()
            if u == self.goal:
                return dist[u]
            for d in self.open_dirs(u):
                v = self.move(u, d)
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist.get(self.goal, -1)

    def render(self, marks=None):
        """ASCII rendering, for logs/debugging. marks: dict cell -> char."""
        marks = marks or {}
        out = []
        for r in range(self.size):
            top = ''
            mid = ''
            for c in range(self.size):
                top += '+' + ('   ' if 'N' not in self.walls[(r, c)] and r > 0 else '---')
                ch = marks.get((r, c), ' ')
                if (r, c) == self.start:
                    ch = 'S'
                elif (r, c) == self.goal:
                    ch = 'G'
                mid += (' ' if 'W' not in self.walls[(r, c)] and c > 0 else '|') + f' {ch} '
            out.append(top + '+')
            out.append(mid + '|')
        out.append('+' + '---+' * self.size)
        return '\n'.join(out)
