"""L3 maze generator: grid maze with planted shortest path, detour loops, and a ring corridor.

The maze is the L3 analog of the L2 double-bridge + ring world:
  - start (S) and goal (G) on opposite sides,
  - a guaranteed shortest path (planted by construction of a perfect maze),
  - extra wall openings creating delta-detour loops (the "绕路" that can be amplified),
  - one ring corridor near the start (the physical site where a mill can form).

Representation: cells (r, c) on a size x size grid; walls[(r,c)] = set of blocked
directions among 'N','S','E','W'. A *perfect* maze (spanning tree) is carved by
randomized DFS, then `detour_openings` extra walls are removed, and one 2x2 block
closest to the start is fully opened to guarantee a ring.
"""
import random
from collections import deque

DIRS = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1)}
OPP = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}


class Maze:
    def __init__(self, size=9, detour_openings=4, ring=True, seed=0):
        self.size = size
        rng = random.Random(seed)
        cells = [(r, c) for r in range(size) for c in range(size)]
        self.walls = {cell: set(DIRS) for cell in cells}

        # --- perfect maze via randomized DFS -------------------------------
        start = (0, 0)
        seen = {start}
        stack = [start]
        while stack:
            u = stack[-1]
            nbrs = []
            for d, (dr, dc) in DIRS.items():
                v = (u[0] + dr, u[1] + dc)
                if 0 <= v[0] < size and 0 <= v[1] < size and v not in seen:
                    nbrs.append((d, v))
            if not nbrs:
                stack.pop()
                continue
            d, v = rng.choice(nbrs)
            self._open(u, v, d)
            seen.add(v)
            stack.append(v)

        self.start = (0, 0)
        self.goal = (size - 1, size - 1)

        # --- ring corridor: fully open the 2x2 block nearest the start -----
        self.ring_cells = []
        if ring:
            best, best_d = None, None
            for r in range(size - 1):
                for c in range(size - 1):
                    d = r + c
                    if best_d is None or d < best_d:
                        best, best_d = (r, c), d
            r, c = best
            blk = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
            pairs = [((r, c), (r, c + 1), 'E'), ((r + 1, c), (r + 1, c + 1), 'E'),
                     ((r, c), (r + 1, c), 'S'), ((r, c + 1), (r + 1, c + 1), 'S')]
            for u, v, d in pairs:
                self._open(u, v, d)
            self.ring_cells = blk

        # --- detour loops: remove extra walls away from the ring -----------
        candidates = []
        for r in range(size):
            for c in range(size):
                for d in ('S', 'E'):
                    dr, dc = DIRS[d]
                    v = (r + dr, c + dc)
                    if 0 <= v[0] < size and 0 <= v[1] < size and d in self.walls[(r, c)]:
                        if (r, c) not in self.ring_cells and v not in self.ring_cells:
                            candidates.append(((r, c), v, d))
        rng.shuffle(candidates)
        for u, v, d in candidates[:detour_openings]:
            self._open(u, v, d)

    def _open(self, u, v, d):
        self.walls[u].discard(d)
        self.walls[v].discard(OPP[d])

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
