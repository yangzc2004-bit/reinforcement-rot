"""INDEPENDENT topology validation for the controlled maze generator.

The constructor's _validate() uses the maze's own helper methods
(open_dirs/move/shortest_path_len/_bfs_blocked). If one of those helpers were
buggy, constructor asserts and the "independent" check would share the same
blind spot. This validator therefore rebuilds the adjacency list DIRECTLY from
the raw `walls` representation and re-derives every invariant with its own BFS:

  1. walls symmetry (u opens to v  <=>  v opens to u, on-grid);
  2. full connectivity (all size*size cells reachable from start);
  3. cyclomatic number == (delta > 0) + ring, from the edge count;
  4. BFS shortest S->G == len(backbone) - 1  >=  min_shortest;
  5. detour: shortest A->B with backbone edges BLOCKED == d_ab + delta,
     and delta is EVEN (bipartite parity);
  6. ring: start/goal outside the ring assembly, and exactly ONE edge
     between the assembly and the rest of the maze (single neck);
  7. backbone is a simple path of pairwise-adjacent cells.

Run:  python experiments/validate_mazes.py            (from the repo root)
"""
import sys, os
from collections import deque
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from llm.maze import Maze

N_SEEDS = 1000
DIRS = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1)}
OPP = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}


def adjacency_from_walls(maze):
    """Rebuild adjacency from the raw walls dict — no maze helper methods."""
    size, adj = maze.size, {}
    for (r, c), blocked in maze.walls.items():
        nbrs = []
        for d, (dr, dc) in DIRS.items():
            if d in blocked:
                continue
            v = (r + dr, c + dc)
            assert 0 <= v[0] < size and 0 <= v[1] < size, \
                f'{(r, c)} opens {d} off-grid'
            assert OPP[d] not in maze.walls[v], \
                f'asymmetric wall {(r, c)}-{v}'
            nbrs.append(v)
        adj[(r, c)] = nbrs
    return adj


def bfs_dist(adj, src, dst, blocked_edges=frozenset()):
    dist = {src: 0}
    q = deque([src])
    while q:
        u = q.popleft()
        if u == dst:
            return dist[u]
        for v in adj[u]:
            if v not in dist and frozenset((u, v)) not in blocked_edges:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist.get(dst, -1)


def check(maze, delta, ring, min_shortest):
    adj = adjacency_from_walls(maze)
    size = maze.size

    # 2. connectivity
    seen = set()
    q = deque([maze.start])
    seen.add(maze.start)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)
    assert len(seen) == size * size, f'connected {len(seen)} != {size * size}'

    # 3. cyclomatic number from edge count
    n_edges = sum(len(v) for v in adj.values()) // 2
    cyc = n_edges - size * size + 1
    want = (1 if delta > 0 else 0) + (1 if ring else 0)
    assert cyc == want, f'cyclomatic {cyc} != {want}'

    # 7. backbone is a simple path of adjacent cells
    bb = maze.backbone
    assert bb[0] == maze.start and bb[-1] == maze.goal
    assert len(set(bb)) == len(bb), 'backbone revisits a cell'
    for u, v in zip(bb, bb[1:]):
        assert v in adj[u], f'backbone edge {u}->{v} not open'

    # 4. BFS shortest == designed backbone length
    sp = bfs_dist(adj, maze.start, maze.goal)
    assert sp == len(bb) - 1, f'shortest {sp} != designed {len(bb) - 1}'
    assert sp >= min_shortest

    # 5. detour route == d_ab + delta (backbone segment blocked)
    if delta > 0:
        assert delta % 2 == 0, 'delta must be EVEN (bipartite parity)'
        A, B = maze.detour_endpoints
        i, j = bb.index(A), bb.index(B)
        assert A in bb and B in bb and i < j
        blocked = {frozenset(e) for e in zip(bb, bb[1:])}
        d = bfs_dist(adj, A, B, blocked)
        assert d == (j - i) + delta, f'detour {d} != {j - i}+{delta}'
        # detour cells are off the backbone and internally disjoint
        assert all(c not in bb for c in maze.detour_cells)

    # 6. ring geometry
    if ring:
        assert maze.start not in maze.ring_cells
        assert maze.goal not in maze.ring_cells
        assert maze.neck in bb, 'neck must lie on the backbone'
        assembly = (set(maze.ring_cells) | set(maze.corridor_cells)
                    | set(maze.ring_interior))
        ext = [(c, v) for c in assembly for v in adj[c] if v not in assembly]
        assert len(ext) == 1, f'ring assembly has {len(ext)} external edges'
        assert ext[0][1] == maze.neck, 'the outside endpoint must be the neck'
        # ring cells actually form a cycle: each has exactly 2 assembly-neighbors
        # (perimeter) — interior cells are excluded from this check
        for c in maze.ring_cells:
            deg = sum(1 for v in adj[c] if v in set(maze.ring_cells))
            assert deg == 2, f'ring cell {c} has {deg} ring-neighbors, want 2'


def main():
    fails = []
    for seed in range(N_SEEDS):
        try:
            m = Maze(size=15, delta=4, ring=True, seed=seed)
            check(m, delta=4, ring=True, min_shortest=30)
        except Exception as e:
            fails.append((seed, str(e)[:80]))
    print(f'{N_SEEDS - len(fails)}/{N_SEEDS} seeds pass all INDEPENDENT invariants')
    for seed, err in fails[:10]:
        print(f'  seed {seed}: {err}')
    if fails:
        sys.exit(1)


if __name__ == '__main__':
    main()
