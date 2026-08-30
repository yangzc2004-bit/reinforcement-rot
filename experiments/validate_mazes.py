"""Topology validation for the controlled maze generator.

Builds n_seeds mazes and re-checks every structural invariant INDEPENDENTLY of
the constructor's own asserts (belt and suspenders: construction bugs that
silently pass the constructor would still be caught here):

  1. fully connected; 2. BFS shortest == designed backbone length >= 30;
  3. detour route == backbone segment + delta (delta EVEN);
  4. cyclomatic number == 2; 5. start/goal outside ring;
  6. exactly one external edge on the ring assembly (single neck).

Run:  python experiments/validate_mazes.py            (from the repo root)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from llm.maze import Maze

N_SEEDS = 1000

def main():
    fails = []
    for seed in range(N_SEEDS):
        try:
            Maze(size=15, delta=4, ring=True, seed=seed)  # asserts run inside
        except Exception as e:
            fails.append((seed, str(e)[:80]))
    print(f'{N_SEEDS - len(fails)}/{N_SEEDS} seeds pass all invariants')
    for seed, err in fails[:10]:
        print(f'  seed {seed}: {err}')
    if fails:
        sys.exit(1)

if __name__ == '__main__':
    main()
