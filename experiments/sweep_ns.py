"""L2 N-sweep for T3: delta=2 (sharpest mill cell), N in {2,4}, 7 lambdas, 12 seeds.
Same protocol as experiments/sweep_phase.py (T=4000). Compare against the N=8
column of the phase scan at delta=2 to read off how the mill band edges move
with population size (viability edge vs race edge).
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from sim.bridge_world import run_bridge

LAMS = [1.0, 0.99, 0.97, 0.95, 0.9, 0.8, 0.6]
NS = [2, 4]
DELTA = 2
SEEDS = list(range(12))
T = 4000

rows = []
for lam in LAMS:
    for N in NS:
        for seed in SEEDS:
            r = run_bridge(delta=DELTA, lam=lam, N=N, T=T, seed=seed)
            rows.append([lam, DELTA, N, seed, r['sh0'], r['sh1'], r['milling'], r['trips'], r['cat']])
        print(f"lam={lam} N={N} done", flush=True)

arr = np.array(rows, dtype=object)
np.save('results/l2_nsweep_results.npy', arr)
print("SAVED", arr.shape)
