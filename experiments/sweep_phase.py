"""Phase-diagram sweep: lambda x delta grid on the L2 pheromone world.

NOTE (fixed 2026-08): this script now matches the archived dataset
results/data/l2_phase_results.csv.gz.b64 — 7 lambda x 5 delta x 12 seeds = 420
runs at T=4000, N=8. lam is the PER-TICK scent retention (the sim applies
scent *= lam**10 every 10 ticks). An earlier script revision said 16 seeds /
T=5000; the archived data was produced with the parameters below.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sim'))
import numpy as np
import time
from multiprocessing import Pool
from bridge_world import run_bridge


def _job(args):
    lam, d, n, seed = args
    r = run_bridge(delta=d, lam=lam, N=n, T=4000, seed=seed)
    return (lam, d, n, seed, r['sh0'], r['sh1'], r['milling'], r['trips'], r['cat'])


if __name__ == '__main__':
    lams = [1.0, 0.99, 0.97, 0.95, 0.9, 0.8, 0.6]
    deltas = [1, 2, 3, 6, 10]
    jobs = [(lam, d, 8, s) for lam in lams for d in deltas for s in range(12)]
    t0 = time.time()
    with Pool(2) as pool:
        results = pool.map(_job, jobs)
    os.makedirs('results', exist_ok=True)
    np.save('results/l2_phase_results.npy', np.array(results, dtype=object))
    print(f'{len(results)} runs in {(time.time() - t0) / 60:.1f} min')
