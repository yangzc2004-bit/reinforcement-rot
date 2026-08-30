"""Phase-diagram sweep: lambda x delta grid on the L2 pheromone world.
First-pass scan (16 seeds/cell, T=5000, N=8). ~7 min on 2 cores.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sim'))
import numpy as np
import time
from multiprocessing import Pool
from bridge_world import run_bridge


def _job(args):
    lam, d, n, seed = args
    r = run_bridge(delta=d, lam=lam, N=n, T=5000, seed=seed)
    return (lam, d, n, seed, r['sh0'], r['sh1'], r['milling'], r['trips'], r['cat'])


if __name__ == '__main__':
    lams = [1.0, 0.99, 0.97, 0.95, 0.9, 0.8, 0.6]
    deltas = [1, 2, 3, 6, 10]
    jobs = [(lam, d, 8, s) for lam in lams for d in deltas for s in range(16)]
    t0 = time.time()
    with Pool(2) as pool:
        results = pool.map(_job, jobs)
    os.makedirs('results', exist_ok=True)
    np.save('results/l2_phase_results.npy', np.array(results, dtype=object))
    print(f'{len(results)} runs in {(time.time() - t0) / 60:.1f} min')
