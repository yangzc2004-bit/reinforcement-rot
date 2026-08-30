"""Robustness verification of the phase diagram.
Task A: denser lambda grid x {1,2,3,6} deltas, 32 seeds, T=8000 (band replication).
Task B: one-at-a-time implementation-parameter perturbations at 3 key cells.
~50 min on 2 cores.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sim'))
import numpy as np
import time
from multiprocessing import Pool
from bridge_world import run_bridge


def _job(args):
    tag, lam, d, seed, kw, T = args
    r = run_bridge(delta=d, lam=lam, T=T, seed=seed, **kw)
    return (tag, lam, d, seed, str(sorted(kw.items())), r['sh0'], r['sh1'], r['milling'], r['cat'])


if __name__ == '__main__':
    jobs = []
    lams_A = [1.0, 0.99, 0.97, 0.96, 0.95, 0.93, 0.9, 0.85, 0.8]
    for lam in lams_A:
        for d in [1, 2, 3, 6]:
            for s in range(32):
                jobs.append(('A', lam, d, s, {}, 8000))
    perturbs = [
        ('base', {}),
        ('eps0.3', {'eps': 0.3}), ('eps1.0', {'eps': 1.0}),
        ('merit1.1', {'w_merit': 1.1}), ('merit1.4', {'w_merit': 1.4}),
        ('beta1.5', {'beta': 1.5}), ('beta2.5', {'beta': 2.5}),
        ('R8', {'ring_R': 8}), ('R16', {'ring_R': 16}),
    ]
    for name, kw in perturbs:
        for lam in [1.0, 0.95, 0.8]:
            for s in range(24):
                jobs.append(('B:' + name, lam, 2, s, kw, 5000))
    t0 = time.time()
    with Pool(2) as pool:
        results = pool.map(_job, jobs)
    os.makedirs('results', exist_ok=True)
    np.save('results/l2_verify_results.npy', np.array(results, dtype=object))
    print(f'{len(results)} runs in {(time.time() - t0) / 60:.1f} min')
