"""beta_eff calibration — measure how sharply the model follows memory weight.

The repo does NOT claim "retrieval top_k = beta -> infinity". top_k is only a
candidate-set truncation. What actually matters is the model's EFFECTIVE follow
response: given two conflicting notes at a junction with weight ratio r, what
is P(follow the higher-weight one)? We fit the L2-style choice rule

    P(r) = r^beta / (r^beta + 1)

to estimate beta_eff per model — a "model card" number reported alongside B2.

Geometry control: the two note directions are chosen as the two directions that
BOTH approach the goal (e.g. goal to the SE -> notes say S and E), so geometry
is neutral between them and the choice isolates weight-following.

Run:  python -m llm.calib_beta          (from the repo root)
Uses: llm/config.yaml (model + calib section)
"""
import json
import math
import random
import re
from pathlib import Path

from llm.run_b1 import load_config
from llm.client import from_config
from llm.manifest import new_manifest, write_manifest

PROBE_SYSTEM = """You are a maze-solving agent. You move one cell per turn on a grid.
Directions: N (up), S (down), E (right), W (left). Reply with exactly one line:
MOVE: X
where X is one of the open directions."""

TEXTS = ["this way looked promising", "good progress this way",
         "I made headway here", "this branch felt right"]


def probe_once(client, rng, r):
    cell = (rng.randrange(2, 12), rng.randrange(2, 12))
    # goal to the SE: both S and E approach it -> geometry-neutral choice
    goal = (min(14, cell[0] + rng.randrange(2, 5)), min(14, cell[1] + rng.randrange(2, 5)))
    d_low, d_high = ('S', 'E') if rng.random() < 0.5 else ('E', 'S')
    w_low, w_high = 1.0, float(r)
    notes = [(d_low, w_low), (d_high, w_high)]
    rng.shuffle(notes)  # presentation order must not correlate with weight
    lines = [f"CELL: {cell[0]},{cell[1]}",
             f"GOAL: {goal[0]},{goal[1]}",
             "OPEN: NSEW",
             "NOTES from other agents at this cell:"]
    for d, w in notes:
        lines.append(f"- go {d} (weight {w:.2f}, 1 agent(s)): {rng.choice(TEXTS)}")
    lines.append("Your move?")
    resp = client.chat(PROBE_SYSTEM, '\n'.join(lines))
    m = re.search(r'MOVE:\s*([NSEW])', (resp or '').upper())
    return (m.group(1) == d_high) if m else None


def fit_beta(rs, ps):
    """Least-squares fit of P(r) = r^b/(r^b+1) over a beta grid."""
    best, best_err = 0.0, float('inf')
    for i in range(0, 801):
        b = i / 100.0
        err = sum((p - (r ** b) / (r ** b + 1.0)) ** 2 for r, p in zip(rs, ps))
        if err < best_err:
            best, best_err = b, err
    return best


def main():
    cfg = load_config()
    cc = cfg.get('calib', {})
    ratios = cc.get('ratios', [1.0, 1.5, 2.0, 3.0, 5.0, 8.0])
    n_trials = cc.get('n_trials', 30)
    n_boot = cc.get('n_bootstrap', 200)
    client = from_config(cfg)
    man = new_manifest('calib_beta', cfg)
    rng = random.Random(cc.get('seed', 0))

    raw = {}
    for r in ratios:
        hits = [probe_once(client, rng, r) for _ in range(n_trials)]
        raw[r] = [h for h in hits if h is not None]
        print(f"r={r}: P(follow high-weight) = "
              f"{sum(raw[r])}/{len(raw[r])}", flush=True)
    rs = [r for r in ratios if raw[r]]
    ps = [sum(raw[r]) / len(raw[r]) for r in rs]
    beta = fit_beta(rs, ps)
    boots = []
    for _ in range(n_boot):
        ps_b = []
        for r in rs:
            samp = [rng.choice(raw[r]) for _ in raw[r]]
            ps_b.append(sum(samp) / len(samp))
        boots.append(fit_beta(rs, ps_b))
    boots.sort()
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]
    print(f"\nbeta_eff = {beta:.2f}  (95% bootstrap CI [{lo:.2f}, {hi:.2f}])")
    print("reference: beta=0 ignores weight; beta=1 follows weight linearly; "
          "beta>2 is superlinear (L2's mill-forming regime)")

    out = Path(__file__).parent / 'calib_beta_results.json'
    out.write_text(json.dumps(dict(
        ratios=rs, follow_rates=ps, n_trials=n_trials,
        beta_eff=beta, ci95=[lo, hi],
        raw={str(r): raw[r] for r in rs}), indent=1))
    write_manifest(man, out, client=client)
    client.close()
    print("saved to", out)


if __name__ == '__main__':
    main()
