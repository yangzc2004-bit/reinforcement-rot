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

Forced binary choice: OPEN lists exactly the two noted directions, so the
response space matches the two-alternative softmax being fit. Responses that
fail to parse or name a closed direction are counted separately as `invalid`
and excluded from the fit (reported, never silently dropped).

Run:  python -m llm.calib_beta          (from the repo root)
Uses: llm/config.yaml (model + calib section)
"""
import json
import math
import os
import random
import re
from pathlib import Path

from llm.run_b1 import load_config
from llm.client import from_config, CallLimitExceeded
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
    open_dirs = ''.join(sorted([d_low, d_high]))  # forced binary choice
    lines = [f"CELL: {cell[0]},{cell[1]}",
             f"GOAL: {goal[0]},{goal[1]}",
             f"OPEN: {open_dirs}",
             "NOTES from other agents at this cell:"]
    for d, w in notes:
        lines.append(f"- go {d} (weight {w:.2f}, 1 agent(s)): {rng.choice(TEXTS)}")
    lines.append("Your move?")
    resp = client.chat(PROBE_SYSTEM, '\n'.join(lines))
    m = re.search(r'MOVE:\s*([NSEW])', (resp or '').upper())
    if not m or m.group(1) not in (d_low, d_high):
        return None  # invalid: parse failure or closed direction; not fitted
    return m.group(1) == d_high


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

    raw, invalid, completed = {}, {}, {}
    interrupted = False
    error_message = None
    fatal_error = None
    try:
        for r in ratios:
            hits = []
            n_inv = 0
            raw[r] = hits
            invalid[r] = 0
            completed[r] = 0
            for _ in range(n_trials):
                h = probe_once(client, rng, r)
                completed[r] += 1
                if h is None:
                    n_inv += 1
                else:
                    hits.append(h)
                invalid[r] = n_inv
            raw[r], invalid[r] = hits, n_inv
            print(f"r={r}: P(follow high-weight) = "
                  f"{sum(hits)}/{len(hits)} (invalid: {n_inv})", flush=True)
    except CallLimitExceeded as e:
        interrupted = True
        error_message = str(e)
        print(f"\nCOST FUSE: {e} — fitting on partial data.", flush=True)
    except Exception as e:
        interrupted = True
        fatal_error = e
        error_message = f'{type(e).__name__}: {e}'
        print(f"\nRUN ERROR: {error_message} — fitting on partial data.", flush=True)
    finally:
        client.close()
    rs = [r for r in ratios if raw.get(r)]
    if not rs:
        out = Path(os.environ.get(
            'LLM_CALIB_OUTPUT', Path(__file__).parent / 'calib_beta_results.json'))
        out.write_text(json.dumps(dict(error='no completed ratios',
                                       interrupted=interrupted,
                                       error_message=error_message,
                                       n_trials_requested=n_trials,
                                       trials_completed=completed,
                                       invalid=invalid), indent=1))
        man['interrupted'] = interrupted
        man['error'] = error_message
        man['trials_requested'] = n_trials * len(ratios)
        man['trials_completed'] = sum(completed.values())
        man['invalid_trials'] = sum(invalid.values())
        write_manifest(man, out, client=client)
        print("no completed ratios; partial results + manifest saved")
        if fatal_error is not None:
            raise fatal_error
        return
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

    out = Path(os.environ.get(
        'LLM_CALIB_OUTPUT', Path(__file__).parent / 'calib_beta_results.json'))
    out.write_text(json.dumps(dict(
        ratios=rs, follow_rates=ps, n_trials=n_trials,
        n_trials_requested=n_trials,
        trials_completed={str(r): completed[r] for r in ratios},
        invalid={str(r): invalid.get(r, 0) for r in ratios},
        interrupted=interrupted,
        error_message=error_message,
        beta_eff=beta, ci95=[lo, hi],
        raw={str(r): raw[r] for r in rs}), indent=1))
    man['interrupted'] = interrupted
    man['error'] = error_message
    man['trials_requested'] = n_trials * len(ratios)
    man['trials_completed'] = sum(completed.values())
    man['invalid_trials'] = sum(invalid.values())
    write_manifest(man, out, client=client)
    if fatal_error is not None:
        raise fatal_error
    print("saved to", out)


if __name__ == '__main__':
    main()
