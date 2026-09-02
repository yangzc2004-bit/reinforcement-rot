"""B1 — capability gate.

Question: WITHOUT any shared memory, can a single agent solve these mazes?
The gate for proceeding to B2+ is a no-memory success rate of 80-90%
(mazes too easy -> no room for rot; too hard -> death is attributable to
capability, not reinforcement).

Run:  python -m llm.run_b1            (from the repo root)
Uses: llm/config.yaml (copy config.example.yaml and fill in model + API)
"""
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

try:
    import yaml
except ImportError:  # tiny fallback: accept JSON in config.yaml
    yaml = None

from llm.maze import Maze
from llm.client import from_config, CallLimitExceeded
from llm.agents import SolverAgent
from llm.manifest import new_manifest, write_manifest


def load_config():
    p = Path(os.environ.get('LLM_CONFIG_PATH',
                            Path(__file__).parent / 'config.yaml'))
    if not p.exists():
        sys.exit("llm/config.yaml not found — copy config.example.yaml and fill in "
                 "your model name, api_key and base_url.")
    text = p.read_text()
    if yaml:
        return yaml.safe_load(text)
    return json.loads(text)


def main():
    cfg = load_config()
    mz, b1 = cfg['maze'], cfg['b1']
    if 'B1_MAX_CALLS' in os.environ:
        cfg['model']['max_calls'] = int(os.environ['B1_MAX_CALLS'])
    client = from_config(cfg)
    n_mazes = int(os.environ.get('B1_N_MAZES', b1['n_mazes']))
    max_steps = int(os.environ.get('B1_MAX_STEPS', b1['max_steps']))
    seed_start = int(os.environ.get('B1_SEED_START', mz.get('seed', 0)))
    deceptive_fill_roots = int(os.environ.get(
        'B1_DECEPTIVE_FILL_ROOTS', mz.get('deceptive_fill_roots', 0)))
    ring_entry_bias = os.environ.get(
        'B1_RING_ENTRY_BIAS', mz.get('ring_entry_bias', 'any'))
    navigation_guard = os.environ.get(
        'B1_NAVIGATION_GUARD',
        str(b1.get('navigation_guard', False)),
    ).lower() in ('1', 'true', 'yes', 'on')
    navigation_ledger = os.environ.get(
        'B1_NAVIGATION_LEDGER',
        str(b1.get('navigation_ledger', False)),
    ).lower() in ('1', 'true', 'yes', 'on')
    stagnation_repeats = int(os.environ.get(
        'B1_STAGNATION_REPEATS', b1.get('stagnation_repeats', 0)))
    max_failures_raw = os.environ.get(
        'B1_MAX_FAILURES', b1.get('max_failures'))
    max_failures = (int(max_failures_raw)
                    if max_failures_raw not in (None, '') else None)
    manifest_cfg = deepcopy(cfg)
    manifest_cfg['b1']['n_mazes'] = n_mazes
    manifest_cfg['b1']['max_steps'] = max_steps
    manifest_cfg['b1']['navigation_ledger'] = navigation_ledger
    manifest_cfg['b1']['navigation_guard'] = navigation_guard
    manifest_cfg['b1']['stagnation_repeats'] = stagnation_repeats
    manifest_cfg['b1']['max_failures'] = max_failures
    manifest_cfg['b1']['seed_start'] = seed_start
    manifest_cfg['maze']['deceptive_fill_roots'] = deceptive_fill_roots
    manifest_cfg['maze']['ring_entry_bias'] = ring_entry_bias
    man = new_manifest('b1', manifest_cfg)
    out = Path(os.environ.get(
        'LLM_B1_OUTPUT', Path(__file__).parent / 'b1_results.json'))
    results = []
    if out.exists():
        try:
            loaded = json.loads(out.read_text())
            if isinstance(loaded, list):
                results = loaded
        except (OSError, json.JSONDecodeError):
            results = []
    observed_seeds = [row.get('maze_seed') for row in results]
    expected_prefix = [seed_start + k for k in range(len(results))]
    if observed_seeds != expected_prefix:
        raise ValueError(
            f'{out} contains seeds {observed_seeds[:3]}... but this run expects '
            f'a contiguous prefix starting at {seed_start}; use a fresh output '
            f'path or the matching B1_SEED_START')
    interrupted = False
    early_stopped = False
    error_message = None
    fatal_error = None
    try:
        for k in range(len(results), n_mazes):
            if (max_failures is not None
                    and sum(not row['success'] for row in results) >= max_failures):
                early_stopped = True
                break
            seed = seed_start + k
            maze = Maze(size=mz['size'], delta=mz.get('delta', 4),
                        ring=mz.get('ring', True), seed=seed,
                        min_shortest=mz.get('min_shortest', 30),
                        max_backbone=mz.get('max_backbone', 60),
                        neck_min_dist=mz.get('neck_min_dist', 6),
                        max_corridor=mz.get('max_corridor', 4),
                        deceptive_fill_roots=deceptive_fill_roots,
                        ring_entry_bias=ring_entry_bias)
            agent = SolverAgent(client, maze, memory=None,
                                breadcrumbs=b1.get('breadcrumbs', True),
                                write_rule='none', agent_id=0,
                                navigation_ledger=navigation_ledger,
                                navigation_guard=navigation_guard,
                                stagnation_repeats=stagnation_repeats)
            traj = agent.run(max_steps=max_steps, maze_seed=seed)
            results.append(traj)
            out.write_text(json.dumps(results, indent=1))
            print(f"maze {k}: success={traj['success']} steps={traj['steps']} "
                  f"shortest={traj['shortest']} illegal={traj['illegal']}", flush=True)
            if (max_failures is not None
                    and sum(not row['success'] for row in results) >= max_failures):
                early_stopped = True
                print(f"EARLY STOP: reached max_failures={max_failures}.",
                      flush=True)
                break
    except CallLimitExceeded as e:
        interrupted = True
        error_message = str(e)
        print(f"\nCOST FUSE: {e} — saving partial results.", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        error_message = 'KeyboardInterrupt'
        print("\nINTERRUPTED — saving completed episodes.", flush=True)
    except Exception as e:
        interrupted = True
        fatal_error = e
        error_message = f'{type(e).__name__}: {e}'
        print(f"\nRUN ERROR: {error_message} — saving partial results.", flush=True)
    finally:
        client.close()
    if not results:
        print("no completed episodes; nothing to gate on.")
    else:
        sr = sum(t['success'] for t in results) / len(results)
        incomplete = len(results) < n_mazes
        print(f"\nB1 no-memory success rate: {sr:.1%} over {len(results)} mazes"
              + (" (PARTIAL)" if interrupted or incomplete else ""))
        print(f"API calls: {client.n_calls} (attempts: {client.n_attempts}), "
              f"tokens: {client.n_tokens}")
        if not interrupted and not incomplete:
            if 0.80 <= sr <= 0.90:
                print("GATE PASSED (80-90%): difficulty is calibrated, proceed to B2.")
            else:
                print("GATE NOT MET: adjust maze.size / delta / max_steps "
                      "(or breadcrumbs) and rerun.")
        elif early_stopped:
            print("GATE SKIPPED: pilot hit its predeclared failure boundary.")
        else:
            print("GATE SKIPPED: run was interrupted; rerun with a higher fuse.")
    out.write_text(json.dumps(results, indent=1))
    man['interrupted'] = interrupted
    man['early_stopped'] = early_stopped
    man['error'] = error_message
    man['n_expected'] = n_mazes
    man['n_observed'] = len(results)
    write_manifest(man, out, client=client)
    if fatal_error is not None:
        raise fatal_error
    print("trajectories saved to", out)


if __name__ == '__main__':
    main()
