"""Run manifest: every experiment run leaves a birth certificate.

Motivation: the L2 phase scan once drifted out of sync with its own script
(seeds/T in the repo script no longer matched the stored data). From now on,
every results file is accompanied by <name>.manifest.json recording everything
needed to interpret and reproduce it: code version, FULL config, seeds, timing,
environment, and the SHA-256 of the output file itself.

Usage:
    m = new_manifest('b2', cfg)
    ... run experiment ...
    write_manifest(m, out_path, client=client)
"""
import hashlib
import json
import platform
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path


def _git_sha():
    try:
        r = subprocess.run(['git', 'rev-parse', 'HEAD'],
                           capture_output=True, text=True, timeout=5)
        sha = r.stdout.strip()
        return sha or None
    except Exception:
        return None


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def new_manifest(experiment, config):
    """Start a manifest, omitting credentials from the reproducibility record."""
    safe_config = deepcopy(config)
    model_config = safe_config.get('model')
    if isinstance(model_config, dict) and 'api_key' in model_config:
        model_config['api_key'] = '<redacted>'
    return {
        'experiment': experiment,
        'git_sha': _git_sha(),
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'config': safe_config,
        'finished_utc': None,
        'runtime_s': None,
        'output_sha256': None,
        'api_calls': None,
        'api_attempts': None,
        'api_tokens': None,
        'api_errors': None,
        'interrupted': False,
        '_t0': time.time(),
    }


def write_manifest(m, out_path, client=None):
    """Finalize and write <out_path>.manifest.json (or standalone if out_path is None)."""
    m['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    m['runtime_s'] = round(time.time() - m.pop('_t0'), 2)
    if client is not None:
        m['api_calls'] = getattr(client, 'n_calls', None)
        m['api_attempts'] = getattr(client, 'n_attempts', None)
        m['api_tokens'] = getattr(client, 'n_tokens', None)
        m['api_errors'] = getattr(client, 'n_errors', None)
    out_path = Path(out_path) if out_path else None
    if out_path is not None and out_path.exists():
        m['output_sha256'] = file_sha256(out_path)
        mpath = out_path.with_suffix(out_path.suffix + '.manifest.json')
    else:
        mpath = Path('manifest.json')
    mpath.write_text(json.dumps(m, indent=1, default=str))
    return mpath
