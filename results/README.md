# Results

Raw L2 simulation outputs live in `data/` as gzip+base64 text chunks
(this repo is maintained through a text-only API, so binaries are
stored encoded; multi-part files are split into `.00`, `.01`, ...).

## Restore the data

```
python results/decode.py
```

produces:

- `l2_phase_results.csv` — 420 runs, phase scan over
  lambda x delta x N x seeds (columns: lam, delta, N, seed, sh0, sh1,
  milling, trips, category)
- `l2_verify_results.csv` — 1800 runs, robustness verification
  (32 seeds, T=8000, parameter perturbations; columns: task, lam,
  delta, seed, params, sh0, sh1, milling, category)

`category` is one of `mill` / `optimal` / `detour` / `mixed`.

## Regenerate the figures

```
python experiments/plot_phase.py
```

writes `L2_phase_diagram.png` (mill probability over the
lambda x delta grid) and `L2_verification.png` (mill fraction per
verification task) into this directory.
