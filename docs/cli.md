# CLI reference

`harmonic` fits a TTV model by default; passing `--predict` switches to the prediction workflow, which reads the fit from `-o` and needs no other model flags (they are recovered from `fit_config.json`).

```bash
harmonic -i data.csv -c config.ini -o results/          # fit
harmonic -o results/ --predict "2023-09-17 16:00" "2023-09-17 21:30"   # predict
```

## Input / output

| Flag | Default | Meaning |
|------|---------|---------|
| `-i`, `--input` | none | input transit-times CSV |
| `-c`, `--config` | none | configuration INI |
| `-o`, `--outdir` | `.` | output directory |
| `-l`, `--letters` | `bcdefghijk` | planet-letter designations (in period order) |
| `--clobber` | off | overwrite an existing fit in `--outdir` |

## Sampling

| Flag | Default | Meaning |
|------|---------|---------|
| `-w`, `--walkers` | 100 | number of MCMC walkers |
| `--steps` | 2000 | number of post-burn steps |
| `-b`, `--burn` | 1000 | number of burn-in steps |
| `--thin` | 10 | chain thinning factor |
| `--nproc` | 10 | number of parallel processes |
| `--seed` | 42 | random seed (fit is fully reproducible) |

## Model

| Flag | Default | Meaning |
|------|---------|---------|
| `-n`, `--non-transiting-outer` | off | add a non-transiting outer planet (needs `[OUTER]` in the config) |
| `--phase-offsets` | off | give each planet in a pair its own TTV phase |
| `-m`, `--mstar` | 1.0 | host-star mass (M☉), scales the mass constraints |

## Prediction

| Flag | Default | Meaning |
|------|---------|---------|
| `-p`, `--predict` | none | predict transits in a window (two ISO timestamps) |
| `--predict-list` | none | also write predicted times to this CSV |
| `--t-offset` | 0 | offset added to get BJD (`0` for BJD data, `2454833` for BKJD) |
| `--sigma` | median `tc_unc` per planet | assumed timing precision (days) for the information-gain ranking |
| `--rank-by` | `total` | greedy-order criterion: `total` info gain or `ttv` (TTV parameters only) |
| `-t`, `--tmax` | none | maximum time for the posterior TTV plot |

## Logging

| Flag | Default | Meaning |
|------|---------|---------|
| `-v`, `--verbose` | off | debug-level output |
| `-q`, `--quiet` | off | warnings only |
