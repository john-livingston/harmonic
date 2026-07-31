# Guide

## Installation

```bash
git clone https://github.com/john-livingston/harmonic.git
cd harmonic
pip install -e ".[dev]"
```

Dependencies: `numpy`, `pandas`, `scipy`, `matplotlib`, `astropy`, `emcee`, `corner`.

## Command line

Fit a model and generate plots + mass constraints:

```bash
harmonic -i examples/kep51.csv -c examples/kep51.ini -o results/
```

Predict transits in a time window (the fit-defining options `-l` / `--phase-offsets` / `-n` / `--t-offset` are recovered automatically from `fit_config.json` in the output directory, so `-o` is all you need):

```bash
harmonic -o results/ --predict "2023-09-17 16:00" "2023-09-17 21:30"
```

Predicted transits are ranked by the information gain of observing them (see [The TTV model](model.md#ranking-future-transits)); `--sigma` sets the assumed timing precision and `--rank-by` picks the criterion for the greedy observing order.

See the [CLI reference](cli.md) for the full flag list.

## Python API

```python
from harmonic import Harmonic

h = Harmonic(
    fp_data='examples/kep51.csv',
    fp_config='examples/kep51.ini',
    outdir='results/',
)

h.fit(walkers=100, steps=2000)      # least-squares init + emcee sampling
h.plot_samples()                    # posterior TTV curves + residuals -> fit.png
constraints = h.print_constraints(mstar=1.0)   # Lithwick masses (returns a DataFrame)
dbic = h.delta_bic()               # ΔBIC vs a linear ephemeris (TTV detection metric)
transits = h.predict(['2023-09-17 16:00', '2023-09-17 21:30'])   # returns ranked DataFrame
```

## Input data format

A CSV of transit times, one row per observed transit:

```csv
planet,epoch,tc,tc_unc
0,0,2454992.1099,0.0013
0,1,2455037.2630,0.0012
1,0,2454918.3523,0.0015
1,1,2454963.5045,0.0011
```

- `planet` — integer identifier, contiguous from 0 (`0, 1, 2, …`)
- `epoch` — transit number (integer, 0-indexed)
- `tc` — transit center time (BJD)
- `tc_unc` — uncertainty in `tc` (days)

Planet integers are mapped to letters in period order (`0 → b`, `1 → c`, …); override the letters with `-l`.

## Configuration file

INI format. `[INIT]` gives initial guesses for each planet pair's TTV amplitude, super-period, and phase reference; `[OUTER]` gives the ephemeris of a non-transiting outer planet (used with `-n`); `[T14]` gives transit durations (days) used only for prediction.

`[T14]` keys must be bare planet letters (`b = 0.24`), one per transiting planet, matching the letters in use (see `-l`). Names or prefixed forms such as `planet_b` are rejected with an error naming the key. Durations for planets outside a given fit are ignored, so one config can serve fits over different subsets of the system.

```ini
[INIT]
a_bc = 0.02          # TTV amplitude, planet b contribution (days)
a_cb = -0.002        # TTV amplitude, planet c contribution (days)
per_bc = 1000        # TTV super-period (days)
t_bc = 2455333       # TTV phase reference (BJD)
# phi_bc = 0.0       # optional initial relative phase (radians), used with --phase-offsets

[OUTER]
per = 260.354        # non-transiting outer planet period (days)
t0 = 2455833         # reference time (BJD)

[T14]
b = 0.24             # transit durations (days)
c = 0.12
```

Each pair needs both `a_ij` (inner planet) and `a_ji` (outer planet), except the final pair when its outer planet is the non-transiting one (`-n`), which needs only `a_ij`. The optional `phi_ij` (radians, default 0.0) seeds the pair's initial relative phase, used only with `--phase-offsets`.

The `[INIT]` keys are converted internally to the sampled `(as, ac, r)` parametrization (see [The TTV model](model.md)); the file format is unchanged from earlier versions.

## Output files

Written to the `-o` directory:

- `samples.csv.gz` — MCMC chain samples
- `fit_config.json` — fit-defining options (recovered automatically by `--predict`)
- `fit_stats.json` — system-wide ΔBIC (harmonic vs. linear ephemeris) + MCMC diagnostics
- `args.txt` — the exact command used
- `data.csv`, `config.ini` — copies of the input CSV and configuration file
- `harmonic_<timestamp>.log` — full log of the run
- `fit.png`, `init.png` — posterior TTV curves and the initial best fit
- `corner.png`, `trace.png` — posterior corner and trace plots
- `predict-<window>.png` — transit-prediction plots
