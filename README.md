# Harmonic

A Python package for multi-harmonic Transit Timing Variation (TTV) model fitting and transit prediction for exoplanet systems.

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: WTFPL](https://img.shields.io/badge/License-WTFPL-brightgreen.svg)](http://www.wtfpl.net/about/)

## Overview

**Harmonic** implements multi-harmonic TTV model fitting for analyzing gravitational interactions in multi-planet exoplanet systems. The package uses Markov Chain Monte Carlo (MCMC) methods to fit observed transit times and predict future transit windows.

Transit Timing Variations occur when planets in a multi-planet system gravitationally perturb each other, causing observable deviations from strictly periodic transit times. By analyzing these variations, we can constrain planetary masses and orbital properties even for non-transiting planets.

## Features

- **MCMC Model Fitting**: Uses `emcee` for robust Bayesian parameter estimation
- **Multi-Planet Support**: Handles complex multi-planet systems with harmonic interactions
- **Transit Prediction**: Forecast future transit times with uncertainty estimates
- **Visualization**: Comprehensive plotting of fits, corner plots, and trace plots
- **CLI Interface**: Command-line tool for batch processing
- **Python API**: Flexible programmatic interface for custom workflows
- **Non-Transiting Planets**: Support for inferring properties of non-transiting outer planets

## Installation

### From Source

```bash
git clone https://github.com/john-livingston/harmonic.git
cd harmonic
pip install -e .
```

### Dependencies

- `numpy`
- `pandas` 
- `matplotlib`
- `astropy`
- `scipy`
- `emcee`
- `corner`

## Quick Start

### Using the Command Line Interface

```bash
# Fit TTV model to Kepler-51 data
harmonic -i examples/kep51.csv -c examples/kep51.ini -o results/

# Predict transits for a specific time window  
harmonic -o results/ --predict "2023-09-17 16:00" "2023-09-17 21:30"
```

### Using the Python API

```python
from harmonic import Harmonic

# Initialize with data and configuration
h = Harmonic(
    fp_data='examples/kep51.csv',
    fp_config='examples/kep51.ini', 
    outdir='results/'
)

# Fit the model
h.fit(walkers=100, steps=2000)

# Generate plots
h.plot_samples()

# Print mass constraints
h.print_constraints(mstar=1.0)

# Predict future transits
h.predict(['2023-09-17 16:00', '2023-09-17 21:30'])
```

## Input Data Format

### Transit Times CSV

The input data should be a CSV file with the following columns:

```csv
planet,epoch,tc,tc_unc
0,0,2454992.1099,0.0013
0,1,2455037.2630,0.0012
1,0,2454918.3523,0.0015
1,1,2454963.5045,0.0011
```

- `planet`: Planet identifier (0, 1, 2, ...)
- `epoch`: Transit number (0-indexed)
- `tc`: Transit center time (BJD)
- `tc_unc`: Uncertainty in transit time (days)

### Configuration File

Configuration uses INI format:

```ini
[INIT]
# Initial parameter guesses
a_bc = 0.02          # TTV amplitude (days)
per_bc = 1000        # TTV period (days)  
t_bc = 2455333       # TTV phase reference (BJD)

[OUTER]
# Non-transiting outer planet (optional)
per = 260.354        # Orbital period (days)
t0 = 2455833        # Reference time (BJD)

[T14]
# Transit durations for prediction
b = 0.24            # Planet b duration (days)
c = 0.12            # Planet c duration (days)
```

## Output Files

The package generates several output files in the specified directory:

- `samples.csv.gz`: MCMC chain samples
- `corner.png`: Corner plot of posterior distributions
- `trace.png`: MCMC trace plots
- `fit.png`: Best-fit model comparison with data
- `init.png`: Initial parameter guess visualization
- `predict-TIMESTAMP.png`: Transit prediction plots

## Scientific Background

This package implements the harmonic TTV analysis method described in the literature for studying multi-planet systems. The approach models TTVs as superpositions of sinusoidal variations at specific periods related to planetary orbital resonances.

Key equations and methods are based on:
- Lithwick et al. (2012) - TTV mass constraints
- Nesvorný & Vokrouhlický (2016) - general Hamiltonian TTV formalism (resonant and near-resonant)
- Agol et al. (2005) - TTV theory

### Model Parameters

For each planet, the model fits:
- `per_i`: Mean orbital period of planet i
- `t0_i`: Reference transit time for planet i

For each planet pair (i,j), the sampled parameters are:
- `as_ij`, `ac_ij`: Sine/cosine amplitudes of the TTV sinusoid (days)
- `r_ji`: Signed outer/inner amplitude ratio (shared-phase mode; sign carries the anti-phase)
- `per_ij`: Period of TTV oscillation

With `--phase-offsets`, each planet in a pair gets independent `as`/`ac` amplitudes instead of `r_ji`. The TTV amplitude and phase are derived from the sampled parameters and reported in the summary; they are not sampled directly. The `[INIT]` config format (`a_ij`, `per_ij`, `t_ij`) is unchanged — initial guesses are converted internally.

### Shared phase vs. phase offsets

The default (shared-phase) mode assumes the two planets' TTV sinusoids are exactly anti-correlated (relative phase of 180°, encoded by the sign of `r_ji`). In the Lithwick, Xie & Wu (2012) formalism, exact anti-phase holds in two limits: when the TTVs are dominated by forced eccentricity (|Z_free| << |f||Δ|) and when they are dominated by free eccentricity (|Z_free| >> |f||Δ|). In between, the complex amplitudes V ∝ (−f − (3/2)Z*_free/Δ) and V′ ∝ (−g + (3/2)Z*_free/Δ) rotate by different angles, so the relative phase deviates from 180° — maximally near the crossover |Z_free| ≈ (2/3)|f||Δ|. Deviations from perfect anti-phase are thus a prediction of the analytic near-resonant theory itself, not only of full N-body dynamics — the Lithwick regime is the non-resonant special case of the more general Hamiltonian formalism of Nesvorný & Vokrouhlický (2016), which also covers near-resonant and resonant configurations.

Because the crossover eccentricity scales with the distance from commensurability |Δ|, pairs far from resonance have the highest crossover — so even nearly circular (dynamically cool) systems can show large phase offsets for their widest-of-resonance pair while their tightly coupled pairs stay near anti-phase. V1298 Tau is a working example: the near-commensurate c/d and b/e pairs sit within a few degrees of anti-phase, while d/b (|Δ| ≈ 0.02, the furthest from commensurability) shows a ~27° offset.

When the shared-phase model cannot represent such a pair, the fit compensates by collapsing the inner planet's amplitude toward zero and inflating `r_ji`, so the `r_ji` posterior piles up against its prior bound (harmonic warns when it detects this). In that case re-fit with `--phase-offsets` — the measured relative phase is itself physically interesting, since the deviation from 180° encodes the free eccentricities.

**TODO:** try a symmetric reparametrization of the shared-phase mode — `(v_in, v_out, φ_shared)` instead of `(as, ac, r)` — which keeps the shared-phase constraint but has no ratio divergence when either amplitude is consistent with zero.

## Examples

### Example 1: Basic Analysis

```python
# Analyze the provided Kepler-51 system
from harmonic import Harmonic

h = Harmonic('examples/kep51.csv', 'examples/kep51.ini', 'kep51_results/')
h.fit()
h.plot_samples()
h.print_constraints()
```

### Example 2: Custom Planet Letters

```python
# Use custom planet designations
h = Harmonic(
    fp_data='my_data.csv',
    fp_config='my_config.ini', 
    letters='cdef',  # Start with planet c
    outdir='results/'
)
```

### Example 3: Including Non-Transiting Planet

```python
# Include a non-transiting outer planet
h = Harmonic(
    fp_data='data.csv',
    fp_config='config.ini',
    non_transiting_outer=True,
    outdir='results/'
)
```

## Command Line Options

```bash
harmonic --help
```

Key options:
- `-i, --input`: Input CSV file path
- `-c, --config`: Configuration file path  
- `-o, --outdir`: Output directory
- `-w, --walkers`: Number of MCMC walkers (default: 100)
- `--steps`: Number of MCMC steps (default: 2000)
- `-b, --burn`: Burn-in steps (default: 1000)
- `--seed`: Random seed (default: 42)
- `--predict`: Predict transits in time window
- `--predict-list`: Output CSV file with predicted transit times
- `--phase-offsets`: Allow different phase offsets for each planet pair
- `--t-offset`: Timing offset to add to get BJD (e.g. 2454833 for BKJD data)
- `-n, --non-transiting-outer`: Include non-transiting outer planet
- `--clobber`: Overwrite existing results

## Development

### Setting up for Development

```bash
git clone https://github.com/john-livingston/harmonic.git
cd harmonic
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the WTFPL v3 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This package builds on extensive work in the exoplanet TTV community and uses several excellent Python packages including `emcee`, `corner`, `scipy`, and `astropy`.
