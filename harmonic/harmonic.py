import os
import sys
import shutil
import logging
import numpy as np
import pandas as pd
import configparser
from astropy.time import Time

from .plot import plot_corner, plot_trace, plot_bestfit, plot_samples
from .model import TTV, model
from .params import build_spec, derived_frame
from .lithwick import print_constraints

logger = logging.getLogger(__name__)


class Harmonic:
    """
    TTV model fitting using harmonic analysis and MCMC sampling.
    """

    def __init__(self, fp_data=None, fp_config=None, letters='bcdefghijk', outdir='.', non_transiting_outer=False, phase_offsets=False):
        """
        Initialize Harmonic TTV model.

        Parameters
        ----------
        fp_data : str, optional
            Path to input timing dataset (CSV format)
        fp_config : str, optional
            Path to configuration file (INI format)
        letters : str, optional
            Planet letter designations (default: 'bcdefghijk')
        outdir : str, optional
            Output directory path (default: '.')
        non_transiting_outer : bool, optional
            Include non-transiting outer planet (default: False)
        phase_offsets : bool, optional
            Allow different phase offsets for each planet in a pair (default: False)
        """
        self.fp_data = fp_data
        self.fp_config = fp_config
        self.outdir = outdir
        self.non_transiting_outer = non_transiting_outer
        self.phase_offsets = phase_offsets
        self._setup(letters)

    def _setup(self, letters):

        fp_data = self.fp_data
        fp_config = self.fp_config
        outdir = self.outdir
        non_transiting_outer = self.non_transiting_outer
        phase_offsets = self.phase_offsets

        if fp_config is None:
            fp_config = os.path.join(outdir, 'config.ini')
        # Load configuration with error handling
        from .exceptions import ConfigurationError, DataError

        try:
            config = configparser.ConfigParser()
            if not config.read(fp_config):
                raise ConfigurationError(f"could not read config file: {fp_config}")
        except ConfigurationError:
            # Re-raise ConfigurationErrors as-is
            raise
        except Exception as e:
            raise ConfigurationError(f"error parsing config file: {fp_config}") from e

        # Load data with error handling
        if fp_data is None:
            fp_data = os.path.join(outdir, 'data.csv')

        try:
            times = pd.read_csv(fp_data, comment='#')
            if times.empty:
                raise DataError(f"data file contains no data: {fp_data}")
        except FileNotFoundError as e:
            raise DataError(f"data file not found: {fp_data}") from e
        except pd.errors.EmptyDataError as e:
            raise DataError(f"data file is empty or has no columns: {fp_data}") from e
        except pd.errors.ParserError as e:
            raise DataError(f"CSV parsing error in data file: {fp_data}") from e
        except Exception as e:
            raise DataError(f"unexpected error reading data file: {fp_data}") from e

        # Validate required columns (outside try-except to ensure proper error propagation)
        required_columns = ['planet', 'epoch', 'tc', 'tc_unc']
        missing_columns = [col for col in required_columns if col not in times.columns]
        if missing_columns:
            raise DataError(f"data file missing required columns: {', '.join(missing_columns)}")

        ids = sorted(times.planet.unique())
        if ids != list(range(len(ids))):
            raise DataError(f"planet column must be contiguous integers 0..n-1, got {ids}")
        if (times.tc_unc <= 0).any():
            raise DataError("tc_unc must be positive")

        nplanets = len(times.planet.unique())
        if non_transiting_outer:
            nplanets += 1

        planet_num_to_let = {i:letters[i] for i in range(nplanets)}
        times['planet'] = times.planet.replace(planet_num_to_let)
        planet_letters = letters[:nplanets]

        ttv = TTV(times)

        ephem = ttv.ephem
        if non_transiting_outer:
            per_guess = config.getfloat('OUTER', 'per')
            t0_guess = config.getfloat('OUTER', 't0')
            ephem = pd.concat([ephem, pd.DataFrame(dict(per=per_guess, tc=t0_guess), index=[planet_letters[-1]])])
            
        epoch_min = times.groupby('planet')['epoch'].min() 
        epoch_max = times.groupby('planet')['epoch'].max() 
        epochi = []
        for planet in epoch_max.index:
            for j in range(epoch_min[planet], epoch_max[planet]+1): 
                epochi.append(dict(planet=planet, epoch=j))
        epochi = pd.DataFrame(epochi) 
        planeti = epochi.planet
        epochi = epochi.epoch

        # add params
        p_init = {k:float(v) for k,v in config['INIT'].items()}
        spec = build_spec(p_init, ephem, times, nplanets, planet_letters, non_transiting_outer=non_transiting_outer, phase_offsets=phase_offsets)

        fp = os.path.join(outdir, 'samples.csv.gz')
        if os.path.exists(fp):
            self.flatchain = pd.read_csv(fp)
        else:
            self.flatchain = None

        # Deferred chain-column check: computed here but only raised from the
        # methods that consume the chain, so --clobber can still regenerate a
        # stale chain without the constructor bricking. Empty list -> None.
        self._chain_mismatch = (
            [n for n in spec.names if n not in self.flatchain.columns]
            if self.flatchain is not None else None
        ) or None

        self.config = config
        self.times = times
        self.ttv = ttv
        self.nplanets = nplanets
        self.planet_letters = planet_letters
        self.ephem = ephem
        self.epochi = epochi
        self.planeti = planeti
        self.spec = spec

    def _require_chain(self):
        from .exceptions import PredictionError
        if self.flatchain is None:
            raise PredictionError(f"no MCMC samples found in {self.outdir}; run a fit first")
        if self._chain_mismatch:
            cols = set(self.flatchain.columns)
            shown = ', '.join(self._chain_mismatch[:3])
            suffix = ', ...' if len(self._chain_mismatch) > 3 else ''
            if any(c.startswith('a_') or (c.startswith('t_') and not c.startswith('t0_')) for c in cols):
                raise PredictionError(
                    f"chain in samples.csv.gz predates the v0.4 parametrization "
                    f"(missing columns: {shown}{suffix}); re-fit with --clobber")
            raise PredictionError(
                f"chain columns do not match current flags (missing: {shown}{suffix}); "
                f"check --non-transiting-outer/--phase-offsets against args.txt in {self.outdir}")

    def fit(self, walkers=100, burn=1000, steps=2000, thin=10, nproc=10, clobber=False, seed=42):
        """
        Fit TTV model using MCMC sampling with initial optimization.

        Parameters
        ----------
        walkers : int, optional
            Number of MCMC walkers (default: 100)
        burn : int, optional
            Number of burn-in steps (default: 1000)
        steps : int, optional
            Number of MCMC steps (default: 2000)
        thin : int, optional
            Thinning factor for chain (default: 10)
        nproc : int, optional
            Number of parallel processes (default: 10)
        clobber : bool, optional
            Overwrite existing results (default: False)
        seed : int, optional
            Random seed (default: 42)
        """
        # Skip-path (existing chain, no clobber) feeds plot_samples/print_constraints;
        # guard it here so a stale chain fails loudly instead of KeyError downstream.
        if not clobber and self.flatchain is not None and self._chain_mismatch:
            self._require_chain()
        if self.flatchain is None or clobber:

            outdir = self.outdir
            ttv = self.ttv
            times = self.times
            nplanets = self.nplanets
            planet_letters = self.planet_letters
            non_transiting_outer = self.non_transiting_outer
            phase_offsets = self.phase_offsets
            planeti = self.planeti
            epochi = self.epochi

            planet = np.array(times.planet)
            epoch = np.array(times.epoch)
            tc = np.array(times.tc)
            tc_err = np.array(times.tc_unc)

            from .fit import run_fit
            from .params import derived_frame
            fc, chain, diag = run_fit(self.spec, planet, epoch, tc, tc_err,
                                      nplanets, planet_letters, non_transiting_outer,
                                      phase_offsets, walkers, burn, steps, thin, nproc, seed)
            tci = model(self.spec.to_dict(diag['x_opt']), planeti, epochi, nplanets,
                        planet_letters, non_transiting_outer, phase_offsets,
                        t_ref=self.spec.t_ref)
            plot_bestfit(ttv, times, tci, planeti, epochi, nplanets, planet_letters,
                         non_transiting_outer, fp=os.path.join(outdir, 'init.png'))
            fc.to_csv(os.path.join(outdir, 'samples.csv.gz'), index=False)
            plot_trace(chain, self.spec.labels(), fp=os.path.join(outdir, 'trace.png'))
            plot_corner(fc, labels=self.spec.labels(), fp=os.path.join(outdir, 'corner.png'))
            dv = derived_frame(fc, planet_letters, non_transiting_outer, phase_offsets)
            for col in dv.columns:
                logger.info("%s = %.5f +/- %.5f", col, dv[col].median(), dv[col].std())
            self.flatchain = fc
            self._chain_mismatch = None

    def plot_samples(self, tmax=None):
        """
        Plot MCMC samples showing TTV model fit.

        Parameters
        ----------
        tmax : float, optional
            Maximum time for plotting samples (default: None)
        """
        self._require_chain()
        fp = os.path.join(self.outdir, 'fit.png')
        plot_samples(
            self.ttv,
            model,
            self.times,
            self.ephem,
            self.flatchain,
            self.planeti,
            self.nplanets,
            self.planet_letters,
            self.non_transiting_outer,
            self.phase_offsets,
            tmax=tmax,
            fp=fp,
            t_ref=self.spec.t_ref
            )

    def print_constraints(self, mstar=1.0, seed=42):
        """
        Print Lithwick constraints from MCMC samples.

        Parameters
        ----------
        mstar : float, optional
            Mass of host star in solar masses (default: 1.0)
        seed : int, optional
            Random seed for ABC sampling (default: 42)
        """
        self._require_chain()
        return print_constraints(
            self.flatchain,
            self.nplanets,
            self.planet_letters,
            self.non_transiting_outer,
            mstar=mstar,
            phase_offsets=self.phase_offsets,
            ephem=self.ephem,
            seed=seed)

    def predict(self, window, t_offset=0, output_list=None):
        """
        Predict transit times within a specified time window.

        Parameters
        ----------
        window : list of str
            Time window as two ISO timestamps (in JD/BJD)
        t_offset : float, optional
            Timing offset to add to ephemeris times to get BJD (default: 0 for BJD data,
            use 2454833 for BKJD data)
        output_list : str, optional
            Path to output CSV file with transit list (default: None, no file output)
        """
        from .predict import plot_prediction, scan_transits

        config = self.config
        outdir = self.outdir

        # Validate configuration and inputs
        from .exceptions import ConfigurationError, PredictionError

        self._require_chain()

        if 'T14' not in config:
            raise ConfigurationError("transit duration section [T14] missing from configuration")

        try:
            t14s = {k:float(v) for k,v in config['T14'].items()}
        except ValueError as e:
            raise ConfigurationError("invalid T14 values in configuration: must be numeric") from e

        # Validate window length
        if len(window) != 2:
            raise PredictionError(f"time window must contain exactly two timestamps, got: {window}")

        try:
            window = [Time(i) for i in window]
        except Exception as e:
            raise PredictionError(f"invalid time window format: {window}") from e

        date1 = window[0].iso.split()[0]
        date2 = window[1].iso.split()[0]
        hour1 = window[0].iso.split()[1].split(':')[0]
        hour2 = window[1].iso.split()[1].split(':')[0]

        if date1 == date2:
            timestamp = f'{date1}-{hour1}-{hour2}'
        else:
            timestamp = f'{date1}-{hour1}-{date2}-{hour2}'

        transit_df = scan_transits(
            self.flatchain,
            self.ephem,
            self.nplanets,
            self.planet_letters,
            self.non_transiting_outer,
            t14s,
            window,
            self.phase_offsets,
            t_offset=t_offset,
            t_ref=self.spec.t_ref
        )

        fp = os.path.join(outdir, f'predict-{timestamp}.png')
        plot_prediction(
            transit_df,
            self.planet_letters,
            self.non_transiting_outer,
            t14s,
            window,
            fp=fp
            )
        logger.info("Created file: %s", fp)

        # Generate transit list if requested
        if output_list is not None:
            transit_df.to_csv(output_list, index=False, float_format='%.5f')
            logger.info("Created transit list: %s", output_list)
            logger.info("Found %d transits", len(transit_df))

def _build_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Fit harmonic TTV model")
    parser.add_argument('-i', '--input', help='Input timing dataset (CSV)', type=str, default=None)
    parser.add_argument('-c', '--config', help='Configuration file (INI)', type=str, default=None)
    parser.add_argument('-o', '--outdir', help='Output directory', type=str, default='.')
    parser.add_argument('--clobber', help='Overwrite previous results', action="store_true")
    parser.add_argument('-l', '--letters', help='Planet letters', type=str, default='bcdefghijk')
    parser.add_argument('-w', '--walkers', help='Number of walkers', type=int, default=100)
    parser.add_argument('--steps', help='Number of steps', type=int, default=2000)
    parser.add_argument('-b', '--burn', help='Number of burn-in steps', type=int, default=1000)
    parser.add_argument('--thin', help='Number of thinning steps', type=int, default=10)
    parser.add_argument('--nproc', help='Number of processes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('-n', '--non-transiting-outer', help='Include a non-transiting outer planet', action="store_true")
    parser.add_argument('--phase-offsets', help='Allow different phase offsets for sinusoids in each planet pair', action="store_true")
    parser.add_argument('-t', '--tmax', help='Maximum time for plotting samples', type=float, default=None)
    parser.add_argument('-m', '--mstar', help='Mass of host star (Msun)', type=float, default=1.0)
    parser.add_argument('-p', '--predict',
        help='Predict transit(s) within window (2 ISO timestamps, e.g. "2023-09-17 16:00" "2023-09-17 21:30")',
        type=str, nargs=2, default=None
        )
    parser.add_argument('--predict-list', help='Output CSV file with predicted transit times', type=str, default=None)
    parser.add_argument('--t-offset', help='Timing offset to add to get BJD (0 for BJD data, 2454833 for BKJD data)', type=float, default=0)
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output (debug level)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Minimal output (warnings only)')
    return parser


# Fit-defining options a predict run must match; recovered from the fit's
# args.txt so `harmonic -o <dir> --predict ...` needs nothing else.
_FIT_OPTIONS = ['letters', 'non_transiting_outer', 'phase_offsets', 't_offset']


def _fit_config_from_outdir(outdir, parser):
    """Recover the fit's CLI options by re-parsing the args.txt the fit wrote
    into outdir. Returns a Namespace, or None if absent/unparseable."""
    import shlex
    fp = os.path.join(outdir, 'args.txt')
    if not os.path.exists(fp):
        return None
    try:
        with open(fp) as f:
            tokens = shlex.split(f.read().strip())
        return parser.parse_args(tokens[1:])  # drop the program name
    except (SystemExit, OSError, ValueError):
        return None


def _resolve_predict_options(args, parser):
    """Fit-defining options for a predict run: taken from the fit's args.txt
    when present (a predict must match its fit), falling back to the CLI
    values otherwise. Explicit conflicting CLI flags are ignored with a
    warning."""
    fit = _fit_config_from_outdir(args.outdir, parser)
    if fit is None:
        return {n: getattr(args, n) for n in _FIT_OPTIONS}
    resolved = {}
    for n in _FIT_OPTIONS:
        fit_val, cli_val = getattr(fit, n), getattr(args, n)
        if cli_val != parser.get_default(n) and cli_val != fit_val:
            logger.warning("--%s=%s ignored; using %s from the fit's args.txt in %s",
                           n.replace('_', '-'), cli_val, fit_val, args.outdir)
        resolved[n] = fit_val
    logger.info("Recovered fit configuration from %s/args.txt (letters=%s, "
                "non_transiting_outer=%s, phase_offsets=%s, t_offset=%.10g)",
                args.outdir, resolved['letters'], resolved['non_transiting_outer'],
                resolved['phase_offsets'], resolved['t_offset'])
    return resolved


def cli():

    import time
    tick = time.time()

    parser = _build_parser()
    args = parser.parse_args()

    # Configure logging based on CLI args
    from . import setup_logging

    # Create output directory if it doesn't exist
    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Create log file path with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(outdir, f'harmonic_{timestamp}.log')
    
    if args.verbose:
        setup_logging(console_level=logging.DEBUG, file_level=logging.DEBUG, log_file=log_file)
    elif args.quiet:
        setup_logging(console_level=logging.WARNING, file_level=logging.INFO, log_file=log_file)
    else:
        # Default logging setup
        setup_logging(console_level=logging.INFO, file_level=logging.INFO, log_file=log_file)
    
    # Inform user about log file location (only in non-quiet mode)
    if not args.quiet:
        logger.info("Log file: %s", log_file)

    # Import custom exceptions
    from .exceptions import HarmonicError, DataError, ConfigurationError

    fp_data = args.input
    fp_config = args.config
    outdir = args.outdir

    clobber = args.clobber
    letters = args.letters
    walkers = args.walkers
    steps = args.steps
    burn = args.burn
    thin = args.thin
    nproc = args.nproc
    non_transiting_outer = args.non_transiting_outer
    phase_offsets = args.phase_offsets
    tmax = args.tmax
    mstar = args.mstar
    t_offset = args.t_offset

    # Main execution with comprehensive error handling
    try:
        # Validate file inputs exist if provided
        if fp_data and not os.path.exists(fp_data):
            raise DataError(f"input data file not found: {fp_data}")
        if fp_config and not os.path.exists(fp_config):
            raise ConfigurationError(f"config file not found: {fp_config}")

        # Save input files with error handling
        if args.predict is None or clobber: # only save the input used to create the fit
            with open(os.path.join(outdir, 'args.txt'), 'w') as w:
                w.write(" ".join(sys.argv)+'\n')
            if fp_data:
                shutil.copyfile(fp_data, os.path.join(outdir, 'data.csv'))
            if fp_config:
                shutil.copyfile(fp_config, os.path.join(outdir, 'config.ini'))

        if args.predict is None:
            # Fitting workflow
            logger.info("Starting TTV fitting workflow")
            harmonic = Harmonic(fp_data, fp_config, letters, outdir, non_transiting_outer, phase_offsets)
            harmonic.fit(walkers, burn, steps, thin, nproc, clobber, seed=args.seed)
            harmonic.plot_samples(tmax)
            harmonic.print_constraints(mstar, seed=args.seed)
            logger.info("TTV fitting completed successfully")

        else:
            # Prediction workflow: recover fit-defining options from the fit's
            # args.txt in outdir so they don't have to be re-supplied.
            logger.info("Starting transit prediction workflow")
            opts = _resolve_predict_options(args, parser)
            harmonic = Harmonic(letters=opts['letters'], outdir=outdir,
                                non_transiting_outer=opts['non_transiting_outer'],
                                phase_offsets=opts['phase_offsets'])
            harmonic.predict(args.predict, opts['t_offset'], output_list=args.predict_list)
            logger.info("Transit prediction completed successfully")

        logger.debug("Script executed in %.1f seconds", time.time() - tick)

    except HarmonicError as e:
        logger.error("%s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("interrupted")
        sys.exit(1)
    except Exception:
        logger.exception("unexpected error")
        sys.exit(1)
