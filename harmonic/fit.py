"""Direct scipy + emcee inference pipeline."""
import logging
import numpy as np
import pandas as pd
import emcee
from scipy.optimize import least_squares

from .model import model, jacobian, residual

logger = logging.getLogger(__name__)


def log_prior(theta, spec):
    if np.any(theta <= spec.lo) or np.any(theta >= spec.hi):
        return -np.inf
    lp = 0.0
    for name in spec.log_scale:            # log-uniform: p(x) ~ 1/x
        lp -= np.log(theta[spec.index[name]])
    return lp


def log_prob(theta, spec, planet, epoch, tc, tc_err, planet_letters,
             non_transiting_outer, phase_offsets):
    lp = log_prior(theta, spec)
    if not np.isfinite(lp):
        return -np.inf
    r = residual(spec.to_dict(theta), planet, epoch, tc, tc_err,
                 planet_letters, non_transiting_outer, phase_offsets,
                 t_ref=spec.t_ref)
    return lp - 0.5 * np.sum(r**2)


def _pair_flip_sets(spec, phase_offsets):
    """Per-pair sign-flip variants of the initial guess: the TTV phase guesses
    from the ini are the least reliable inputs, and a wrong pair phase can
    steer the optimizer into the degenerate (inner amp -> 0, r -> R_MAX)
    valley of the shared-phase parametrization. Flipping (as, ac) rotates a
    pair's phase by pi; flipping r (or the outer as/ac) flips the planets'
    relative sign."""
    flip_sets = []
    if phase_offsets:
        keys = [n[3:] for n in spec.names if n.startswith('as_')]
        for k in keys:
            if k[::-1] in keys and keys.index(k) > keys.index(k[::-1]):
                continue  # handle each pair once, from the inner side
            first = [f'as_{k}', f'ac_{k}']
            second = [f'as_{k[::-1]}', f'ac_{k[::-1]}'] if k[::-1] in keys else None
            flip_sets.append((first, second))
    else:
        pairs = [n[3:] for n in spec.names if n.startswith('as_')]
        for k in pairs:
            first = [f'as_{k}', f'ac_{k}']
            rname = f'r_{k[::-1]}'
            second = [rname] if rname in spec.index else None
            flip_sets.append((first, second))
    out = []
    for first, second in flip_sets:
        variants = [first]
        if second is not None:
            variants += [second, first + second]
        out.append(variants)
    return out


def optimize(spec, planet, epoch, tc, tc_err, planet_letters,
             non_transiting_outer, phase_offsets):
    def f(theta):
        return residual(spec.to_dict(theta), planet, epoch, tc, tc_err,
                        planet_letters, non_transiting_outer, phase_offsets,
                        t_ref=spec.t_ref)

    def jac(theta):
        return -jacobian(spec.to_dict(theta), spec.names, planet, epoch,
                         planet_letters, non_transiting_outer,
                         phase_offsets, t_ref=spec.t_ref) / tc_err[:, None]

    eps = 1e-12 * (spec.hi - spec.lo)

    def run(x0):
        return least_squares(f, np.clip(x0, spec.lo + eps, spec.hi - eps),
                             jac=jac, bounds=(spec.lo, spec.hi),
                             method='trf', x_scale='jac')

    # Greedy multi-start over per-pair phase/sign flips of the initial guess
    # (1 + 3 x npairs TRF runs): immune to wrong ini phase guesses.
    best_x0 = spec.x0.copy()
    best = run(best_x0)
    nstarts = 1
    for variants in _pair_flip_sets(spec, phase_offsets):
        for names in variants:
            x0 = best_x0.copy()
            for nm in names:
                x0[spec.index[nm]] *= -1
            cand = run(x0)
            nstarts += 1
            if float(np.sum(cand.fun**2)) < float(np.sum(best.fun**2)) - 1e-9:
                best, best_x0 = cand, x0
    chisq = float(np.sum(best.fun**2))
    dof = max(len(tc) - len(spec), 1)
    logger.info("least_squares: %d starts, status=%d chisq=%.2f reduced=%.3f",
                nstarts, best.status, chisq, chisq / dof)
    return best


def _walker_ball(res, spec, nwalkers, rng):
    scale = spec.hi - spec.lo
    try:
        cov = np.linalg.pinv(res.jac.T @ res.jac)
        cov = (cov + cov.T) / 2  # pinv output is numerically asymmetric; MVN requires symmetric
        p0 = rng.multivariate_normal(res.x, cov, size=nwalkers, check_valid='ignore')
    except np.linalg.LinAlgError:
        p0 = np.repeat(res.x[None, :], nwalkers, axis=0)
    # Independent per-parameter jitter (0.1% of the prior width). When JtJ is
    # rank-deficient (an unconstrained direction, common with --phase-offsets +
    # non-transiting) pinv gives the Laplace covariance a zero-spread direction,
    # so the walkers come out linearly dependent and emcee rejects the initial
    # state. The jitter keeps every column independent; it is negligible for
    # well-constrained parameters and only matters in the degenerate directions.
    p0 = p0 + 1e-3 * scale * rng.standard_normal(p0.shape)
    eps = 1e-10 * scale
    return np.clip(p0, spec.lo + eps, spec.hi - eps)


_EDGE_ZONE = 0.05  # fraction of prior width counted as "at the bound"
_EDGE_WARN = 0.01  # warn when more than this fraction of samples pile there


def _check_ratio_pileup(fc, spec):
    """Warn when a shared-phase amplitude-ratio posterior piles against its
    prior bound: the ratio only diverges when the inner planet's amplitude is
    consistent with zero, i.e. the strict anti-phase (shared-phase) model
    cannot represent the pair. Within the Lithwick+2012 formalism itself,
    anti-phase deviations peak near the crossover |Z_free| ~ (2/3)|f||Delta|
    (anti-phase is restored in both the forced- and free-eccentricity-dominated
    limits), so pairs far from commensurability are the usual culprits -- see
    README "Shared phase vs. phase offsets"."""
    for name in spec.names:
        if not name.startswith('r_'):
            continue
        i = spec.index[name]
        lo, hi = spec.lo[i], spec.hi[i]
        zone = _EDGE_ZONE * (hi - lo)
        v = fc[name].values
        frac = float(((v > hi - zone) | (v < lo + zone)).mean())
        if frac > _EDGE_WARN:
            pair = name[2:][::-1]
            logger.warning(
                "posterior for %s piles against its prior bound (%.0f%% of samples): "
                "the shared-phase model may be misspecified for pair %s -- "
                "consider re-fitting with --phase-offsets", name, 100 * frac, pair)


def run_fit(spec, planet, epoch, tc, tc_err, planet_letters,
            non_transiting_outer, phase_offsets, walkers=100, burn=1000,
            steps=2000, thin=10, nproc=1, seed=42):
    rng = np.random.default_rng(seed)
    res = optimize(spec, planet, epoch, tc, tc_err, planet_letters,
                   non_transiting_outer, phase_offsets)
    p0 = _walker_ball(res, spec, walkers, rng)
    args = (spec, planet, epoch, tc, tc_err, planet_letters,
            non_transiting_outer, phase_offsets)
    if nproc > 1:
        from multiprocessing import Pool
        with Pool(nproc) as pool:
            sampler = emcee.EnsembleSampler(walkers, len(spec), log_prob, args=args, pool=pool)
            sampler.random_state = np.random.RandomState(seed).get_state()  # seed emcee without mutating the global RNG
            sampler.run_mcmc(p0, burn + steps, progress=False)
    else:
        sampler = emcee.EnsembleSampler(walkers, len(spec), log_prob, args=args)
        sampler.random_state = np.random.RandomState(seed).get_state()
        sampler.run_mcmc(p0, burn + steps, progress=False)
    accept = float(np.mean(sampler.acceptance_fraction))
    with np.errstate(invalid='ignore', divide='ignore'):
        # a fully-stuck walker gives acf[0]=0 inside emcee's autocorr (0/0 ->
        # NaN); non-finite tau is already handled below, so silence the noise
        tau = sampler.get_autocorr_time(discard=burn, quiet=True)
    tau_max = float(np.nanmax(tau)) if np.all(np.isfinite(tau)) else float('nan')
    # tau is estimated with discard=burn, so convergence uses post-burn steps only
    converged = bool(np.isfinite(tau_max) and steps > 50 * tau_max)
    logger.info("emcee: acceptance=%.2f tau_max=%.1f converged=%s", accept, tau_max, converged)
    if not converged:
        logger.warning("chain may not be converged (steps < 50*tau); consider more --steps")
    chain = sampler.get_chain(discard=burn)                       # (nsteps, nwalkers, ndim)
    flat = sampler.get_chain(discard=burn, thin=thin, flat=True)  # (n, ndim)
    fc = pd.DataFrame(flat + spec.offset, columns=spec.names)  # chain in absolute time
    _check_ratio_pileup(fc, spec)
    return fc, chain, dict(accept_frac=accept, tau_max=tau_max, converged=converged, x_opt=res.x)
