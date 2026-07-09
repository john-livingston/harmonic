import logging
import numpy as np
import pandas as pd
import astropy.constants as c

# Lithwick, Xie & Wu 2012 (arXiv:1207.4192, v2).
#
# Verified against the paper on 2026-07-08 (PDF fetched from arxiv.org).
#
# Eq. 6 (normalized distance to resonance):
#     Delta = (P'/P) * (j-1)/j - 1
# Eq. 7 (super-period):
#     P^j = P' / (j |Delta|)     ->  |Delta| = P' / (j * P^j)
# Eq. 8 (inner planet's complex TTV, depends on OUTER mass mu'):
#     V  = P  * mu' / (pi * j**(2/3) * (j-1)**(1/3) * Delta) * (-f - 1.5*Zfree*/Delta)
# Eq. 9 (outer planet's complex TTV, depends on INNER mass mu):
#     V' = P' * mu  / (pi * j * Delta) * (-g + 1.5*Zfree*/Delta)
# The paper states f < 0 and g > 0. Both amplitude formulas above match the
# paper verbatim (signs of the 3/2 Zfree*/Delta term: minus in Eq. 8, plus in
# Eq. 9).
#
# Table 3 (coefficients of the disturbing function, expanded to first order in
# Delta). Values read directly from the paper's Table 3:
#     j:j-1  |        f         |        g
#     2:1    | -1.190 + 2.20 D  | 0.4284 -  3.69 D
#     3:2    | -2.025 + 6.21 D  | 2.484  -  5.99 D
#     4:3    | -2.840 + 12.20 D | 3.283  - 11.9  D
#     5:4    | -3.650 + 20.15 D | 4.084  - 19.86 D
# CORRECTION vs. the task brief's starting constants: the brief had the f
# linear-in-Delta slopes as (3.27, 4.38, 5.49) for j=(3,4,5); the paper's
# Table 3 gives (6.21, 12.20, 20.15). The g coefficients and the j=2 f slope
# all matched the brief already. The g(j=2) = 0.4284 - 3.69*Delta anchor
# reproduces the shipped code and the paper exactly.
# (Table 3 footnote b: the 2:1 g contains the indirect term for an internal
# perturber; a distinct g_ext = 0.4284 - 1.17*Delta applies to Eq. 8. The
# paper notes this distinction is "unlikely to be of practical importance", so
# we use a single g per pair.)
TABLE3 = {
    # j: (f0, f1, g0, g1)  ->  f = f0 + f1*Delta, g = g0 + g1*Delta
    2: (-1.190, 2.20, 0.4284, -3.69),
    3: (-2.025, 6.21, 2.484, -5.99),
    4: (-2.840, 12.20, 3.283, -11.9),
    5: (-3.650, 20.15, 4.084, -19.86),
}
SUPPORTED_J = sorted(TABLE3)
RESONANCE_TOL = 0.06  # relative distance to j/(j-1) beyond which we skip

logger = logging.getLogger(__name__)
Mearth_per_Msun = (c.M_earth / c.M_sun).value


def choose_j(ratio):
    """Nearest first-order resonance j:(j-1) to the period ratio, or None."""
    js = np.array(SUPPORTED_J)
    dist = np.abs(ratio - js / (js - 1)) / (js / (js - 1))
    k = int(np.argmin(dist))
    return int(js[k]) if dist[k] < RESONANCE_TOL else None


def get_delta(p_prime, p_super, j):
    """|Delta| recovered from the super-period P^j = P'/(j|Delta|) (eq. 7).

    Returns the MAGNITUDE only; the super-period fixes |Delta| but not its sign.
    The sign (eq. 6: Delta = (P'/P)*(j-1)/j - 1; negative inside resonance) is
    applied by the caller.
    """
    return p_prime / p_super / j


def _amp(flatchain, p_i, p_j, phase_offsets, inner):
    """|TTV amplitude| samples for the inner or outer planet of pair (p_i, p_j)."""
    pair = f'{p_i}{p_j}'
    a_in = np.hypot(flatchain[f'as_{pair}'].values, flatchain[f'ac_{pair}'].values)
    if inner:
        return a_in
    if phase_offsets:
        return np.hypot(flatchain[f'as_{p_j}{p_i}'].values, flatchain[f'ac_{p_j}{p_i}'].values)
    return np.abs(flatchain[f'r_{p_j}{p_i}'].values) * a_in


_MIN_ACCEPT = 3  # need at least this many accepted samples to report a constraint


def _constrain(v_obs, v_model_fn, mu, z, mstar):
    """ABC acceptance: |model amplitude| within 16-84th pct of |observed|.

    The observed amplitude band is narrow when the chain is tightly
    constrained, so only a small fraction of the (mu, z) prior draws land in
    it; _MIN_ACCEPT is a low floor guarding against reporting a constraint from
    essentially no accepted samples.
    """
    v_hat = np.abs(v_model_fn(mu, z))
    lo, hi = np.percentile(v_obs, [16, 84])
    ix = (v_hat > lo) & (v_hat < hi)
    if ix.sum() < _MIN_ACCEPT:
        return None
    m = mu[ix] * mstar / Mearth_per_Msun  # Mp[Me] = (Mp/Mstar) * Mstar[Msun] / (Me/Msun)
    return m.mean(), m.std(), z[ix].mean(), z[ix].std()


def print_constraints(flatchain, nplanets, planet_letters, non_transiting_outer,
                      mstar=1.0, phase_offsets=False, ephem=None, seed=42,
                      mu_min_me=1.0, mu_max_me=30.0, z_max=0.1):
    """Lithwick+2012 mass/eccentricity constraints for every adjacent pair.

    Returns a DataFrame; also logs a summary. Priors: mass ratio log-uniform
    [mu_min_me, mu_max_me] Mearth (per Msun of host); |Zfree| uniform [0, z_max]
    (z_max=0.1 default: physically plausible free eccentricities, replaces the
    old implicit U(0,1)).
    """
    rng = np.random.default_rng(seed)
    ns = len(flatchain)
    mu = np.exp(rng.uniform(np.log(mu_min_me * Mearth_per_Msun),
                            np.log(mu_max_me * Mearth_per_Msun), ns))
    z = rng.uniform(0, z_max, ns)
    rows = []
    for i in range(nplanets - 1):
        p_i, p_j = planet_letters[i], planet_letters[i + 1]
        outer_transits = not (non_transiting_outer and p_j == planet_letters[-1])
        p_ = flatchain[f'per_{p_i}'].values
        if outer_transits:
            p_prime = flatchain[f'per_{p_j}'].values
        else:
            if ephem is None:
                logger.warning("pair %s%s: no ephem for non-transiting outer, skipping", p_i, p_j)
                continue
            p_prime = np.full(ns, float(ephem.loc[p_j, 'per']))
        p_super = flatchain[f'per_{p_i}{p_j}'].values
        ratio = float(np.median(p_prime / p_))
        j = choose_j(ratio)
        if j is None:
            logger.warning("pair %s%s: period ratio %.4f not near a supported MMR, skipping", p_i, p_j, ratio)
            continue
        # |Delta| magnitude from the super-period (eq. 7); sign from eq. 6
        # (Delta = (P'/P)*(j-1)/j - 1, negative inside resonance). The eqs. 8/9
        # amplitude formulas and the Table 3 linear forms f=f0+f1*Delta,
        # g=g0+g1*Delta all require SIGNED Delta. The per-pair scalar sign from
        # the median period ratio is correct; magnitude stays per-sample from
        # the precise super-period measurement.
        delta = np.sign(ratio * (j - 1) / j - 1) * get_delta(p_prime, p_super, j)
        f0, f1, g0, g1 = TABLE3[j]
        f, g = f0 + f1 * delta, g0 + g1 * delta
        logger.info("Delta_%s%s = %.4f +/- %.4f (j=%d, ratio=%.3f)", p_i, p_j, delta.mean(), delta.std(), j, ratio)
        # V (inner planet's observed TTV) constrains OUTER mass mu' (eq. 8)
        v = _amp(flatchain, p_i, p_j, phase_offsets, inner=True)
        res = _constrain(v, lambda m, zz: p_ * m / (np.pi * j**(2/3) * (j-1)**(1/3) * delta) * (-f - 1.5 * zz / delta), mu, z, mstar)
        if res is not None:
            rows.append(dict(pair=p_i+p_j, planet=p_j, mass_me=res[0], mass_err_me=res[1], z=res[2], z_err=res[3], delta=delta.mean(), delta_err=delta.std(), j=j))
        # V' (outer planet's observed TTV) constrains INNER mass mu (eq. 9)
        if outer_transits:
            vp = _amp(flatchain, p_i, p_j, phase_offsets, inner=False)
            res = _constrain(vp, lambda m, zz: p_prime * m / (np.pi * j * delta) * (-g + 1.5 * zz / delta), mu, z, mstar)
            if res is not None:
                rows.append(dict(pair=p_i+p_j, planet=p_i, mass_me=res[0], mass_err_me=res[1], z=res[2], z_err=res[3], delta=delta.mean(), delta_err=delta.std(), j=j))
    df = pd.DataFrame(rows)
    for _, r in df.iterrows():
        logger.info("M_%s = %.2f +/- %.2f Me   |Zfree| = %.3f +/- %.3f  (pair %s, j=%d)",
                    r.planet, r.mass_me, r.mass_err_me, r.z, r.z_err, r.pair, r.j)
    return df
