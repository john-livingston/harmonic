"""Information-gain ranking of predicted transits.

Current knowledge is the posterior covariance C estimated from the MCMC
chain; a hypothetical future observation with timing precision sigma
contributes rank-one Fisher information j j^T / sigma^2, where j is the
analytic model gradient at the candidate transit. Gains are Gaussian
posterior-entropy reductions, reported in bits.

Real chains mix units-disparate parameters (a period variance ~1e4 next to
an amplitude variance ~1e-6) and, when short or unconverged, carry near-
degenerate directions; the raw covariance C can have a condition number of
1e15 or worse. At that conditioning pinv(C) is numerically meaningless, and
the nested pinv of its nuisance-block submatrix is worse still. All the
gains below are invariant under per-parameter diagonal rescaling, so the
actual computation is done in correlation space: C is whitened to
R = C / outer(d, d) with d = sqrt(diag(C)), which typically drops the
condition number by many orders of magnitude while leaving every gain
unchanged in exact arithmetic. Gradients are whitened to match (Jw = J * d)
so quadratic forms in R equal the original ones in C.
"""
import logging
import numpy as np
import pandas as pd

from .model import jacobian

logger = logging.getLogger(__name__)

_RANK_COLS = ['sigma', 'gain_total', 'gain_ttv', 'greedy_rank', 'greedy_gain']


def _sym_pinv(M):
    """Symmetrized pseudo-inverse (pinv output is numerically asymmetric)."""
    Mi = np.linalg.pinv(M)
    return (Mi + Mi.T) / 2


def _ephem_mask(names):
    """Boolean mask of the linear-ephemeris (nuisance) parameters: t0_* and
    single-letter per_* (planet periods). Pair super-periods per_<xy> have a
    two-letter suffix and are targets, as are all amplitudes and ratios."""
    return np.array([n.startswith('t0_')
                     or (n.startswith('per_') and len(n[4:]) == 1)
                     for n in names])


def _gains(C, Fnn_inv, nuis, J, sig):
    """Independent information gains (bits) for each candidate row of J.

    gain_total = 0.5 log2(1 + j^T C j / sigma^2) is the D-optimal entropy
    reduction over all parameters; j^T C j is the variance of the predicted
    transit time, so ranking by gain_total means observing where the model
    prediction is most uncertain relative to the assumed precision.

    gain_ttv subtracts the part absorbed by the ephemeris block N (via the
    Schur identity det C_TT = det F_NN / det F): the entropy reduction of the
    marginal posterior of the TTV parameters alone.
    """
    q = np.einsum('ij,jk,ik->i', J, C, J)
    qn = np.einsum('ij,jk,ik->i', J[:, nuis], Fnn_inv, J[:, nuis])
    gt = 0.5 * np.log2(1.0 + q / sig**2)
    gn = 0.5 * np.log2(1.0 + qn / sig**2)
    return gt, np.clip(gt - gn, 0.0, None)


def rank_transits(flatchain, names, transit_df, planet_letters,
                  non_transiting_outer, phase_offsets, sigmas, t_ref,
                  rank_by='total'):
    """Rank predicted transits by the information gain of observing them.

    Parameters
    ----------
    flatchain : DataFrame
        MCMC samples; must contain all columns in `names`.
    names : list of str
        Parameter order (spec.names); defines the covariance layout.
    transit_df : DataFrame
        scan_transits output (needs `planet` and `epoch` columns).
    planet_letters : str
        Planet letter designations.
    non_transiting_outer : bool
        Whether the last letter is a non-transiting outer planet.
    phase_offsets : bool
        Whether the fit used independent per-planet phases.
    sigmas : dict
        Assumed future timing precision (days) per planet letter.
    t_ref : float
        Phase-pivot time (spec.t_ref).
    rank_by : str
        'total' or 'ttv': the criterion driving the greedy observing order.

    Returns
    -------
    DataFrame
        Copy of transit_df with added columns: sigma, gain_total, gain_ttv
        (independent gains, bits), greedy_rank (1-based observing order) and
        greedy_gain (chosen-criterion gain at pick time, bits).
    """
    if rank_by not in ('total', 'ttv'):
        raise ValueError(f"rank_by must be 'total' or 'ttv', got {rank_by!r}")
    out = transit_df.copy()
    if len(out) == 0:
        for c in _RANK_COLS:
            out[c] = pd.Series(dtype=int if c == 'greedy_rank' else float)
        return out

    X = flatchain[list(names)].to_numpy(float)
    theta = {n: float(np.median(X[:, k])) for k, n in enumerate(names)}
    C = np.cov(X, rowvar=False)
    C = (C + C.T) / 2
    nuis = _ephem_mask(names)

    # Whiten to correlation space (see module docstring): all gains below
    # are invariant under this rescaling, but the raw covariance C is
    # usually far too ill-conditioned for pinv to be numerically trustworthy.
    d = np.sqrt(np.diag(C))
    d = np.where(d > 0, d, 1.0)
    R = C / np.outer(d, d)
    R = (R + R.T) / 2
    F = _sym_pinv(R)

    J = jacobian(theta, list(names), np.asarray(out.planet),
                 np.asarray(out.epoch, dtype=float), planet_letters,
                 non_transiting_outer, phase_offsets, t_ref=t_ref)
    Jw = J * d
    sig = np.array([float(sigmas[p]) for p in out.planet])

    gt, gv = _gains(R, _sym_pinv(F[np.ix_(nuis, nuis)]), nuis, Jw, sig)
    out['sigma'] = sig
    out['gain_total'] = gt
    out['gain_ttv'] = gv

    # Greedy observing order: pick the best remaining candidate against the
    # current covariance, then update as if it had been observed
    # (Sherman-Morrison on R; additive Fisher update on the nuisance block).
    # All bookkeeping stays in whitened space throughout.
    remaining = list(range(len(out)))
    ranks = np.zeros(len(out), dtype=int)
    ggain = np.zeros(len(out))
    Cg = R.copy()
    Fnn = F[np.ix_(nuis, nuis)].copy()
    for r in range(1, len(out) + 1):
        g_t, g_v = _gains(Cg, _sym_pinv(Fnn), nuis, Jw[remaining], sig[remaining])
        g = g_t if rank_by == 'total' else g_v
        b = int(np.argmax(g))
        i = remaining.pop(b)
        ranks[i], ggain[i] = r, float(g[b])
        jw, s2 = Jw[i], sig[i]**2
        Cj = Cg @ jw
        Cg = Cg - np.outer(Cj, Cj) / (s2 + float(jw @ Cj))
        Cg = (Cg + Cg.T) / 2
        jn = jw[nuis]
        Fnn = Fnn + np.outer(jn, jn) / s2
    out['greedy_rank'] = ranks
    out['greedy_gain'] = ggain
    return out
