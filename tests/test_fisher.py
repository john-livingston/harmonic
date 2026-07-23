import numpy as np
import pandas as pd
import pytest
from harmonic.fisher import rank_transits, _ephem_mask, _gains, _sym_pinv
from harmonic.model import model, jacobian

# Synthetic Gaussian "posterior": MVN around a TRUE-like point with
# per-parameter spreads chosen so candidate gains are >> 1 bit.
BASE = {'t0_b': 100.0, 'per_b': 45.155, 't0_c': 110.0, 'per_c': 85.32,
        'as_bc': 0.010, 'ac_bc': -0.006, 'r_cb': -2.0, 'per_bc': 650.0}
SPREAD = {'t0_b': 1e-3, 'per_b': 5e-4, 't0_c': 1e-3, 'per_c': 5e-4,
          'as_bc': 1e-3, 'ac_bc': 1e-3, 'r_cb': 0.05, 'per_bc': 5.0}
NAMES = list(BASE)


def _chain(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({k: BASE[k] + rng.normal(0, SPREAD[k], n) for k in NAMES})


def _tdf(rows):
    return pd.DataFrame(rows, columns=['planet', 'epoch'])


def _rand_spd(ndim, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(ndim, ndim))
    return A @ A.T + ndim * np.eye(ndim)


def test_gain_total_matches_slogdet():
    ndim = 6
    C = _rand_spd(ndim, 1)
    rng = np.random.default_rng(2)
    J = rng.normal(size=(3, ndim))
    sig = np.array([0.5, 1.0, 2.0])
    F = np.linalg.inv(C)
    nuis = np.array([True, True, False, False, False, False])
    gt, _ = _gains(C, _sym_pinv(F[np.ix_(nuis, nuis)]), nuis, J, sig)
    for k in range(3):
        Fp = F + np.outer(J[k], J[k]) / sig[k]**2
        expect = 0.5 * (np.linalg.slogdet(Fp)[1] - np.linalg.slogdet(F)[1]) / np.log(2)
        assert abs(gt[k] - expect) < 1e-9


def test_gain_ttv_matches_marginal_slogdet():
    ndim = 6
    C = _rand_spd(ndim, 3)
    rng = np.random.default_rng(4)
    J = rng.normal(size=(2, ndim))
    sig = np.array([1.0, 0.7])
    nuis = np.array([True, True, True, False, False, False])
    tgt = ~nuis
    F = np.linalg.inv(C)
    _, gv = _gains(C, _sym_pinv(F[np.ix_(nuis, nuis)]), nuis, J, sig)
    for k in range(2):
        j = J[k]
        Cj = C @ j
        Cp = C - np.outer(Cj, Cj) / (sig[k]**2 + j @ Cj)
        expect = 0.5 * (np.linalg.slogdet(C[np.ix_(tgt, tgt)])[1]
                        - np.linalg.slogdet(Cp[np.ix_(tgt, tgt)])[1]) / np.log(2)
        assert abs(gv[k] - expect) < 1e-9


def test_gain_invariants_and_sigma_scaling():
    ndim = 6
    C = _rand_spd(ndim, 5)
    rng = np.random.default_rng(6)
    J = rng.normal(size=(4, ndim))
    F = np.linalg.inv(C)
    nuis = np.array([True, False, True, False, False, False])
    Fnn_inv = _sym_pinv(F[np.ix_(nuis, nuis)])
    gt1, gv1 = _gains(C, Fnn_inv, nuis, J, np.full(4, 1.0))
    gt2, gv2 = _gains(C, Fnn_inv, nuis, J, np.full(4, 2.0))
    assert np.all(gt1 >= gv1) and np.all(gv1 >= 0)
    assert np.all(gt2 < gt1) and np.all(gv2 <= gv1)


def test_ephem_mask():
    names = ['t0_b', 'per_b', 'as_bc', 'ac_bc', 'r_cb', 'per_bc', 't0_c', 'per_c']
    m = _ephem_mask(names)
    assert list(m) == [True, True, False, False, False, False, True, True]
    # phase-offsets style (no r_*) and pair per still target
    m2 = _ephem_mask(['t0_b', 'per_b', 'as_bc', 'ac_bc', 'as_cb', 'ac_cb', 'per_bc'])
    assert list(m2) == [True, True, False, False, False, False, False]


def test_rank_transits_end_to_end():
    fc = _chain()
    tdf = _tdf([{'planet': 'b', 'epoch': 30}, {'planet': 'c', 'epoch': 18},
                {'planet': 'b', 'epoch': 31}])
    out = rank_transits(fc, NAMES, tdf, 'bc', False, False,
                        {'b': 0.001, 'c': 0.001}, 0.0)
    for c in ('sigma', 'gain_total', 'gain_ttv', 'greedy_rank', 'greedy_gain'):
        assert c in out.columns
    assert len(out) == 3
    assert np.all(np.isfinite(out.gain_total)) and np.all(np.isfinite(out.gain_ttv))
    assert np.all(out.gain_total >= out.gain_ttv - 1e-12)
    assert np.all(out.gain_ttv >= 0)
    assert sorted(out.greedy_rank) == [1, 2, 3]
    # rank 1 pick's greedy_gain equals its independent gain (chosen criterion)
    r1 = out[out.greedy_rank == 1].iloc[0]
    assert abs(r1.greedy_gain - r1.gain_total) < 1e-12
    # rank_by='ttv': greedy_gain of the first pick matches gain_ttv instead
    out2 = rank_transits(fc, NAMES, tdf, 'bc', False, False,
                         {'b': 0.001, 'c': 0.001}, 0.0, rank_by='ttv')
    r1b = out2[out2.greedy_rank == 1].iloc[0]
    assert abs(r1b.greedy_gain - r1b.gain_ttv) < 1e-12
    with pytest.raises(ValueError):
        rank_transits(fc, NAMES, tdf, 'bc', False, False,
                      {'b': 0.001, 'c': 0.001}, 0.0, rank_by='bogus')


def test_greedy_demotes_duplicate():
    fc = _chain()
    # duplicate the same (planet, epoch) -> identical j; plus one distinct
    # candidate far away in epoch (different TTV phase + ephemeris lever arm)
    tdf = _tdf([{'planet': 'b', 'epoch': 30}, {'planet': 'b', 'epoch': 30},
                {'planet': 'c', 'epoch': 25}])
    out = rank_transits(fc, NAMES, tdf, 'bc', False, False,
                        {'b': 0.001, 'c': 0.001}, 0.0)
    assert out.gain_total.iloc[0] > 1.0  # premise: strong candidate, >> 0.5 bits
    assert out.gain_total.iloc[2] > 1.0  # premise: distinct candidate informative too
    dup_rank = out.greedy_rank.iloc[1]
    assert out.greedy_rank.iloc[0] == 1 and dup_rank == 3
    # repeat measurement is worth at most ~0.5 bits
    assert out.greedy_gain.iloc[1] < 0.6


def test_laplace_variance_matches_sampled_prediction_variance():
    # jT C j is the Laplace prediction variance; on an exactly Gaussian chain
    # it must match the MC variance of the model prediction (validates the
    # "observe where the prediction is most uncertain" equivalence)
    fc = _chain(n=8000, seed=7)
    names = NAMES
    C = np.cov(fc[names].to_numpy(float), rowvar=False)
    med = {k: float(fc[k].median()) for k in names}
    planet, epoch = np.array(['b']), np.array([40.0])
    j = jacobian(med, names, planet, epoch, 'bc', False, False, t_ref=0.0)[0]
    laplace_var = float(j @ C @ j)
    preds = np.array([model(dict(zip(names, row)), planet, epoch, 'bc', False,
                            False, t_ref=0.0)[0]
                      for row in fc[names].to_numpy(float)[:2000]])
    assert abs(laplace_var - preds.var()) / preds.var() < 0.1


def test_rank_follows_prediction_uncertainty():
    # first-order sanity for the headline claim: with equal assumed sigma,
    # the gain_total ordering must match the ordering of the SAMPLED
    # prediction variances (ground truth from the chain, no Laplace), and
    # later epochs of the same planet (larger accumulated ephemeris drift)
    # must outrank earlier ones
    fc = _chain()
    tdf = _tdf([{'planet': 'b', 'epoch': 20}, {'planet': 'b', 'epoch': 60},
                {'planet': 'b', 'epoch': 120}, {'planet': 'c', 'epoch': 40}])
    out = rank_transits(fc, NAMES, tdf, 'bc', False, False,
                        {'b': 0.001, 'c': 0.001}, 0.0)
    X = fc[NAMES].to_numpy(float)[:2000]
    var = []
    for _, r in tdf.iterrows():
        preds = np.array([model(dict(zip(NAMES, row)), np.array([r.planet]),
                                np.array([float(r.epoch)]), 'bc', False, False,
                                t_ref=0.0)[0] for row in X])
        var.append(preds.var())
    assert list(np.argsort(out.gain_total.to_numpy())) == list(np.argsort(var))
    b = out[out.planet == 'b'].sort_values('epoch')
    assert b.gain_total.is_monotonic_increasing


def test_empty_transit_df():
    fc = _chain(n=500)
    tdf = pd.DataFrame(columns=['planet', 'epoch'])
    out = rank_transits(fc, NAMES, tdf, 'bc', False, False, {'b': 0.001, 'c': 0.001}, 0.0)
    assert len(out) == 0
    for c in ('sigma', 'gain_total', 'gain_ttv', 'greedy_rank', 'greedy_gain'):
        assert c in out.columns


def test_gain_ttv_survives_ill_conditioned_chain():
    # regression: a units-disparate, near-degenerate chain (real unconverged
    # chains look like this; observed cond(C) ~ 2e15) made pinv(C) garbage,
    # qn > q, and the clip silently floored gain_ttv to 0 for all candidates.
    # Whitened (correlation-space) computation must keep gain_ttv positive and
    # finite for a candidate that plainly carries TTV information.
    #
    # Reproduction note: the raw suggested recipe (per_bc scale 500) did NOT
    # trigger the bug against the pre-fix code -- cond(C) landed at ~1e29-1e30,
    # a regime where numpy's pinv truncates the null space cleanly and
    # self-consistently. The failure needs cond(C) landing near numpy pinv's
    # rcond truncation boundary (~1/eps ~ 1e15-1e17 relative to the largest
    # singular value), where the truncation decision for C and for the nested
    # F[nuis, nuis] submatrix become inconsistent. Raising per_bc's scale from
    # 500 to 3e4 (60x) lands cond(C) ~ 5e23 with the *whitened* cond(R) still
    # sitting in that pathological ~1e16 band pre-fix, which reproduces a
    # clean (non-NaN) gain_ttv == 0.0 floor for every candidate.
    rng = np.random.default_rng(11)
    n = 300  # few samples in 8 dims, autocorrelation-like redundancy
    fc = _chain(n=n, seed=11)
    fc['per_bc'] = BASE['per_bc'] + rng.normal(0, 3e4, n)        # huge units scale
    lam = rng.normal(0, 1e-6, n)
    fc['as_bc'] = BASE['as_bc'] + lam                            # near-perfectly
    fc['ac_bc'] = BASE['ac_bc'] + lam * (1 + rng.normal(0, 1e-4, n))  # correlated pair
    tdf = _tdf([{'planet': 'b', 'epoch': 60}, {'planet': 'c', 'epoch': 30}])
    out = rank_transits(fc, NAMES, tdf, 'bc', False, False,
                        {'b': 1e-5, 'c': 1e-5}, 0.0)
    assert np.all(np.isfinite(out.gain_total)) and np.all(np.isfinite(out.gain_ttv))
    assert (out.gain_ttv > 0).all()   # was: exactly 0 for all candidates
    assert (out.gain_total >= out.gain_ttv - 1e-9).all()
