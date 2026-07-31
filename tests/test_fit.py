import os
import numpy as np
import pandas as pd
import pytest
from harmonic.model import model
from harmonic.params import build_spec
from harmonic.fit import run_fit, optimize
from .conftest import TRUE_BC

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRUE = TRUE_BC

def _make_synth(shift=0.0):
    rng = np.random.default_rng(3)
    planet = np.array(['b']*30 + ['c']*16)
    epoch = np.array(list(range(30)) + list(range(16)), dtype=float)
    sigma = 0.0015
    true = dict(TRUE, t0_b=TRUE['t0_b'] + shift, t0_c=TRUE['t0_c'] + shift)
    tc = model(true, planet, epoch, 'bc', False, t_ref=shift) + rng.normal(0, sigma, len(epoch))
    times = pd.DataFrame(dict(planet=planet, epoch=epoch, tc=tc, tc_unc=sigma))
    ephem = pd.DataFrame({'per': [45.155, 85.32], 'tc': [100.0 + shift, 110.0 + shift]}, index=['b', 'c'])
    p_init = {'a_bc': 0.008, 'a_cb': -0.015, 'per_bc': 700.0, 't_bc': 200.0 + shift}
    spec = build_spec(p_init, ephem, times, 'bc')
    return spec, planet, epoch, tc, np.full(len(tc), sigma)


@pytest.fixture
def synth():
    return _make_synth()

def test_optimize_recovers_truth(synth):
    spec, planet, epoch, tc, err = synth
    res = optimize(spec, planet, epoch, tc, err, 'bc', False, False)
    d = spec.to_dict(res.x)
    assert abs(d['per_bc'] - TRUE['per_bc']) < 30.0
    a_fit = np.hypot(d['as_bc'], d['ac_bc'])
    assert abs(a_fit - np.hypot(TRUE['as_bc'], TRUE['ac_bc'])) < 0.002

def test_mcmc_roundtrip_small(synth):
    spec, planet, epoch, tc, err = synth
    fc, chain, diag = run_fit(spec, planet, epoch, tc, err, 'bc', False, False,
                              walkers=32, burn=200, steps=200, thin=5, nproc=1, seed=1)
    assert list(fc.columns) == spec.names
    assert chain.shape[1] == 32 and chain.shape[2] == len(spec)
    assert 0.05 < diag['accept_frac'] < 0.9
    med = fc.median()
    assert abs(med['per_bc'] - TRUE['per_bc']) < 50.0
    assert abs(np.hypot(med['as_bc'], med['ac_bc']) - 0.01166) < 0.003

def test_seed_reproducible(synth):
    spec, planet, epoch, tc, err = synth
    fc1, _, _ = run_fit(spec, planet, epoch, tc, err, 'bc', False, False, 16, 50, 50, 2, 1, seed=5)
    fc2, _, _ = run_fit(spec, planet, epoch, tc, err, 'bc', False, False, 16, 50, 50, 2, 1, seed=5)
    pd.testing.assert_frame_equal(fc1, fc2)


def _mini_spec():
    from harmonic.params import ParamSpec
    s = ParamSpec()
    s.add('as_db', 0.001, -0.05, 0.05, r'$A^{\sin}_{db}$')
    s.add('ac_db', 0.001, -0.05, 0.05, r'$A^{\cos}_{db}$')
    s.add('r_bd', 1.0, -20.0, 20.0, r'$r_{bd}$')
    return s.freeze()


def test_ratio_pileup_warns(caplog):
    import logging
    from harmonic.fit import _check_ratio_pileup
    rng = np.random.default_rng(0)
    fc = pd.DataFrame({
        'as_db': rng.normal(0, 0.001, 1000),
        'ac_db': rng.normal(0, 0.001, 1000),
        'r_bd': np.concatenate([rng.uniform(19.2, 20.0, 100), rng.uniform(5, 15, 900)]),
    })
    with caplog.at_level(logging.WARNING, logger='harmonic.fit'):
        _check_ratio_pileup(fc, _mini_spec())
    assert any('r_bd' in r.message and 'phase-offsets' in r.message for r in caplog.records)


def test_ratio_pileup_silent_when_healthy(caplog):
    import logging
    from harmonic.fit import _check_ratio_pileup
    rng = np.random.default_rng(0)
    fc = pd.DataFrame({
        'as_db': rng.normal(0.01, 0.001, 1000),
        'ac_db': rng.normal(0.01, 0.001, 1000),
        'r_bd': rng.normal(-1.1, 0.05, 1000),
    })
    with caplog.at_level(logging.WARNING, logger='harmonic.fit'):
        _check_ratio_pileup(fc, _mini_spec())
    assert not caplog.records


def test_optimize_recovers_truth_bjd_frame():
    # regression: absolute-BJD times (t0 ~ 2.45e6) must not break the
    # optimizer (norm-based termination + trust-region conditioning)
    spec, planet, epoch, tc, err = _make_synth(shift=2454833.0)
    res = optimize(spec, planet, epoch, tc, err, 'bc', False, False)
    d = spec.to_dict(res.x)
    assert abs(d['per_bc'] - TRUE['per_bc']) < 30.0
    a_fit = np.hypot(d['as_bc'], d['ac_bc'])
    assert abs(a_fit - np.hypot(TRUE['as_bc'], TRUE['ac_bc'])) < 0.002
    chisq = np.sum(res.fun**2)
    assert chisq / len(tc) < 2.0


def test_linear_chi2_matches_weighted_polyfit():
    from harmonic.fit import _linear_chi2
    planet = np.array(['b']*5 + ['c']*4)
    epoch = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3], dtype=float)
    rng = np.random.default_rng(0)
    tc = np.concatenate([100 + 45.0*np.arange(5), 200 + 80.0*np.arange(4)]) + rng.normal(0, 0.01, 9)
    err = np.array([0.005, 0.02, 0.01, 0.03, 0.008, 0.015, 0.006, 0.025, 0.012])
    chi2, k = _linear_chi2(planet, epoch, tc, err, 'bc')
    expect = 0.0
    for sl in (slice(0, 5), slice(5, 9)):
        coef = np.polyfit(epoch[sl], tc[sl], 1, w=1.0/err[sl])
        expect += np.sum(((tc[sl] - np.polyval(coef, epoch[sl]))/err[sl])**2)
    assert k == 4
    assert abs(chi2 - expect) < 1e-9


def test_linear_chi2_excludes_non_transiting_outer():
    # data for b,c only; with a non-transiting outer the transiting set is 'bc'
    from harmonic.fit import _linear_chi2
    planet = np.array(['b']*4 + ['c']*4)
    epoch = np.array([0, 1, 2, 3]*2, dtype=float)
    tc = np.concatenate([100 + 45.0*np.arange(4), 200 + 80.0*np.arange(4)])
    err = np.full(8, 0.01)
    _, k = _linear_chi2(planet, epoch, tc, err, 'bc')  # 'bc' = transiting letters, outer excluded
    assert k == 4


def test_bic_evidence_labels():
    from harmonic.fit import _bic_evidence
    assert _bic_evidence(-5.0) == 'linear favored (no TTV detection)'
    assert _bic_evidence(1.0) == 'inconclusive'
    assert _bic_evidence(3.0) == 'positive'
    assert _bic_evidence(8.0) == 'strong'
    assert _bic_evidence(50.0) == 'very strong'
    assert _bic_evidence(0.0) == 'inconclusive'
    assert _bic_evidence(2.0) == 'positive'
    assert _bic_evidence(6.0) == 'strong'
    assert _bic_evidence(10.0) == 'very strong'


def test_bic_evidence_non_finite_is_inconclusive():
    # a NaN delta-BIC fails every comparison in _bic_evidence (dbic < 0 and each
    # dbic >= thresh are all False), so it falls through to the trailing return.
    # That return looks unreachable but is not: deleting it would return None here.
    from harmonic.fit import _bic_evidence
    assert _bic_evidence(float('nan')) == 'inconclusive'


def test_delta_bic_formula(synth):
    from harmonic.fit import delta_bic, _linear_chi2
    from harmonic.model import residual
    spec, planet, epoch, tc, err = synth
    res = optimize(spec, planet, epoch, tc, err, 'bc', False, False)
    d = delta_bic(spec, planet, epoch, tc, err, 'bc', False, False, res.x)
    r = residual(spec.to_dict(res.x), planet, epoch, tc, err, 'bc', False, False, t_ref=spec.t_ref)
    chi2_harm = np.sum(r**2)
    chi2_lin, k_lin = _linear_chi2(planet, epoch, tc, err, 'bc')
    expect = (chi2_lin - chi2_harm) - (len(spec) - k_lin) * np.log(len(tc))
    assert d['k_lin'] == 4 and d['k_harm'] == len(spec) and d['n_data'] == len(tc)
    assert abs(d['delta_bic'] - expect) < 1e-9


def test_delta_bic_detects_strong_ttv(synth):
    # _make_synth carries real 0.01 d TTVs at sigma 0.0015 -> decisive detection
    from harmonic.fit import delta_bic
    spec, planet, epoch, tc, err = synth
    res = optimize(spec, planet, epoch, tc, err, 'bc', False, False)
    d = delta_bic(spec, planet, epoch, tc, err, 'bc', False, False, res.x)
    assert d['delta_bic'] > 50
    assert d['evidence'] == 'very strong'


def test_delta_bic_negative_when_no_ttv():
    # pure linear data + noise: the harmonic model cannot beat the linear null by
    # enough to pay its extra-parameter penalty -> ΔBIC < 0
    from harmonic.fit import delta_bic
    rng = np.random.default_rng(7)
    planet = np.array(['b']*30 + ['c']*16)
    epoch = np.array(list(range(30)) + list(range(16)), dtype=float)
    sigma = 0.0015
    tc = np.where(planet == 'b', 100.0 + 45.155*epoch, 110.0 + 85.32*epoch) + rng.normal(0, sigma, len(epoch))
    err = np.full(len(tc), sigma)
    times = pd.DataFrame(dict(planet=planet, epoch=epoch, tc=tc, tc_unc=err))
    ephem = pd.DataFrame({'per': [45.155, 85.32], 'tc': [100.0, 110.0]}, index=['b', 'c'])
    p_init = {'a_bc': 0.001, 'a_cb': -0.001, 'per_bc': 650.0, 't_bc': 200.0}
    spec = build_spec(p_init, ephem, times, 'bc')
    res = optimize(spec, planet, epoch, tc, err, 'bc', False, False)
    d = delta_bic(spec, planet, epoch, tc, err, 'bc', False, False, res.x)
    assert d['delta_bic'] < 0


def test_chain_columns_absolute_in_bjd_frame():
    spec, planet, epoch, tc, err = _make_synth(shift=2454833.0)
    fc, chain, diag = run_fit(spec, planet, epoch, tc, err, 'bc', False, False,
                              walkers=32, burn=100, steps=100, thin=5, nproc=1, seed=1)
    assert abs(fc['t0_b'].median() - (TRUE['t0_b'] + 2454833.0)) < 0.1


def test_optimize_escapes_bad_phase_basin_kep51(tmp_path):
    # regression: single-start TRF on the shipped kep51 example fell into the
    # degenerate (as->0, r->R_MAX) valley of the cd pair (reduced chisq 4.32,
    # r_dc railed at +20); phase/sign multi-start must reach the true optimum
    from harmonic.harmonic import Harmonic
    h = Harmonic(os.path.join(REPO, 'examples/kep51.csv'),
                os.path.join(REPO, 'examples/kep51.ini'), outdir=str(tmp_path))
    t = h.times
    res = optimize(h.spec, np.array(t.planet), np.array(t.epoch), np.array(t.tc),
                   np.array(t.tc_unc), h.planet_letters, False, False)
    dof = len(t) - len(h.spec)
    assert np.sum(res.fun**2) / dof < 4.0
    d = h.spec.to_dict(res.x)
    assert abs(d['r_dc']) < 19.0  # not railed at R_MAX


def test_walker_ball_independent_unconstrained_param():
    # regression: a parameter the data doesn't constrain (a zero Jacobian column,
    # as arises in over-parametrized --phase-offsets + non-transiting fits) makes
    # pinv(JtJ) give that direction zero spread -> identical walker coordinates ->
    # emcee rejects the initial state ("large condition number"). The per-param
    # jitter in _walker_ball must keep the walkers linearly independent.
    from emcee.ensemble import walkers_independent
    from harmonic.fit import _walker_ball
    from harmonic.params import ParamSpec
    spec = ParamSpec()
    spec.add('a', 0.0, -1.0, 1.0, '$a$')
    spec.add('b', 500.0, 1.0, 1e4, '$b$', log=True)   # very different scale
    spec.add('c', 0.0, -1.0, 1.0, '$c$')              # unconstrained
    spec.freeze()

    class Res:
        pass
    res = Res()
    res.x = spec.x0.copy()
    res.jac = np.array([[1.0, 1e-2, 0.0],   # third column (param 'c') is all zeros
                        [2.0, 3e-2, 0.0],
                        [0.5, 2e-2, 0.0],
                        [1.5, 1e-2, 0.0]])
    p0 = _walker_ball(res, spec, 30, np.random.default_rng(0))
    assert np.all((p0 > spec.lo) & (p0 < spec.hi))
    assert walkers_independent(p0)  # the exact check emcee runs at init


# --- phase-offsets mode: the fit pipeline had no coverage here, only the
# model/jacobian and params layers did ---

TRUE_PO = {'t0_b': 100.0, 'per_b': 45.155, 't0_c': 110.0, 'per_c': 85.32,
           'as_bc': 0.010, 'ac_bc': -0.006,   # planet b
           'as_cb': 0.010, 'ac_cb': 0.017,    # planet c, ~90 deg from b
           'per_bc': 650.0}


def _make_synth_po():
    """Synthetic pair whose planets are ~90 deg apart in TTV phase, i.e. NOT
    the strict anti-correlation the shared-phase model assumes. Only
    --phase-offsets can represent it."""
    rng = np.random.default_rng(3)
    planet = np.array(['b']*30 + ['c']*16)
    epoch = np.array(list(range(30)) + list(range(16)), dtype=float)
    sigma = 0.0015
    tc = model(TRUE_PO, planet, epoch, 'bc', False, True, t_ref=0.0) \
        + rng.normal(0, sigma, len(epoch))
    times = pd.DataFrame(dict(planet=planet, epoch=epoch, tc=tc, tc_unc=sigma))
    ephem = pd.DataFrame({'per': [45.155, 85.32], 'tc': [100.0, 110.0]}, index=['b', 'c'])
    p_init = {'a_bc': 0.008, 'a_cb': 0.015, 'per_bc': 700.0, 't_bc': 200.0, 'phi_bc': 0.0}
    spec = build_spec(p_init, ephem, times, 'bc', phase_offsets=True)
    return spec, planet, epoch, tc, np.full(len(tc), sigma)


def _po_spec(letters, non_transiting_outer):
    """Minimal phase-offsets spec for `letters`, for structural checks."""
    transiting = letters[:-1] if non_transiting_outer else letters
    planet = np.array(sum([[c]*8 for c in transiting], []))
    epoch = np.array(list(range(8)) * len(transiting), dtype=float)
    times = pd.DataFrame(dict(planet=planet, epoch=epoch,
                              tc=100.0 + 45.0*epoch, tc_unc=0.001))
    ephem = pd.DataFrame({'per': [45.0*(i+1) for i in range(len(letters))],
                          'tc': [100.0*(i+1) for i in range(len(letters))]},
                         index=list(letters))
    p_init = {}
    for a, b in zip(letters[:-1], letters[1:]):
        p_init.update({f'a_{a}{b}': 0.01, f'per_{a}{b}': 600.0, f't_{a}{b}': 200.0})
        if b in transiting:
            p_init[f'a_{b}{a}'] = 0.01
    return build_spec(p_init, ephem, times, letters,
                      non_transiting_outer=non_transiting_outer, phase_offsets=True)


def test_optimize_phase_offsets_recovers_relative_phase():
    # regression: the whole optimize() path (including the phase-offsets
    # branch of _pair_flip_sets and every sign-flip restart it generates) was
    # never exercised with phase_offsets=True. Assertions are on
    # phase-reference-invariant quantities: the fit measures phase against
    # spec.t_ref while the data is generated at t_ref=0, so absolute phases
    # differ by a common 2*pi*t_ref/per_ttv. The RELATIVE phase of the pair is
    # invariant, and it is the quantity --phase-offsets exists to measure.
    spec, planet, epoch, tc, err = _make_synth_po()
    res = optimize(spec, planet, epoch, tc, err, 'bc', False, True)
    d = spec.to_dict(res.x)
    amp_b, amp_c = np.hypot(d['as_bc'], d['ac_bc']), np.hypot(d['as_cb'], d['ac_cb'])
    assert abs(amp_b - np.hypot(TRUE_PO['as_bc'], TRUE_PO['ac_bc'])) < 0.002
    assert abs(amp_c - np.hypot(TRUE_PO['as_cb'], TRUE_PO['ac_cb'])) < 0.002
    rel_fit = np.arctan2(d['ac_cb'], d['as_cb']) - np.arctan2(d['ac_bc'], d['as_bc'])
    rel_true = (np.arctan2(TRUE_PO['ac_cb'], TRUE_PO['as_cb'])
                - np.arctan2(TRUE_PO['ac_bc'], TRUE_PO['as_bc']))
    wrap = lambda x: (x + np.pi) % (2*np.pi) - np.pi
    assert abs(wrap(rel_fit - rel_true)) < 0.15   # ~9 deg; truth is ~90 deg apart
    assert abs(wrap(rel_true) - np.pi/2) < 0.05   # premise: not anti-correlated
    assert abs(d['per_bc'] - TRUE_PO['per_bc']) < 30.0
    assert np.sum(res.fun**2) / (len(tc) - len(spec)) < 2.0


def test_pair_flip_sets_phase_offsets_structure():
    # each pair must be flipped once (from the inner side), every generated
    # name must resolve in spec.index (optimize() does x0[spec.index[nm]] *= -1,
    # so a bad name is a KeyError on every phase-offsets fit), and a pair whose
    # outer planet does not transit gets only the inner variant
    from harmonic.fit import _pair_flip_sets
    for letters, nto, n_pairs, n_variants in [('bc', False, 1, 3),
                                              ('bcd', False, 2, 6),
                                              ('bcd', True, 2, 4)]:
        spec = _po_spec(letters, nto)
        groups = _pair_flip_sets(spec, True)
        assert len(groups) == n_pairs, letters
        assert sum(len(v) for v in groups) == n_variants, letters
        for name in (nm for g in groups for v in g for nm in v):
            assert name in spec.index, (letters, name)
    # non-transiting outer pair: inner-only variant, no outer as/ac to flip
    assert _pair_flip_sets(_po_spec('bcd', True), True)[1] == [['as_cd', 'ac_cd']]
