import numpy as np
import pandas as pd
import pytest
from harmonic.model import model
from harmonic.params import build_spec
from harmonic.fit import run_fit, optimize

TRUE = {'t0_b': 100.0, 'per_b': 45.155, 't0_c': 110.0, 'per_c': 85.32,
        'as_bc': 0.010, 'ac_bc': -0.006, 'r_cb': -2.0, 'per_bc': 650.0}

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
    h = Harmonic('examples/kep51.csv', 'examples/kep51.ini', outdir=str(tmp_path))
    t = h.times
    res = optimize(h.spec, np.array(t.planet), np.array(t.epoch), np.array(t.tc),
                   np.array(t.tc_unc), h.planet_letters, False, False)
    dof = len(t) - len(h.spec)
    assert np.sum(res.fun**2) / dof < 4.0
    d = h.spec.to_dict(res.x)
    assert abs(d['r_dc']) < 19.0  # not railed at R_MAX


def test_walker_ball_independent_rank_deficient_jtj():
    # regression: with --phase-offsets + non-transiting the kep51-ttv-new fit
    # has a rank-deficient JtJ (an unconstrained parameter direction); pinv then
    # gave a walker covariance with a zero-spread direction -> linearly dependent
    # walkers -> emcee rejects the initial state ("large condition number").
    import os
    from emcee.ensemble import walkers_independent
    from harmonic.harmonic import Harmonic
    from harmonic.fit import optimize, _walker_ball
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = os.path.join(repo, 'examples', 'kep51-ttv-new.csv')
    if not os.path.exists(data):
        import pytest
        pytest.skip('kep51-ttv-new.csv not present')
    h = Harmonic(data, os.path.join(repo, 'examples', 'kep51.ini'),
                 outdir='/tmp/kep51n_test', non_transiting_outer=True, phase_offsets=True)
    t = h.times
    a = (np.array(t.planet), np.array(t.epoch), np.array(t.tc), np.array(t.tc_unc))
    res = optimize(h.spec, *a, h.planet_letters, True, True)
    p0 = _walker_ball(res, h.spec, 2 * len(h.spec), np.random.default_rng(42))
    assert np.all((p0 > h.spec.lo) & (p0 < h.spec.hi))
    assert walkers_independent(p0)  # the exact check emcee runs at init
