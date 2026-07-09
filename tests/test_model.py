import numpy as np
import pytest
from harmonic.model import model, jacobian

LET = 'bc'

def params(phase_offsets=False):
    p = {'t0_b': 100.0, 'per_b': 45.155, 't0_c': 110.0, 'per_c': 85.32,
         'as_bc': 0.008, 'ac_bc': -0.005, 'per_bc': 700.0}
    if phase_offsets:
        p.update({'as_cb': -0.015, 'ac_cb': 0.011})
    else:
        p['r_cb'] = -2.1
    return p

def data():
    planet = np.array(['b']*12 + ['c']*7)
    epoch = np.array(list(range(12)) + list(range(7)), dtype=float)
    return planet, epoch

def test_linear_plus_sinusoid_by_hand():
    p, (planet, epoch) = params(), data()
    tc = model(p, planet, epoch, LET, False)
    tlin_b = p['t0_b'] + p['per_b']*epoch[:12]
    th = 2*np.pi*tlin_b/p['per_bc']
    expected_b = tlin_b + p['as_bc']*np.sin(th) + p['ac_bc']*np.cos(th)
    np.testing.assert_allclose(tc[:12], expected_b, rtol=0, atol=1e-12)
    tlin_c = p['t0_c'] + p['per_c']*epoch[12:]
    th = 2*np.pi*tlin_c/p['per_bc']
    expected_c = tlin_c + p['r_cb']*(p['as_bc']*np.sin(th) + p['ac_bc']*np.cos(th))
    np.testing.assert_allclose(tc[12:], expected_c, rtol=0, atol=1e-12)

def test_phase_from_linear_ephemeris_not_perturbed():
    # doubling the inner amplitude must NOT change the argument of the sinusoid:
    # tc response must be exactly linear in as/ac (audit: old model was self-referential)
    p, (planet, epoch) = params(), data()
    base = model(dict(p, as_bc=0.0, ac_bc=0.0), planet, epoch, LET, False)
    d1 = model(dict(p, as_bc=0.008, ac_bc=0.0), planet, epoch, LET, False) - base
    d2 = model(dict(p, as_bc=0.016, ac_bc=0.0), planet, epoch, LET, False) - base
    np.testing.assert_allclose(d2, 2*d1, rtol=0, atol=1e-12)

@pytest.mark.parametrize('phase_offsets', [False, True])
@pytest.mark.parametrize('nto', [False, True])
def test_jacobian_matches_numeric(phase_offsets, nto):
    p = params(phase_offsets)
    planet, epoch = data()
    if nto:
        p = {k: v for k, v in p.items() if k not in ('t0_c', 'per_c', 'r_cb', 'as_cb', 'ac_cb')}
        planet, epoch = planet[:12], epoch[:12]
    names = sorted(p)
    J = jacobian(p, names, planet, epoch, LET, nto, phase_offsets)
    for k, nm in enumerate(names):
        h = 1e-6 * max(1.0, abs(p[nm]))
        hi = model(dict(p, **{nm: p[nm]+h}), planet, epoch, LET, nto, phase_offsets)
        lo = model(dict(p, **{nm: p[nm]-h}), planet, epoch, LET, nto, phase_offsets)
        np.testing.assert_allclose(J[:, k], (hi-lo)/(2*h), rtol=1e-5, atol=1e-9,
                                   err_msg=f'd/d{nm}')

def test_three_planet_middle_gets_both_pairs():
    p = {'t0_b': 100.0, 'per_b': 45.155, 't0_c': 110.0, 'per_c': 85.32,
         't0_d': 120.0, 'per_d': 130.18,
         'as_bc': 0.008, 'ac_bc': -0.005, 'r_cb': -2.1, 'per_bc': 700.0,
         'as_cd': 0.02, 'ac_cd': 0.01, 'r_dc': -1.5, 'per_cd': 1500.0}
    planet = np.array(['c']*5); epoch = np.arange(5.0)
    tc = model(p, planet, epoch, 'bcd', False)
    tlin = p['t0_c'] + p['per_c']*epoch
    th1 = 2*np.pi*tlin/p['per_bc']; th2 = 2*np.pi*tlin/p['per_cd']
    expected = (tlin + p['r_cb']*(p['as_bc']*np.sin(th1) + p['ac_bc']*np.cos(th1))
                + p['as_cd']*np.sin(th2) + p['ac_cd']*np.cos(th2))
    np.testing.assert_allclose(tc, expected, atol=1e-12)


def test_time_shift_invariance_with_tref():
    # model in a shifted time frame (e.g. BKJD -> BJD) with t_ref set must
    # reproduce the unshifted model exactly, just offset by the shift
    p, (planet, epoch) = params(), data()
    base = model(p, planet, epoch, LET, False)
    S = 2454833.0
    p2 = dict(p, t0_b=p['t0_b'] + S, t0_c=p['t0_c'] + S)
    shifted = model(p2, planet, epoch, LET, False, t_ref=S)
    np.testing.assert_allclose(shifted, base + S, rtol=0, atol=1e-6)


def test_jacobian_shift_invariance_with_tref():
    # exact property: shifting the time frame and t_ref together leaves every
    # Jacobian column unchanged (all dependence is through tlin - t_ref);
    # combined with the t_ref=0 numeric checks above this validates t_ref != 0
    p, (planet, epoch) = params(), data()
    names = sorted(p)
    J0 = jacobian(p, names, planet, epoch, LET, False)
    S = 2454833.0
    p2 = dict(p, t0_b=p['t0_b'] + S, t0_c=p['t0_c'] + S)
    J1 = jacobian(p2, names, planet, epoch, LET, False, t_ref=S)
    np.testing.assert_allclose(J1, J0, rtol=1e-9, atol=1e-12)
