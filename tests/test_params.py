# tests/test_params.py
import numpy as np
import pandas as pd
import pytest
from harmonic.params import build_spec, ParamSpec

@pytest.fixture
def system():
    rng = np.random.default_rng(0)
    rows = []
    for k, (per, t0) in enumerate([(45.155, 100.0), (85.32, 110.0)]):
        for e in range(15):
            rows.append(dict(planet='bc'[k], epoch=e, tc=t0 + per*e + rng.normal(0, 0.005), tc_unc=0.002))
    times = pd.DataFrame(rows)
    ephem = pd.DataFrame({'per': [45.155, 85.32], 'tc': [100.0, 110.0]}, index=['b', 'c'])
    p_init = {'a_bc': 0.01, 'a_cb': -0.02, 'per_bc': 700.0, 't_bc': 300.0}
    return p_init, ephem, times

def test_shared_phase_names(system):
    spec = build_spec(*system, 2, 'bc')
    assert spec.names == ['t0_b', 'per_b', 'as_bc', 'ac_bc', 'r_cb', 'per_bc', 't0_c', 'per_c']

def test_conversion_matches_sine_form(system):
    p_init, ephem, times = system
    spec = build_spec(p_init, ephem, times, 2, 'bc')
    d = spec.to_dict(spec.x0)
    delta = 2*np.pi*(spec.t_ref - p_init['t_bc'])/p_init['per_bc']
    np.testing.assert_allclose(d['as_bc'], 0.01*np.cos(delta), rtol=1e-12)
    np.testing.assert_allclose(d['ac_bc'], 0.01*np.sin(delta), rtol=1e-12)
    np.testing.assert_allclose(d['r_cb'], -2.0, rtol=1e-12)

def test_x0_within_bounds_and_pttv_logscale(system):
    spec = build_spec(*system, 2, 'bc')
    assert np.all(spec.x0 > spec.lo) and np.all(spec.x0 < spec.hi)
    assert 'per_bc' in spec.log_scale
    assert spec.lo[spec.index['per_bc']] == pytest.approx(5*85.32)

def test_phase_offsets_names(system):
    p_init, ephem, times = system
    p_init = dict(p_init, phi_bc=0.3)
    spec = build_spec(p_init, ephem, times, 2, 'bc', phase_offsets=True)
    assert 'as_cb' in spec.names and 'ac_cb' in spec.names and 'r_cb' not in spec.names

def test_non_transiting_outer_names(system):
    p_init, ephem, times = system
    times = times[times.planet == 'b']
    spec = build_spec(p_init, ephem, times, 2, 'bc', non_transiting_outer=True)
    assert spec.names == ['t0_b', 'per_b', 'as_bc', 'ac_bc', 'per_bc']

def test_zero_inner_amplitude_raises(system):
    p_init, ephem, times = system
    p_init = dict(p_init, a_bc=0.0)
    with pytest.raises(Exception, match='a_bc'):
        build_spec(p_init, ephem, times, 2, 'bc')


def test_spec_tref_is_rounded_data_median(system):
    p_init, ephem, times = system
    spec = build_spec(p_init, ephem, times, 2, 'bc')
    assert spec.t_ref == round(float(times.tc.median()))


def test_conversion_with_tref_matches_sine_form(system):
    # a*sin(2*pi*(tlin - t_ttv)/P) == as*sin(th) + ac*cos(th) with
    # th = 2*pi*(tlin - t_ref)/P requires delta = 2*pi*(t_ref - t_ttv)/P
    p_init, ephem, times = system
    spec = build_spec(p_init, ephem, times, 2, 'bc')
    d = spec.to_dict(spec.x0)
    delta = 2*np.pi*(spec.t_ref - p_init['t_bc'])/p_init['per_bc']
    np.testing.assert_allclose(d['as_bc'], 0.01*np.cos(delta), rtol=1e-12)
    np.testing.assert_allclose(d['ac_bc'], 0.01*np.sin(delta), rtol=1e-12)
