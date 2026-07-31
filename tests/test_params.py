# tests/test_params.py
import numpy as np
import pandas as pd
import pytest
from harmonic.params import build_spec, ParamSpec, derived_frame

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
    spec = build_spec(*system, 'bc')
    assert spec.names == ['t0_b', 'per_b', 'as_bc', 'ac_bc', 'r_cb', 'per_bc', 't0_c', 'per_c']

def test_x0_within_bounds_and_pttv_logscale(system):
    spec = build_spec(*system, 'bc')
    assert np.all(spec.x0 > spec.lo) and np.all(spec.x0 < spec.hi)
    assert 'per_bc' in spec.log_scale
    assert spec.lo[spec.index['per_bc']] == pytest.approx(5*85.32)

def test_phase_offsets_names(system):
    p_init, ephem, times = system
    p_init = dict(p_init, phi_bc=0.3)
    spec = build_spec(p_init, ephem, times, 'bc', phase_offsets=True)
    assert 'as_cb' in spec.names and 'ac_cb' in spec.names and 'r_cb' not in spec.names

def test_non_transiting_outer_names(system):
    p_init, ephem, times = system
    times = times[times.planet == 'b']
    spec = build_spec(p_init, ephem, times, 'bc', non_transiting_outer=True)
    assert spec.names == ['t0_b', 'per_b', 'as_bc', 'ac_bc', 'per_bc']

def test_zero_inner_amplitude_raises(system):
    p_init, ephem, times = system
    p_init = dict(p_init, a_bc=0.0)
    with pytest.raises(Exception, match='a_bc'):
        build_spec(p_init, ephem, times, 'bc')


@pytest.mark.parametrize('key', ['a_bc', 'per_bc', 't_bc', 'a_cb'])
def test_missing_init_key_raises(system, key):
    from harmonic.exceptions import ConfigurationError
    p_init, ephem, times = system
    p_init = {k: v for k, v in p_init.items() if k != key}
    with pytest.raises(ConfigurationError, match=key) as e:
        build_spec(p_init, ephem, times, 'bc')
    assert 'pair bc' in str(e.value)


def test_missing_outer_amplitude_ignored_when_outer_does_not_transit(system):
    # a_cb is only needed when the outer planet of the pair transits
    p_init, ephem, times = system
    p_init = {k: v for k, v in p_init.items() if k != 'a_cb'}
    spec = build_spec(p_init, ephem, times[times.planet == 'b'], 'bc',
                      non_transiting_outer=True)
    assert spec.names == ['t0_b', 'per_b', 'as_bc', 'ac_bc', 'per_bc']


def test_spec_tref_is_rounded_data_median(system):
    p_init, ephem, times = system
    spec = build_spec(p_init, ephem, times, 'bc')
    assert spec.t_ref == round(float(times.tc.median()))


def test_conversion_with_tref_matches_sine_form(system):
    # a*sin(2*pi*(tlin - t_ttv)/P) == as*sin(th) + ac*cos(th) with
    # th = 2*pi*(tlin - t_ref)/P requires delta = 2*pi*(t_ref - t_ttv)/P
    p_init, ephem, times = system
    spec = build_spec(p_init, ephem, times, 'bc')
    d = spec.to_dict(spec.x0)
    delta = 2*np.pi*(spec.t_ref - p_init['t_bc'])/p_init['per_bc']
    np.testing.assert_allclose(d['as_bc'], 0.01*np.cos(delta), rtol=1e-12)
    np.testing.assert_allclose(d['ac_bc'], 0.01*np.sin(delta), rtol=1e-12)
    np.testing.assert_allclose(d['r_cb'], -2.0, rtol=1e-12)


def test_derived_frame_shared_phase():
    # phase_offsets=False: outer amplitude a_cb comes from the shared-phase
    # ratio r_cb times the inner amplitude a_bc
    fc = pd.DataFrame({'as_bc': [0.03], 'ac_bc': [0.04], 'r_cb': [-2.0]})
    d = derived_frame(fc, 'bc', False, False)
    assert list(d.columns) == ['a_bc', 'phase_bc', 'a_cb']
    assert d['a_bc'].iloc[0] == pytest.approx(0.05)  # hypot(0.03, 0.04)
    assert d['phase_bc'].iloc[0] == pytest.approx(np.arctan2(0.04, 0.03))
    assert d['a_cb'].iloc[0] == pytest.approx(2.0 * 0.05)  # |r_cb| * a_bc


def test_derived_frame_phase_offsets():
    # phase_offsets=True: outer amplitude a_cb comes from its own as_cb/ac_cb,
    # independent of the inner pair's amplitude
    fc = pd.DataFrame({'as_bc': [0.03], 'ac_bc': [0.04], 'as_cb': [0.06], 'ac_cb': [0.08]})
    d = derived_frame(fc, 'bc', False, True)
    assert list(d.columns) == ['a_bc', 'phase_bc', 'a_cb']
    assert d['a_bc'].iloc[0] == pytest.approx(0.05)   # hypot(0.03, 0.04)
    assert d['a_cb'].iloc[0] == pytest.approx(0.10)   # hypot(0.06, 0.08)
    assert d['phase_bc'].iloc[0] == pytest.approx(np.arctan2(0.04, 0.03))


def test_derived_frame_non_transiting_outer_skips_outer_amplitude():
    # the non-transiting outer planet has no as/ac or r column at all, so
    # derived_frame must not try to derive an outer amplitude for that pair
    fc = pd.DataFrame({'as_bc': [0.03], 'ac_bc': [0.04]})
    d = derived_frame(fc, 'bc', True, False)
    assert list(d.columns) == ['a_bc', 'phase_bc']
