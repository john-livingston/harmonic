import numpy as np
import pandas as pd
import pytest
from astropy.time import Time
from harmonic.predict import scan_transits
from harmonic.model import model

def chain_and_ephem(n=200):
    rng = np.random.default_rng(0)
    base = {'t0_b': 100.0, 'per_b': 45.155, 't0_c': 110.0, 'per_c': 85.32,
            'as_bc': 0.01, 'ac_bc': -0.006, 'r_cb': -2.0, 'per_bc': 650.0}
    fc = pd.DataFrame({k: v + rng.normal(0, 1e-5, n) for k, v in base.items()})
    ephem = pd.DataFrame({'per': [45.155, 85.32], 'tc': [100.0, 110.0]}, index=['b', 'c'])
    return fc, ephem

def test_finds_transits_in_window():
    fc, ephem = chain_and_ephem()
    t_offset = 2454833.0
    w0 = Time(100.0 + 45.155*20 - 1.0 + t_offset, format='jd')
    w1 = Time(100.0 + 45.155*20 + 1.0 + t_offset, format='jd')
    df = scan_transits(fc, ephem, 2, 'bc', False, {'b': 0.2, 'c': 0.3}, [w0, w1], t_offset=t_offset)
    b = df[df.planet == 'b']
    assert len(b) == 1 and int(b.epoch.iloc[0]) == 20
    assert abs(b.tc_bjd.iloc[0] - (w0.jd + 1.0)) < 0.05

def test_empty_window():
    fc, ephem = chain_and_ephem()
    t_offset = 2454833.0
    w0 = Time(100.0 + 45.155*20.5 + t_offset, format='jd')
    w1 = Time(w0.jd + 0.01, format='jd')
    df = scan_transits(fc, ephem, 2, 'bc', False, {'b': 0.01, 'c': 0.01}, [w0, w1], t_offset=t_offset)
    assert len(df) == 0

def test_bounded_even_with_nan_chain():
    fc, ephem = chain_and_ephem()
    fc.loc[:, 'per_bc'] = np.nan     # audit: old while-True looped forever on NaN
    t_offset = 2454833.0
    w0 = Time(1000.0 + t_offset, format='jd'); w1 = Time(1002.0 + t_offset, format='jd')
    df = scan_transits(fc, ephem, 2, 'bc', False, {'b': 0.2, 'c': 0.3}, [w0, w1], t_offset=t_offset)
    assert len(df) < 100  # returns, bounded; NaN rows dropped

@pytest.mark.filterwarnings('ignore::erfa.ErfaWarning')
def test_straddling_transit_included():
    fc, ephem = chain_and_ephem()
    t_offset = 0.0
    center = 100.0 + 45.155*20
    w0 = Time(center + 0.05, format='jd')   # window starts during the transit (t14=0.2)
    w1 = Time(center + 1.0, format='jd')
    df = scan_transits(fc, ephem, 2, 'bc', False, {'b': 0.2, 'c': 0.01}, [w0, w1], t_offset=0.0)
    assert 20 in set(df[df.planet == 'b'].epoch)
