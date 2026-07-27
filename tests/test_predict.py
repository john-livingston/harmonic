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
    df = scan_transits(fc, ephem, 'bc', False, {'b': 0.2, 'c': 0.3}, [w0, w1], t_offset=t_offset)
    b = df[df.planet == 'b']
    assert len(b) == 1 and int(b.epoch.iloc[0]) == 20
    assert abs(b.tc_bjd.iloc[0] - (w0.jd + 1.0)) < 0.05

def test_empty_window():
    fc, ephem = chain_and_ephem()
    t_offset = 2454833.0
    w0 = Time(100.0 + 45.155*20.5 + t_offset, format='jd')
    w1 = Time(w0.jd + 0.01, format='jd')
    df = scan_transits(fc, ephem, 'bc', False, {'b': 0.01, 'c': 0.01}, [w0, w1], t_offset=t_offset)
    assert len(df) == 0

def test_bounded_even_with_nan_chain():
    fc, ephem = chain_and_ephem()
    fc.loc[:, 'per_bc'] = np.nan     # audit: old while-True looped forever on NaN
    t_offset = 2454833.0
    w0 = Time(1000.0 + t_offset, format='jd'); w1 = Time(1002.0 + t_offset, format='jd')
    df = scan_transits(fc, ephem, 'bc', False, {'b': 0.2, 'c': 0.3}, [w0, w1], t_offset=t_offset)
    assert len(df) < 100  # returns, bounded; NaN rows dropped

def _cfg(text):
    import configparser
    c = configparser.ConfigParser()
    c.read_string(text)
    return c


def test_t14_bare_letters_resolve():
    # 'C' is lowercased by configparser itself before we ever see it; entries
    # for planets outside this fit are ignored, so one config can serve fits
    # over different subsets (e.g. with and without -n)
    from harmonic.harmonic import _resolve_t14s
    cfg = _cfg("[T14]\nb = 0.24\nC = 0.12\nd = 0.33\nz = 0.99\n")
    assert list(cfg['T14'].keys()) == ['b', 'c', 'd', 'z']
    assert _resolve_t14s(cfg, 'bcd') == {'b': 0.24, 'c': 0.12, 'd': 0.33}
    assert _resolve_t14s(cfg, 'bc') == {'b': 0.24, 'c': 0.12}


def test_t14_matches_uppercase_planet_letters():
    # since configparser lowercases the config side, the case-insensitivity
    # that matters is on the letters side: `-l BC` must still resolve, and the
    # result must be keyed by the letters actually used (scan_transits indexes
    # t14s with the planet values from the data frame)
    from harmonic.harmonic import _resolve_t14s
    cfg = _cfg("[T14]\nb = 0.24\nc = 0.12\n")
    assert _resolve_t14s(cfg, 'BC') == {'B': 0.24, 'C': 0.12}


def test_t14_non_letter_keys_rejected():
    # regression: scan_transits indexes t14s by bare planet letter, so keys
    # like planet_b used to surface as a bare KeyError('b') from inside the
    # scan. They must be rejected here, naming the offending key.
    from harmonic.harmonic import _resolve_t14s
    from harmonic.exceptions import ConfigurationError
    with pytest.raises(ConfigurationError) as e:
        _resolve_t14s(_cfg("[T14]\nplanet_b = 0.24\nplanet_c = 0.12\n"), 'bc')
    assert 'planet_b' in str(e.value) and 'planet_c' in str(e.value)
    assert '(b, c)' in str(e.value)          # states what was expected
    # one bad key among good ones is still rejected
    with pytest.raises(ConfigurationError) as e:
        _resolve_t14s(_cfg("[T14]\nb = 0.24\nkepler-51 c = 0.12\n"), 'bc')
    assert 'kepler-51 c' in str(e.value)


def test_t14_missing_planet_rejected():
    from harmonic.harmonic import _resolve_t14s
    from harmonic.exceptions import ConfigurationError
    with pytest.raises(ConfigurationError) as e:
        _resolve_t14s(_cfg("[T14]\nb = 0.24\n"), 'bc')
    assert 'no transit duration for planet(s): c' in str(e.value)


def test_t14_missing_section_or_bad_values_rejected():
    from harmonic.harmonic import _resolve_t14s
    from harmonic.exceptions import ConfigurationError
    with pytest.raises(ConfigurationError):
        _resolve_t14s(_cfg("[INIT]\na_bc = 0.01\n"), 'bc')
    with pytest.raises(ConfigurationError):
        _resolve_t14s(_cfg("[T14]\nb = wide\nc = 0.12\n"), 'bc')


def test_predict_rejects_bad_t14_keys_before_scanning(tmp_path):
    # end-to-end: predict() must fail on the config, not with KeyError('b')
    # from inside scan_transits. Guards the wiring, not just the helper.
    import os, shutil
    from harmonic.harmonic import Harmonic
    from harmonic.exceptions import ConfigurationError
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shutil.copy(os.path.join(repo, 'examples/kep51.csv'), tmp_path / 'data.csv')
    cfg = open(os.path.join(repo, 'examples/kep51.ini')).read()
    (tmp_path / 'config.ini').write_text(cfg.replace('[T14]\nb =', '[T14]\nplanet_b ='))
    h = Harmonic(str(tmp_path / 'data.csv'), str(tmp_path / 'config.ini'), outdir=str(tmp_path))
    rng = np.random.default_rng(0)
    h.flatchain = pd.DataFrame(
        rng.normal(h.spec.x0 + h.spec.offset, 1e-4 * (h.spec.hi - h.spec.lo),
                   size=(50, len(h.spec))), columns=h.spec.names)
    h._chain_mismatch = None
    with pytest.raises(ConfigurationError) as e:
        h.predict(['2017-05-01 00:00', '2017-07-30 00:00'])
    assert 'planet_b' in str(e.value)


@pytest.mark.filterwarnings('ignore::erfa.ErfaWarning')
def test_straddling_transit_included():
    fc, ephem = chain_and_ephem()
    t_offset = 0.0
    center = 100.0 + 45.155*20
    w0 = Time(center + 0.05, format='jd')   # window starts during the transit (t14=0.2)
    w1 = Time(center + 1.0, format='jd')
    df = scan_transits(fc, ephem, 'bc', False, {'b': 0.2, 'c': 0.01}, [w0, w1], t_offset=0.0)
    assert 20 in set(df[df.planet == 'b'].epoch)
