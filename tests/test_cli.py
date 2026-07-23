import os
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@pytest.mark.slow
def test_kep51_fit_smoke(tmp_path):
    import sys as _sys
    from unittest.mock import patch
    from harmonic.harmonic import cli
    argv = ['harmonic', '-i', os.path.join(REPO, 'examples/kep51.csv'),
            '-c', os.path.join(REPO, 'examples/kep51.ini'),
            '-o', str(tmp_path), '-n', '-w', '48', '--steps', '60', '-b', '40',
            '--thin', '2', '--nproc', '1', '--seed', '1']
    with patch.object(_sys, 'argv', argv):
        cli()
    assert (tmp_path / 'samples.csv.gz').exists()
    assert (tmp_path / 'fit_config.json').exists()
    for f in ('corner.png', 'trace.png', 'fit.png', 'init.png'):
        assert (tmp_path / f).exists(), f
    fc = pd.read_csv(tmp_path / 'samples.csv.gz')
    assert len(fc) > 0 and any(c.startswith('as_') for c in fc.columns)
    assert (tmp_path / 'fit_stats.json').exists()
    import json
    stats = json.loads((tmp_path / 'fit_stats.json').read_text())
    assert set(stats) >= {'delta_bic', 'evidence', 'chi2_lin', 'chi2_harm',
                          'k_lin', 'k_harm', 'n_data', 'reduced_chi2', 'dof',
                          'accept_frac', 'tau_max', 'converged'}

def test_missing_t14_predict_errors(tmp_path, sample_data_file):
    import shutil
    from harmonic.harmonic import Harmonic
    from harmonic.exceptions import ConfigurationError, PredictionError
    shutil.copy(sample_data_file, tmp_path / 'data.csv')
    (tmp_path / 'config.ini').write_text('[INIT]\na_bc=0.01\na_cb=-0.01\nper_bc=100\nt_bc=2454900\n'
                                         'a_cd=0.01\na_dc=-0.01\nper_cd=200\nt_cd=2454950\n')
    h = Harmonic(letters='bcd', outdir=str(tmp_path))
    with pytest.raises((ConfigurationError, PredictionError)):
        h.predict(['2023-09-17 16:00', '2023-09-17 21:30'])

def test_noncontiguous_planet_ids_error(tmp_path, sample_config_file):
    import pandas as pd
    from harmonic.harmonic import Harmonic
    from harmonic.exceptions import DataError
    df = pd.DataFrame(dict(planet=[1, 1, 2, 2], epoch=[0, 1, 0, 1],
                           tc=[100., 145., 110., 195.], tc_unc=[0.01]*4))
    df.to_csv(tmp_path / 'd.csv', index=False)
    with pytest.raises(DataError):
        Harmonic(fp_data=str(tmp_path / 'd.csv'), fp_config=str(sample_config_file),
                 letters='bc', outdir=str(tmp_path))


def test_fit_config_round_trip(tmp_path):
    from harmonic.harmonic import _build_parser, _write_fit_config, _read_fit_config
    args = _build_parser().parse_args(
        ['-o', str(tmp_path), '-l', 'cdbe', '--phase-offsets', '--t-offset', '2454833'])
    _write_fit_config(str(tmp_path), args)
    assert (tmp_path / 'fit_config.json').exists()
    assert _read_fit_config(str(tmp_path)) == {
        'letters': 'cdbe', 'non_transiting_outer': False,
        'phase_offsets': True, 't_offset': 2454833}


def test_read_fit_config_none_when_missing(tmp_path):
    from harmonic.harmonic import _read_fit_config
    assert _read_fit_config(str(tmp_path)) is None


def test_read_fit_config_none_when_malformed(tmp_path):
    from harmonic.harmonic import _read_fit_config
    (tmp_path / 'fit_config.json').write_text('{not valid json')
    assert _read_fit_config(str(tmp_path)) is None
    (tmp_path / 'fit_config.json').write_text('{"letters": "cdbe"}')  # missing keys
    assert _read_fit_config(str(tmp_path)) is None


def test_resolve_predict_options_prefers_fit(tmp_path):
    from harmonic.harmonic import _build_parser, _write_fit_config, _resolve_predict_options
    parser = _build_parser()
    fit_args = parser.parse_args(
        ['-o', str(tmp_path), '-l', 'cdbe', '--phase-offsets', '-n', '--t-offset', '2454833'])
    _write_fit_config(str(tmp_path), fit_args)
    # predict invocation supplies only -o (the pain point being fixed)
    args = parser.parse_args(['-o', str(tmp_path), '--predict', 'a', 'b'])
    opts = _resolve_predict_options(args, parser)
    assert opts == {'letters': 'cdbe', 'non_transiting_outer': True,
                    'phase_offsets': True, 't_offset': 2454833}


def test_resolve_predict_options_warns_on_conflict(tmp_path, caplog):
    import logging
    from harmonic.harmonic import _build_parser, _write_fit_config, _resolve_predict_options
    parser = _build_parser()
    _write_fit_config(str(tmp_path),
                      parser.parse_args(['-o', str(tmp_path), '-l', 'cdbe', '--phase-offsets']))
    args = parser.parse_args(['-o', str(tmp_path), '-l', 'bcde', '--predict', 'a', 'b'])
    with caplog.at_level(logging.WARNING, logger='harmonic.harmonic'):
        opts = _resolve_predict_options(args, parser)
    assert opts['letters'] == 'cdbe'  # fit wins
    assert any('letters' in r.message and 'ignored' in r.message for r in caplog.records)


def test_resolve_predict_options_fallback_no_config(tmp_path):
    from harmonic.harmonic import _build_parser, _resolve_predict_options
    parser = _build_parser()
    args = parser.parse_args(['-o', str(tmp_path), '-l', 'bcde', '--predict', 'a', 'b'])
    opts = _resolve_predict_options(args, parser)
    assert opts['letters'] == 'bcde'  # no fit config -> CLI value


def test_fit_stats_round_trip(tmp_path):
    from harmonic.harmonic import _write_fit_stats, _read_fit_stats
    stats = {'delta_bic': 152.34, 'evidence': 'very strong', 'chi2_lin': 286.1,
             'chi2_harm': 133.8, 'k_lin': 6, 'k_harm': 14, 'n_data': 45,
             'reduced_chi2': 4.32, 'dof': 31, 'accept_frac': 0.42,
             'tau_max': 38.6, 'converged': True}
    _write_fit_stats(str(tmp_path), stats)
    assert (tmp_path / 'fit_stats.json').exists()
    assert _read_fit_stats(str(tmp_path)) == {
        'delta_bic': 152.34, 'evidence': 'very strong', 'chi2_lin': 286.1,
        'chi2_harm': 133.8, 'k_lin': 6, 'k_harm': 14, 'n_data': 45}


def test_read_fit_stats_none_when_missing_or_incomplete(tmp_path):
    from harmonic.harmonic import _read_fit_stats
    assert _read_fit_stats(str(tmp_path)) is None
    (tmp_path / 'fit_stats.json').write_text('{"delta_bic": 1.0}')  # missing keys
    assert _read_fit_stats(str(tmp_path)) is None


def test_delta_bic_recompute_when_no_stats(tmp_path):
    # no fit_stats.json and no stored stats -> method re-optimizes for the MLE.
    # A one-row chain with the right columns satisfies _require_chain; the
    # recompute path ignores the chain contents.
    from harmonic.harmonic import Harmonic
    h = Harmonic('examples/kep51.csv', 'examples/kep51.ini', outdir=str(tmp_path))
    h.flatchain = pd.DataFrame({n: [0.0] for n in h.spec.names})
    h._chain_mismatch = None
    h._fit_stats = None
    d = h.delta_bic()
    assert set(d) == {'delta_bic', 'evidence', 'chi2_lin', 'chi2_harm',
                      'k_lin', 'k_harm', 'n_data'}
    assert d['k_lin'] == 6 and d['k_harm'] == len(h.spec)
    assert d['delta_bic'] > 0  # kep51 has strong, well-detected TTVs


def test_delta_bic_uses_stored_stats(tmp_path):
    from harmonic.harmonic import Harmonic
    h = Harmonic('examples/kep51.csv', 'examples/kep51.ini', outdir=str(tmp_path))
    h.flatchain = pd.DataFrame({n: [0.0] for n in h.spec.names})
    h._chain_mismatch = None
    h._fit_stats = {'delta_bic': 99.0, 'evidence': 'very strong', 'chi2_lin': 1.0,
                    'chi2_harm': 0.5, 'k_lin': 6, 'k_harm': 14, 'n_data': 40}
    d = h.delta_bic()
    assert d['delta_bic'] == 99.0  # returned from stored stats, no recompute
