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
    for f in ('corner.png', 'trace.png', 'fit.png', 'init.png'):
        assert (tmp_path / f).exists(), f
    fc = pd.read_csv(tmp_path / 'samples.csv.gz')
    assert len(fc) > 0 and any(c.startswith('as_') for c in fc.columns)

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
