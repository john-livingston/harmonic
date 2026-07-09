"""
Test configuration and fixtures for harmonic package tests.

This module provides shared fixtures and configuration for all tests,
including sample data and temporary directories.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
import configparser


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_data():
    """Create sample transit timing data for testing."""
    # Generate realistic Kepler-51 style data
    np.random.seed(42)  # For reproducible tests

    data = []
    # Planet b data (12 transits)
    for i in range(12):
        tc = 2454833.0 + i * 45.15 + np.random.normal(0, 0.001)
        tc_unc = 0.002 + np.random.uniform(-0.0005, 0.0005)
        data.append({'planet': 0, 'epoch': i, 'tc': tc, 'tc_unc': tc_unc})

    # Planet c data (8 transits)
    for i in range(8):
        tc = 2454833.0 + i * 85.31 + np.random.normal(0, 0.002)
        tc_unc = 0.003 + np.random.uniform(-0.001, 0.001)
        data.append({'planet': 1, 'epoch': i, 'tc': tc, 'tc_unc': tc_unc})

    # Planet d data (5 transits)
    for i in range(5):
        tc = 2454833.0 + i * 130.17 + np.random.normal(0, 0.003)
        tc_unc = 0.005 + np.random.uniform(-0.002, 0.002)
        data.append({'planet': 2, 'epoch': i, 'tc': tc, 'tc_unc': tc_unc})

    return pd.DataFrame(data)


@pytest.fixture
def sample_data_file(temp_dir, sample_data):
    """Create a temporary CSV file with sample data."""
    data_file = temp_dir / "test_data.csv"
    sample_data.to_csv(data_file, index=False)
    return data_file


def create_config_for_letters(letters='bcd'):
    """Helper function to create config dictionary for given planet letters."""
    config = {'INIT': {}, 'T14': {}}

    # Create INIT parameters for each pair of adjacent planets
    for i in range(len(letters) - 1):
        p_i = letters[i]
        p_j = letters[i + 1]
        config['INIT'][f'a_{p_i}{p_j}'] = '-0.01'
        config['INIT'][f'a_{p_j}{p_i}'] = '0.01'
        config['INIT'][f'per_{p_i}{p_j}'] = str(100 * (i + 1))
        config['INIT'][f't_{p_i}{p_j}'] = str(2454900 + 50 * i)

    # Create T14 (transit duration) for each planet
    transit_durations = [0.24, 0.12, 0.33, 0.2, 0.2]  # Days
    for i, letter in enumerate(letters):
        config['T14'][letter] = str(transit_durations[i % len(transit_durations)])

    return config


@pytest.fixture
def sample_config():
    """Create sample configuration dictionary."""
    return create_config_for_letters('bcd')


@pytest.fixture
def sample_config_file(temp_dir, sample_config):
    """Create a temporary INI configuration file."""
    config_file = temp_dir / "test_config.ini"

    config = configparser.ConfigParser()
    for section_name, section_data in sample_config.items():
        config.add_section(section_name)
        for key, value in section_data.items():
            config.set(section_name, key, value)

    with open(config_file, 'w') as f:
        config.write(f)

    return config_file


@pytest.fixture
def invalid_data_file(temp_dir):
    """Create an invalid CSV file for error testing."""
    invalid_file = temp_dir / "invalid_data.csv"
    # Create file with missing columns
    invalid_data = pd.DataFrame({
        'time': [1, 2, 3],
        'flux': [0.99, 1.01, 0.98]
    })
    invalid_data.to_csv(invalid_file, index=False)
    return invalid_file


@pytest.fixture
def empty_file(temp_dir):
    """Create an empty file for error testing."""
    empty_file = temp_dir / "empty.csv"
    empty_file.touch()
    return empty_file


@pytest.fixture
def corrupted_config_file(temp_dir):
    """Create a corrupted configuration file for error testing."""
    config_file = temp_dir / "corrupted_config.ini"
    with open(config_file, 'w') as f:
        f.write("This is not valid INI format\n")
        f.write("Missing sections and = signs\n")
        f.write("[incomplete section\n")
    return config_file


@pytest.fixture
def sample_time_window():
    """Create a sample time window for prediction testing."""
    return ["2023-09-17 16:00", "2023-09-17 21:30"]


@pytest.fixture(autouse=True)
def _reset_harmonic_logging():
    """setup_logging() sets propagate=False on the 'harmonic' logger; undo it
    after each test so caplog-based tests are order-independent."""
    yield
    import logging
    lg = logging.getLogger('harmonic')
    lg.handlers.clear()
    lg.propagate = True
    lg.setLevel(logging.NOTSET)
