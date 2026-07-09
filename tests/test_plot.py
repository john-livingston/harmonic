import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from harmonic.plot import plot_trace, plot_corner, planet_colors, PALETTE

def test_planet_colors_stable():
    c = planet_colors('bcd')
    assert list(c) == ['b', 'c', 'd'] and c['b'] == PALETTE[0]

def test_plot_trace_returns_fig(tmp_path):
    chain = np.random.default_rng(0).normal(size=(50, 8, 3))
    fig = plot_trace(chain, ['$a$', '$b$', '$c$'], fp=str(tmp_path/'t.png'))
    assert (tmp_path/'t.png').exists() and fig is not None

def test_plot_trace_single_param(tmp_path):
    chain = np.random.default_rng(0).normal(size=(50, 8, 1))
    fig = plot_trace(chain, ['$a$'], fp=str(tmp_path/'t1.png'))  # audit: ndim==1 crashed
    assert (tmp_path/'t1.png').exists()

def test_plot_corner(tmp_path):
    fc = pd.DataFrame(np.random.default_rng(0).normal(size=(500, 3)), columns=list('abc'))
    fig = plot_corner(fc, labels=['$a$', '$b$', '$c$'], fp=str(tmp_path/'c.png'))
    assert (tmp_path/'c.png').exists()
