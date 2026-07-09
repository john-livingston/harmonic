import numpy as np
import pandas as pd
from harmonic.lithwick import choose_j, print_constraints


def make_chain(n=2000, per_b=45.155, per_c=85.32, per_ttv=700.0, a_in=0.01, r=-2.0, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'per_b': per_b + rng.normal(0, 1e-4, n),
        't0_b': rng.normal(100, 1e-3, n),
        'per_c': per_c + rng.normal(0, 1e-4, n),
        't0_c': rng.normal(100, 1e-3, n),
        'as_bc': a_in + rng.normal(0, 1e-4, n),
        'ac_bc': rng.normal(0, 1e-4, n),
        'r_cb': r + rng.normal(0, 0.01, n),
        'per_bc': per_ttv + rng.normal(0, 1.0, n),
    })


class TestChooseJ:
    def test_exact_ratios(self):
        assert choose_j(2.02) == 2
        assert choose_j(1.51) == 3
        assert choose_j(1.34) == 4
        assert choose_j(1.26) == 5

    def test_ratio_152_is_3_2_not_2_1(self):
        # audit bug: round(1.52) == 2 misclassified this as 2:1
        assert choose_j(1.52) == 3

    def test_far_from_resonance_returns_none(self):
        assert choose_j(1.05) is None
        assert choose_j(3.5) is None


class TestConstraints:
    def test_returns_dataframe_both_directions(self):
        df = print_constraints(make_chain(), 2, 'bc', False, seed=1)
        # 2:1-ish pair (85.32/45.155 = 1.889 -> j=2); both planets constrained
        assert set(df.planet) == {'b', 'c'}
        assert (df.j == 2).all()
        assert (df.mass_me > 0).all()

    def test_mstar_scales_mass_up(self):
        m1 = print_constraints(make_chain(), 2, 'bc', False, mstar=1.0, seed=1)
        m2 = print_constraints(make_chain(), 2, 'bc', False, mstar=2.0, seed=1)
        # audit bug: code divided by mstar; mass must scale UP with mstar
        np.testing.assert_allclose(m2.mass_me.values, 2 * m1.mass_me.values, rtol=1e-6)

    def test_seed_reproducible(self):
        a = print_constraints(make_chain(), 2, 'bc', False, seed=7)
        b = print_constraints(make_chain(), 2, 'bc', False, seed=7)
        pd.testing.assert_frame_equal(a, b)

    def test_non_transiting_outer_pair_analyzed(self):
        fc = make_chain().drop(columns=['per_c', 't0_c', 'r_cb'])
        ephem = pd.DataFrame({'per': [45.155, 85.32], 'tc': [100., 100.]}, index=['b', 'c'])
        df = print_constraints(fc, 2, 'bc', True, ephem=ephem, seed=1)
        # audit bug: this pair was silently skipped; inner amplitude constrains outer mass
        assert 'c' in set(df.planet)
