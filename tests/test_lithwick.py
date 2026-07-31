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


def make_chain_phase_offsets(n=2000, per_b=45.155, per_c=85.32, per_ttv=700.0,
                             a_in=0.01, a_out=0.01, seed=0):
    """Like make_chain, but with independent inner/outer amplitudes (as_bc/ac_bc,
    as_cb/ac_cb) instead of a shared-phase ratio r_cb: the shape --phase-offsets
    chains actually have."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'per_b': per_b + rng.normal(0, 1e-4, n),
        't0_b': rng.normal(100, 1e-3, n),
        'per_c': per_c + rng.normal(0, 1e-4, n),
        't0_c': rng.normal(100, 1e-3, n),
        'as_bc': a_in + rng.normal(0, 1e-4, n),
        'ac_bc': rng.normal(0, 1e-4, n),
        'as_cb': a_out + rng.normal(0, 1e-4, n),
        'ac_cb': rng.normal(0, 1e-4, n),
        'per_bc': per_ttv + rng.normal(0, 1.0, n),
    })


def make_chain_bcd(per_d=89.586, seed=0, n=2000):
    """3-planet chain (pairs bc, cd): bc is the same 2:1-ish pair as make_chain
    so it is never the one under test; per_d controls the cd period ratio
    (default 89.586/85.32 = 1.05, far from every supported MMR)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'per_b': 45.155 + rng.normal(0, 1e-4, n),
        't0_b': rng.normal(100, 1e-3, n),
        'per_c': 85.32 + rng.normal(0, 1e-4, n),
        't0_c': rng.normal(100, 1e-3, n),
        'per_d': per_d + rng.normal(0, 1e-4, n),
        't0_d': rng.normal(100, 1e-3, n),
        'as_bc': 0.01 + rng.normal(0, 1e-4, n),
        'ac_bc': rng.normal(0, 1e-4, n),
        'r_cb': -2.0 + rng.normal(0, 0.01, n),
        'per_bc': 700.0 + rng.normal(0, 1.0, n),
        'per_cd': 900.0 + rng.normal(0, 1.0, n),
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
        df = print_constraints(make_chain(), 'bc', False, seed=1)
        # 2:1-ish pair (85.32/45.155 = 1.889 -> j=2); both planets constrained
        assert set(df.planet) == {'b', 'c'}
        assert (df.j == 2).all()
        assert (df.mass_me > 0).all()

    def test_mstar_scales_mass_up(self):
        m1 = print_constraints(make_chain(), 'bc', False, mstar=1.0, seed=1)
        m2 = print_constraints(make_chain(), 'bc', False, mstar=2.0, seed=1)
        # audit bug: code divided by mstar; mass must scale UP with mstar
        np.testing.assert_allclose(m2.mass_me.values, 2 * m1.mass_me.values, rtol=1e-6)

    def test_seed_reproducible(self):
        a = print_constraints(make_chain(), 'bc', False, seed=7)
        b = print_constraints(make_chain(), 'bc', False, seed=7)
        pd.testing.assert_frame_equal(a, b)

    def test_non_transiting_outer_pair_analyzed(self):
        fc = make_chain().drop(columns=['per_c', 't0_c', 'r_cb'])
        ephem = pd.DataFrame({'per': [45.155, 85.32], 'tc': [100., 100.]}, index=['b', 'c'])
        df = print_constraints(fc, 'bc', True, ephem=ephem, seed=1)
        # audit bug: this pair was silently skipped; inner amplitude constrains outer mass
        assert 'c' in set(df.planet)

    def test_phase_offsets_both_directions_constrained(self):
        # same 2:1-ish pair as test_returns_dataframe_both_directions, but with
        # independent as_cb/ac_cb columns instead of r_cb (the --phase-offsets
        # chain shape); a column-name bug here would raise KeyError
        df = print_constraints(make_chain_phase_offsets(), 'bc', False, phase_offsets=True, seed=1)
        assert set(df.planet) == {'b', 'c'}
        assert (df.j == 2).all()
        assert (df.mass_me > 0).all()

    def test_phase_offsets_outer_amplitude_column_drives_inner_mass(self):
        # regression: with phase_offsets=True, the outer planet's OBSERVED
        # amplitude must be read from as_cb/ac_cb (not r_cb, which does not
        # exist in this chain shape) and it constrains the INNER planet's mass
        # (eq. 9); a swapped column or sign would leave 'b' mass insensitive to
        # a_out, or make it decrease instead of increase
        small = print_constraints(make_chain_phase_offsets(a_out=0.006), 'bc', False,
                                  phase_offsets=True, seed=1)
        big = print_constraints(make_chain_phase_offsets(a_out=0.02), 'bc', False,
                                phase_offsets=True, seed=1)
        m_small = small[small.planet == 'b'].mass_me.iloc[0]
        m_big = big[big.planet == 'b'].mass_me.iloc[0]
        assert m_big > m_small

    def test_non_transiting_outer_pair_skipped_without_ephem(self, caplog):
        # ephem=None is the kwarg default: a non-transiting outer pair with no
        # ephem supplied must be skipped with a warning, while the other pair
        # (bc, fully transiting) still returns results
        import logging
        with caplog.at_level(logging.WARNING, logger='harmonic.lithwick'):
            df = print_constraints(make_chain(), 'bcd', True, seed=1)
        assert set(df.planet) == {'b', 'c'}
        assert (df.j == 2).all()
        assert any('cd' in r.message and 'ephem' in r.message for r in caplog.records)

    def test_pair_with_too_few_accepted_samples_warns(self, caplog):
        # cd sits near 2:1 (170/85.32) but its observed TTV amplitude (5 d) is
        # far larger than any (mu, |Zfree|) prior draw can produce, so the ABC
        # step accepts nothing and both cd rows are dropped: that must warn
        # rather than vanish silently, and bc must still return both planets
        import logging
        fc = make_chain_bcd(per_d=170.0)
        rng = np.random.default_rng(1)
        n = len(fc)
        fc['as_cd'] = 5.0 + rng.normal(0, 1e-3, n)
        fc['ac_cd'] = rng.normal(0, 1e-3, n)
        fc['r_dc'] = -2.0 + rng.normal(0, 0.01, n)
        with caplog.at_level(logging.WARNING, logger='harmonic.lithwick'):
            df = print_constraints(fc, 'bcd', False, seed=1)
        assert set(df.planet) == {'b', 'c'}
        assert any('cd' in r.message and 'ABC' in r.message for r in caplog.records)

    def test_pair_off_resonance_skipped(self, caplog):
        # cd's period ratio (1.05) is far from every supported MMR: that pair
        # must be skipped with a warning, while bc (2:1-ish) still returns
        # results for both planets
        import logging
        with caplog.at_level(logging.WARNING, logger='harmonic.lithwick'):
            df = print_constraints(make_chain_bcd(), 'bcd', False, seed=1)
        assert set(df.planet) == {'b', 'c'}
        assert (df.j == 2).all()
        assert any('cd' in r.message and 'MMR' in r.message for r in caplog.records)
