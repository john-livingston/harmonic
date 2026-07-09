"""Parameter registry: single source of truth for vector layout, bounds, labels."""
import numpy as np
import pandas as pd

from .exceptions import ConfigurationError

T0_WINDOW = 0.5      # d, +/- around linear-ephemeris t0
PER_FRAC = 0.01      # +/- fraction around linear-ephemeris period
AMP_FACTOR = 10.0    # amplitude bound = AMP_FACTOR * max(|O-C|, median unc)
R_MAX = 20.0         # |outer/inner amplitude ratio| bound
PTTV_MIN_FACTOR = 5.0   # x outer planet period -> per_ttv lower bound
PTTV_MAX_FACTOR = 20.0  # x data baseline       -> per_ttv upper bound


class ParamSpec:
    def __init__(self):
        self.names, self._x0, self._lo, self._hi = [], [], [], []
        self.t_ref = 0.0
        self.latex, self.log_scale = {}, set()

    def add(self, name, x0, lo, hi, latex, log=False):
        self.names.append(name)
        self._x0.append(x0); self._lo.append(lo); self._hi.append(hi)
        self.latex[name] = latex
        if log:
            self.log_scale.add(name)

    def freeze(self):
        self.x0 = np.asarray(self._x0, float)
        self.lo = np.asarray(self._lo, float)
        self.hi = np.asarray(self._hi, float)
        # Sample t0 as an O(1) offset from its (rounded) initial value:
        # absolute-BJD magnitudes (~2.45e6) in the parameter vector break
        # least_squares' norm-based termination and trust-region conditioning
        # next to day-scale amplitudes. The transform is internal: to_dict()
        # and the saved chain are always in absolute time.
        self.offset = np.where([n.startswith('t0_') for n in self.names],
                               np.round(self.x0), 0.0)
        self.x0 = self.x0 - self.offset
        self.lo = self.lo - self.offset
        self.hi = self.hi - self.offset
        pad = 1e-6 * (self.hi - self.lo)
        self.x0 = np.clip(self.x0, self.lo + pad, self.hi - pad)
        self.index = {n: i for i, n in enumerate(self.names)}
        return self

    def to_dict(self, theta):
        return dict(zip(self.names, np.asarray(theta) + self.offset))

    def labels(self):
        return [self.latex[n] for n in self.names]

    def __len__(self):
        return len(self.names)


def _pairs(planet_letters):
    return list(zip(planet_letters[:-1], planet_letters[1:]))


def build_spec(p_init, ephem, times, nplanets, planet_letters,
               non_transiting_outer=False, phase_offsets=False):
    spec = ParamSpec()
    spec.t_ref = float(round(float(times.tc.median())))
    transiting = planet_letters[:-1] if non_transiting_outer else planet_letters
    baseline = float(times.tc.max() - times.tc.min())
    med_unc = float(times.tc_unc.median())
    maxoc = {}
    for pl in transiting:
        t = times[times.planet == pl]
        oc = t.tc - (ephem.loc[pl, 'tc'] + ephem.loc[pl, 'per'] * t.epoch)
        maxoc[pl] = max(float(np.abs(oc).max()), med_unc)

    def add_planet(pl):
        t0, per = float(ephem.loc[pl, 'tc']), float(ephem.loc[pl, 'per'])
        spec.add(f't0_{pl}', t0, t0 - T0_WINDOW, t0 + T0_WINDOW, rf'$T_{{0,{pl}}}$')
        spec.add(f'per_{pl}', per, per * (1 - PER_FRAC), per * (1 + PER_FRAC), rf'$P_{pl}$')

    for i, (p_i, p_j) in enumerate(_pairs(planet_letters)):
        pair = f'{p_i}{p_j}'
        if p_i == planet_letters[0]:
            add_planet(p_i)
        a_in = float(p_init[f'a_{pair}'])
        if a_in == 0.0:
            raise ConfigurationError(f'a_{pair} must be nonzero in [INIT]')
        per_ttv = float(p_init[f'per_{pair}'])
        t_ttv = float(p_init[f't_{pair}'])
        phi = float(p_init.get(f'phi_{pair}', 0.0))
        delta = 2 * np.pi * (spec.t_ref - t_ttv) / per_ttv
        outer_transits = p_j in transiting
        amp = AMP_FACTOR * (maxoc[p_i] if not outer_transits
                            else max(maxoc[p_i], maxoc[p_j]))
        d_in = delta + phi if phase_offsets else delta
        spec.add(f'as_{pair}', a_in * np.cos(d_in), -amp, amp, rf'$A^{{\sin}}_{{{pair}}}$')
        spec.add(f'ac_{pair}', a_in * np.sin(d_in), -amp, amp, rf'$A^{{\cos}}_{{{pair}}}$')
        if outer_transits:
            a_out = float(p_init[f'a_{p_j}{p_i}'])
            if phase_offsets:
                d_out = delta - phi
                spec.add(f'as_{p_j}{p_i}', a_out * np.cos(d_out), -amp, amp, rf'$A^{{\sin}}_{{{p_j}{p_i}}}$')
                spec.add(f'ac_{p_j}{p_i}', a_out * np.sin(d_out), -amp, amp, rf'$A^{{\cos}}_{{{p_j}{p_i}}}$')
            else:
                spec.add(f'r_{p_j}{p_i}', a_out / a_in, -R_MAX, R_MAX, rf'$r_{{{p_j}{p_i}}}$')
        spec.add(f'per_{pair}', per_ttv, PTTV_MIN_FACTOR * float(ephem.loc[p_j, 'per']),
                 PTTV_MAX_FACTOR * baseline, rf'$P^{{\rm TTV}}_{{{pair}}}$', log=True)
        if outer_transits:
            add_planet(p_j)
    return spec.freeze()


def derived_frame(flatchain, planet_letters, non_transiting_outer, phase_offsets):
    out = {}
    transiting = planet_letters[:-1] if non_transiting_outer else planet_letters
    for p_i, p_j in _pairs(planet_letters):
        pair = f'{p_i}{p_j}'
        a = np.hypot(flatchain[f'as_{pair}'], flatchain[f'ac_{pair}'])
        out[f'a_{pair}'] = a
        out[f'phase_{pair}'] = np.arctan2(flatchain[f'ac_{pair}'], flatchain[f'as_{pair}'])
        if p_j in transiting:
            if phase_offsets:
                out[f'a_{p_j}{p_i}'] = np.hypot(flatchain[f'as_{p_j}{p_i}'], flatchain[f'ac_{p_j}{p_i}'])
            else:
                out[f'a_{p_j}{p_i}'] = np.abs(flatchain[f'r_{p_j}{p_i}']) * a
    return pd.DataFrame(out)
