import numpy as np
import pandas as pd


def residual(pa, planet, epoch, tc, tc_err, nplanets, planet_letters, non_transiting_outer, phase_offsets=False, t_ref=0.0):
    return (tc - model(pa, planet, epoch, nplanets, planet_letters, non_transiting_outer, phase_offsets, t_ref=t_ref))/tc_err

class TTV(object):
    def __init__(self,times):
        self.times = times
        g = times.groupby('planet')
        lin = g.apply(lambda planet: np.polyfit(planet.epoch, planet.tc, 1))
        per = lin.apply(lambda x: x[0])
        tc = lin.apply(lambda x: x[1])
        ephem = pd.DataFrame(dict(per=per,tc=tc))
        ephem = ephem.sort_values(by='per')
        self.g = g
        self.ephem = ephem
        self.nplanets = len(ephem)
    def plot_times(self, planet, ax=None, **kwargs):
        if ax is None:
            raise ValueError("ax is required")
        idx = self.g.groups[planet]
        _times = self.times.loc[idx]
        lintime = np.polyval(np.array(self.ephem.loc[planet]), _times.epoch)
        minperday = 24*60
        ax.errorbar(_times.tc, (_times.tc - lintime)*minperday,
                    ls='', marker='s', yerr=_times.tc_unc * minperday, **kwargs)
    def plot_model(self, planet, tc, epoch, ax=None, **kwargs):
        if ax is None:
            raise ValueError("ax is required")
        lintime = np.polyval(np.array(self.ephem.loc[planet]), epoch)
        minperday = 24*60
        ax.plot(tc, (tc - lintime)*minperday, **kwargs)


def _terms(planet_letters, non_transiting_outer):
    transiting = planet_letters[:-1] if non_transiting_outer else planet_letters
    pairs = list(zip(planet_letters[:-1], planet_letters[1:]))
    return transiting, pairs


def model(p, planet, epoch, nplanets, planet_letters, non_transiting_outer, phase_offsets=False, t_ref=0.0):
    # t_ref: fixed phase pivot (data-frame time). Referencing the sinusoid
    # angle to the data epoch keeps the per_ttv likelihood smooth when tc is
    # in an absolute frame like BJD (with t_ref=0, dtheta/dper ~ tlin/per**2
    # makes the likelihood oscillate with spacing per**2/tlin in per_ttv).
    planet = np.asarray(planet)
    epoch = np.asarray(epoch, dtype=float)
    transiting, pairs = _terms(planet_letters, non_transiting_outer)
    tlin = np.zeros(len(epoch))
    for pl in transiting:
        m = planet == pl
        tlin[m] = p[f't0_{pl}'] + p[f'per_{pl}'] * epoch[m]
    tc = tlin.copy()
    for p_i, p_j in pairs:
        pair = f'{p_i}{p_j}'
        for pl in (p_i, p_j):
            if pl not in transiting:
                continue
            m = planet == pl
            if not m.any():
                continue
            th = 2 * np.pi * (tlin[m] - t_ref) / p[f'per_{pair}']
            if pl == p_i or not phase_offsets:
                s = p[f'as_{pair}'] * np.sin(th) + p[f'ac_{pair}'] * np.cos(th)
                if pl == p_j:
                    s = p[f'r_{p_j}{p_i}'] * s
            else:
                s = p[f'as_{p_j}{p_i}'] * np.sin(th) + p[f'ac_{p_j}{p_i}'] * np.cos(th)
            tc[m] += s
    return tc


def jacobian(p, names, planet, epoch, nplanets, planet_letters, non_transiting_outer, phase_offsets=False, t_ref=0.0):
    planet = np.asarray(planet)
    epoch = np.asarray(epoch, dtype=float)
    transiting, pairs = _terms(planet_letters, non_transiting_outer)
    col = {nm: k for k, nm in enumerate(names)}
    n = len(epoch)
    J = np.zeros((n, len(names)))
    tlin = np.zeros(n)
    for pl in transiting:
        m = planet == pl
        tlin[m] = p[f't0_{pl}'] + p[f'per_{pl}'] * epoch[m]
    dt = np.ones(n)  # d tc / d tlin, accumulated over pair terms
    for p_i, p_j in pairs:
        pair = f'{p_i}{p_j}'
        P = p[f'per_{pair}']
        for pl in (p_i, p_j):
            if pl not in transiting:
                continue
            m = planet == pl
            if not m.any():
                continue
            th = 2 * np.pi * (tlin[m] - t_ref) / P
            s, c = np.sin(th), np.cos(th)
            if pl == p_i or not phase_offsets:
                As, Ac = p[f'as_{pair}'], p[f'ac_{pair}']
                fac = 1.0 if pl == p_i else p[f'r_{p_j}{p_i}']
                J[m, col[f'as_{pair}']] += fac * s
                J[m, col[f'ac_{pair}']] += fac * c
                if pl == p_j:
                    J[m, col[f'r_{p_j}{p_i}']] += As * s + Ac * c
            else:
                As, Ac = p[f'as_{p_j}{p_i}'], p[f'ac_{p_j}{p_i}']
                fac = 1.0
                J[m, col[f'as_{p_j}{p_i}']] += s
                J[m, col[f'ac_{p_j}{p_i}']] += c
            dval = fac * (As * c - Ac * s)          # d(term)/d th
            J[m, col[f'per_{pair}']] += dval * (-2 * np.pi * (tlin[m] - t_ref) / P**2)
            dt[m] += dval * (2 * np.pi / P)
    for pl in transiting:
        m = planet == pl
        J[m, col[f't0_{pl}']] = dt[m]
        J[m, col[f'per_{pl}']] = dt[m] * epoch[m]
    return J
