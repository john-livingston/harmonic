import numpy as np
import pandas as pd
import logging
import matplotlib
import matplotlib.pyplot as plt
from astropy.time import Time

from .model import model
from .plot import planet_colors, _finish, STYLE

logger = logging.getLogger(__name__)

def trunc(x):
    return ':'.join(x.split('.')[0].split(':')[:-1])


def scan_transits(flatchain, ephem, planet_letters, non_transiting_outer,
                  t14s, window, phase_offsets=False, t_offset=0, nsamples=1000, seed=0,
                  t_ref=0.0):
    """
    Scan for predicted transit times within a time window.

    Parameters
    ----------
    flatchain : DataFrame
        MCMC samples (plain columns, e.g. t0_b, per_b, as_bc, ...).
    ephem : DataFrame
        Ephemeris data (in same time system as input data), indexed by planet letter.
    planet_letters : str
        Planet letter designations.
    non_transiting_outer : bool
        Whether there's a non-transiting outer planet.
    t14s : dict
        Transit durations (days) for each planet.
    window : list of Time objects
        Time window [start, end] as astropy Time (BJD).
    phase_offsets : bool
        Whether phase offsets are enabled.
    t_offset : float
        Timing offset to add to ephemeris/chain times to get BJD (default: 0 for
        BJD data, use 2454833 for BKJD data).
    nsamples : int
        Number of chain samples to draw (default: 1000).
    seed : int
        Random seed for sampling (default: 0).
    t_ref : float
        Phase-pivot time passed through to the model (matches the fit's spec.t_ref).

    Returns
    -------
    DataFrame
        Predicted transit times with columns: planet, epoch, tc_mean, tc_std,
        tc_median, tc_bjd, ingress, egress, ingress_iso, egress_iso, tc_iso.
    """
    letters_ = planet_letters[:-1] if non_transiting_outer else planet_letters
    w0, w1 = window[0].jd - t_offset, window[1].jd - t_offset
    samp = flatchain.sample(min(nsamples, len(flatchain)), random_state=seed)
    dicts = [dict(zip(samp.columns, row)) for row in samp.values]
    rows = []
    for planet in letters_:
        period = float(ephem.loc[planet, 'per'])
        t0 = float(ephem.loc[planet, 'tc'])
        t14 = t14s[planet]
        e_lo = int(np.floor((w0 - t14 / 2 - t0) / period)) - 1
        e_hi = int(np.ceil((w1 + t14 / 2 - t0) / period)) + 1
        epochs = np.arange(e_lo, e_hi + 1, dtype=float)
        planeti = np.array([planet] * len(epochs))
        preds = np.array([model(d, planeti, epochs, planet_letters,
                                non_transiting_outer, phase_offsets, t_ref=t_ref) for d in dicts])
        mn, sd, md = preds.mean(0), preds.std(0), np.median(preds, axis=0)
        for k, ep in enumerate(epochs):
            tc_bjd = md[k] + t_offset
            if not np.isfinite(tc_bjd):
                continue
            if tc_bjd + t14 / 2 < window[0].jd or tc_bjd - t14 / 2 > window[1].jd:
                continue
            rows.append(dict(planet=planet, epoch=int(ep), tc_mean=mn[k], tc_std=sd[k],
                             tc_median=md[k], tc_bjd=tc_bjd,
                             ingress=tc_bjd - t14 / 2, egress=tc_bjd + t14 / 2,
                             ingress_iso=Time(tc_bjd - t14 / 2, format='jd').iso,
                             egress_iso=Time(tc_bjd + t14 / 2, format='jd').iso,
                             tc_iso=Time(tc_bjd, format='jd').iso))
    cols = ['planet', 'epoch', 'tc_mean', 'tc_std', 'tc_median', 'tc_bjd',
            'ingress', 'egress', 'ingress_iso', 'egress_iso', 'tc_iso']
    return pd.DataFrame(rows, columns=cols)


get_transit_list = scan_transits


def plot_prediction(df, planet_letters, non_transiting_outer, t14s, window,
                    truncate=True, truncate_pad=1 / 24, fp=None):

    logger.info("Predict window = %s - %s", trunc(window[0].iso), trunc(window[1].iso))

    letters_ = planet_letters[:-1] if non_transiting_outer else planet_letters
    color_of = planet_colors(letters_)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8.5, 1.2))

        seen = set()
        for _, r in df.iterrows():
            planet = r['planet']
            t14 = t14s[planet]
            sd = r['tc_std']
            tc_bjd = r['tc_bjd']

            logger.info("\nPlanet %s", planet)
            logger.info("Predicted Tc = %.5f +/- %.5f BJD", tc_bjd, sd)
            logger.info("Predicted Tc = %s +/- %.0f min", trunc(Time(tc_bjd, format='jd').iso), sd * 1440)
            logger.info("Predicted ingress = %s", trunc(r['ingress_iso']))
            logger.info("Predicted egress = %s", trunc(r['egress_iso']))

            label = planet if planet not in seen else None
            seen.add(planet)
            t_ing = Time(tc_bjd - t14 / 2, format='jd')
            t_egr = Time(tc_bjd + t14 / 2, format='jd')
            ax.fill_between([t_ing.datetime, t_egr.datetime], y1=0, y2=1, alpha=0.3,
                            color=color_of[planet], lw=0, label=label)
            t_ing = Time(tc_bjd - t14 / 2 - sd, format='jd')
            t_egr = Time(tc_bjd + t14 / 2 + sd, format='jd')
            ax.fill_between([t_ing.datetime, t_egr.datetime], y1=0, y2=1, alpha=0.3,
                            color=color_of[planet], lw=0)

        for t in window:
            ax.axvline(t.datetime, ls=':', color='0.3')
        if seen:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
        if truncate:
            from astropy.time import TimeDelta
            xlim = [(t + TimeDelta(s * truncate_pad, format='jd')).datetime for t, s in zip(window, [-1, 1])]
            plt.setp(ax, xlim=xlim)
        ax.xaxis.set_major_formatter(matplotlib.dates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        plt.setp(ax.yaxis, visible=False)
        plt.setp(ax, ylim=(0, 1))
    return _finish(fig, fp)
