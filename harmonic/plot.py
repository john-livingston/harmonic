import numpy as np
import matplotlib.pyplot as plt
import corner

from .model import model

PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#F0E442', '#000000']  # Okabe-Ito
STYLE = {
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'axes.labelsize': 11,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.grid': True, 'grid.alpha': 0.15, 'grid.linewidth': 0.6,
    'legend.frameon': False, 'font.size': 10,
}
SAVE_KW = dict(dpi=200, bbox_inches='tight')


def planet_colors(letters):
    return {l: PALETTE[i % len(PALETTE)] for i, l in enumerate(letters)}


def _finish(fig, fp):
    if fp is not None:
        fig.savefig(fp, **SAVE_KW)
        plt.close(fig)
    return fig


def plot_trace(chain, labels, fp=None):

    nsteps, nwalkers, ndim = chain.shape

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(ndim, 1, figsize=(10, max(ndim, 2)), sharex=True, squeeze=False)
        axes = axes.ravel()
        for i in range(ndim):
            ax = axes[i]
            ax.plot(chain[:, :, i], color='0.2', alpha=0.15, lw=0.5)
            ax.set_xlim(0, nsteps)
            ax.set_ylabel(labels[i], fontsize=11)
            ax.yaxis.set_label_coords(-0.1, 0.5)

        fig.subplots_adjust(hspace=0)
    return _finish(fig, fp)


def plot_corner(fc, labels=None, fp=None):

    hist_kwargs = dict(lw=1.2)
    title_kwargs = dict(fontdict=dict(fontsize=12))
    quantiles = 0.16, 0.5, 0.84

    labels = labels or list(fc.columns)

    with plt.rc_context(STYLE):
        fig = corner.corner(fc,
                      labels=labels,
                      quantiles=quantiles,
                      hist_kwargs=hist_kwargs,
                      plot_datapoints=False,
                      smooth=1, smooth1d=1,
                      show_titles=True,
                      title_fmt='.4f')
    return _finish(fig, fp)


def plot_bestfit(ttv, times, tci, planeti, epochi, nplanets, planet_letters, non_transiting_outer, fp=None):
    if non_transiting_outer:
        npmax = nplanets-1
    else:
        npmax = nplanets

    colors = planet_colors(planet_letters)

    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(nrows=npmax, figsize=(10, 10), sharex=True, squeeze=False)
        axs = axs.ravel()
        for j, splanet in enumerate(planet_letters):
            idx = np.where(planeti == splanet)[0]
            if non_transiting_outer and j == nplanets-1:
                break
            ttv.plot_times(splanet, ax=axs[j], color='0.15', elinewidth=1, capsize=0, ms=4)
            ttv.plot_model(splanet, tci[idx], epochi[idx], ax=axs[j],
                           color=colors[splanet], mew=0, alpha=0.9, lw=1.2)
            axs[j].set_ylabel(f'TTV {splanet} (min)')
    return _finish(fig, fp)


def plot_samples(ttv, times, ephem, flatchain, planeti, nplanets, planet_letters, non_transiting_outer, phase_offsets=False, tmax=None, fp=None, t_ref=0.0):

    if tmax is None:
        tmax = times.tc.max()

    colors = planet_colors(planet_letters)

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(10, 16))
        timeaxL = []
        resaxL = []
        if non_transiting_outer:
            npmax = nplanets - 1
        else:
            npmax = nplanets
        gs = plt.GridSpec(npmax*2, 1, width_ratios=[1], height_ratios=[2, 1]*npmax)
        i = 0
        timeaxL.append(fig.add_subplot(gs[2*i]))
        resaxL.append(fig.add_subplot(gs[2*i+1], sharex=timeaxL[0]))
        for i in range(1, npmax):
            timeaxL.append(fig.add_subplot(gs[2*i], sharex=timeaxL[0]))
            resaxL.append(fig.add_subplot(gs[2*i+1], sharex=timeaxL[0]))

        fig.tight_layout()

        i = 0
        nsamp = min(100, len(flatchain))
        for _, sample in flatchain.sample(nsamp, random_state=0).T.items():
            pas = sample.to_dict()
            for j, planet in enumerate(planet_letters):
                if not non_transiting_outer or j < nplanets-1:
                    ax = timeaxL[j]
                    if i == 0:
                        ttv.plot_times(planet, ax=ax, lw=2, color='0.15', zorder=10,
                                       elinewidth=1, capsize=0, ms=4)

                    ntrans = int((tmax - times[times.planet==planet].tc.min()) / ephem.loc[planet,'per'])
                    epochi = np.arange(0, ntrans)

                    planeti_arr = np.array([planet]*ntrans)
                    tc = model(pas, planeti_arr, epochi, nplanets, planet_letters, non_transiting_outer, phase_offsets, t_ref=t_ref)
                    ttv.plot_model(planet, tc, epochi, ax=ax, color=colors[planet], mew=0, alpha=0.08)
                    ax.set_ylabel(f'TTV {planet} (min)')

                    ax = resaxL[j]
                    resid = times.tc - model(pas, times.planet, times.epoch, nplanets, planet_letters, non_transiting_outer, phase_offsets, t_ref=t_ref)
                    idx = np.where(times.planet==planet)[0]
                    ax.errorbar(times.tc[idx], resid[idx]*24*60, times.tc_unc[idx]*1440,
                                color=colors[planet], marker='.', ls='', mew=0, alpha=0.4)
                    ax.set_ylabel('residuals (min)')
                    ax.axhline(0, color='0.5', lw=0.6, ls='--', alpha=0.5)
            i += 1

        fig.subplots_adjust(hspace=0)
    return _finish(fig, fp)
