# The TTV model

## Linear ephemeris

For each transiting planet $p$, harmonic first fits a linear ephemeris to its observed transit times,

$$
t_{\mathrm{lin}}(p, E) = T_{0,p} + P_p\,E,
$$

where $E$ is the integer transit epoch, $P_p$ the mean orbital period, and $T_{0,p}$ the reference transit time. The transit-timing variation is the departure of the observed time from this line.

## Harmonic model

Near a first-order mean-motion resonance, the dominant TTV of each planet is a sinusoid at the pair's **super-period** $P^{\mathrm{TTV}}_{ij}$ (Lithwick, Xie & Wu 2012). harmonic models the transit time of planet $p$ as its linear ephemeris plus one sinusoid per adjacent pair it belongs to:

$$
t_c(p, E) = t_{\mathrm{lin}}(p, E) + \sum_{\text{pairs } (i,j)\ni p} a_{p}\,\sin\!\left(\frac{2\pi\,(t_{\mathrm{lin}} - t^{\mathrm{TTV}}_{ij})}{P^{\mathrm{TTV}}_{ij}}\right).
$$

An interior planet contributes one term (its outer neighbour); an exterior planet one term (its inner neighbour); a middle planet contributes both.

## Sampled parametrization

Rather than sample amplitude, super-period, and phase $(a, P^{\mathrm{TTV}}, t^{\mathrm{TTV}})$ directly, harmonic samples a **sine/cosine** decomposition that removes the phase degeneracy. Writing the sinusoid argument as $\theta = 2\pi\,(t_{\mathrm{lin}} - t_{\mathrm{ref}})/P^{\mathrm{TTV}}_{ij}$ against a fixed data-frame pivot $t_{\mathrm{ref}}$, the pair $(i,j)$ is parametrized by

- $a^{\sin}_{ij},\ a^{\cos}_{ij}$ — the inner planet's sine and cosine amplitudes,
- $r_{ji}$ — the signed outer/inner amplitude ratio (its sign carries the anti-phase),
- $P^{\mathrm{TTV}}_{ij}$ — the super-period.

The inner planet's term is $a^{\sin}_{ij}\sin\theta + a^{\cos}_{ij}\cos\theta$ and the outer planet's is $r_{ji}$ times the same. The `[INIT]` config keys $(a, P^{\mathrm{TTV}}, t^{\mathrm{TTV}})$ are converted to this form at load time; amplitude and phase are recovered as derived quantities and reported in the fit summary.

### Shared phase vs. phase offsets

The default (shared-phase) mode ties the two planets of a pair to a single TTV phase and takes their sinusoids to be exactly anti-correlated (relative phase 180°, encoded by the sign of $r_{ji}$). This is the leading-order near-resonant behaviour (Lithwick, Xie & Wu 2012); in general, free eccentricity can shift a planet's TTV phase, so the true relative phase of a pair may depart from 180°.

When the shared-phase model cannot represent such a pair, the fit compensates by collapsing the inner planet's amplitude toward zero and inflating $r_{ji}$, so the $r_{ji}$ posterior piles up against its prior bound (harmonic warns when it detects this). In that case re-fit with `--phase-offsets`, which gives each planet its own independent sine/cosine amplitudes.

## Priors and fitting

Bounds are auto-scaled from the data per system: $T_0$ within $\pm 0.5$ d of the linear-ephemeris value, $P$ within $\pm 1\%$, amplitudes within $\pm 10\times$ the largest observed $|O\!-\!C|$ of the pair, and the super-period log-uniform between five times the outer planet's orbital period and twenty times the data baseline.

Fitting proceeds in two stages: a bounded `scipy.optimize.least_squares` (TRF) fit with an analytic Jacobian and a phase/sign multi-start finds the optimum, then `emcee` samples the posterior, initialized in a small ball around it. Everything is seeded (`--seed`, default 42) for reproducibility.

## Mass constraints

From the posterior, harmonic derives planet masses following Lithwick, Xie & Wu (2012). The complex TTV amplitudes of the inner and outer planet of a near-resonant pair are

$$
V \propto -f - \frac{3}{2}\frac{Z^\ast_{\mathrm{free}}}{\Delta}, \qquad
V' \propto -g + \frac{3}{2}\frac{Z^\ast_{\mathrm{free}}}{\Delta},
$$

where $\Delta$ measures the fractional distance from exact commensurability, $Z_{\mathrm{free}}$ is a free-eccentricity combination, and $f, g$ are order-unity coefficients tabulated per resonance (with $f<0$, $g>0$). harmonic identifies the resonance $j\!:\!(j\!-\!1)$ nearest each pair's period ratio, then uses the measured amplitudes to constrain the perturbing masses by approximate Bayesian sampling over the mass ratio and free eccentricity. The result is reported in Earth masses, scaled by the host mass (`--mstar`).
