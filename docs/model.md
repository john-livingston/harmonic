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

## Detecting TTVs

To quantify whether the TTVs are detected at all, as a single number for the whole system rather than per planet, harmonic compares the best-fit harmonic model against the linear-ephemeris null (each planet strictly periodic, i.e. all TTV amplitudes zero) with the Bayesian Information Criterion:

$$
\Delta\mathrm{BIC} = \left(\chi^2_{\mathrm{lin}} - \chi^2_{\mathrm{harm}}\right) - \left(k_{\mathrm{harm}} - k_{\mathrm{lin}}\right)\ln N,
$$

where each $\chi^2$ is evaluated at that model's maximum-likelihood fit, $k$ is its number of free parameters, and $N$ is the total number of transit times. Because the linear model is nested inside the harmonic one, the Gaussian log-likelihood constant cancels. A positive $\Delta\mathrm{BIC}$ favors the harmonic model, i.e. the TTVs are detected, and larger is stronger (roughly $>2$ positive, $>6$ strong, $>10$ very strong). The value and its ingredients are written to `fit_stats.json` and printed at the end of a fit.

## Ranking future transits

A prediction run also ranks each upcoming transit by its expected information gain: the reduction in posterior entropy from observing it at an assumed timing precision $\sigma$ (each planet's median measured uncertainty by default, or a global `--sigma`). Under a Gaussian approximation of the posterior with covariance $C$ (estimated from the chain), an observation with model gradient $j$ (computed analytically at the candidate transit) is a rank-one Fisher information update, and the total gain in bits is

$$
\Delta I_{\mathrm{tot}} = \tfrac{1}{2}\log_2\!\left(1 + \frac{j^{\mathsf T} C\, j}{\sigma^2}\right),
$$

where $j^{\mathsf T} C j$ is the variance of the predicted transit time: the most informative transit is the one whose predicted time is most uncertain relative to the precision it can be measured with. A second, targeted gain counts only the information reaching the TTV parameters (amplitudes, ratios, super-periods), discounting what merely re-pins the linear ephemerides:

$$
\Delta I_{\mathrm{TTV}} = \Delta I_{\mathrm{tot}} - \tfrac{1}{2}\log_2\!\left(1 + \frac{j_N^{\mathsf T} F_{NN}^{-1} j_N}{\sigma^2}\right),
$$

where $N$ is the ephemeris block $(T_0, P)$ of the information matrix $F = C^{-1}$. Because consecutive transits carry nearly duplicate information, harmonic also reports a greedy observing order: the best transit is selected, the covariance is updated as if it had been observed, and the remaining candidates are re-scored (`greedy_rank`, with `greedy_gain` the gain at selection time). The criterion driving the order is chosen with `--rank-by` (`total`, the default, or `ttv`).

## Mass constraints

From the posterior, harmonic derives planet masses following Lithwick, Xie & Wu (2012). The complex TTV amplitudes of the inner and outer planet of a near-resonant pair are

$$
V \propto -f - \frac{3}{2}\frac{Z^\ast_{\mathrm{free}}}{\Delta}, \qquad
V' \propto -g + \frac{3}{2}\frac{Z^\ast_{\mathrm{free}}}{\Delta},
$$

where $\Delta$ measures the fractional distance from exact commensurability, $Z_{\mathrm{free}}$ is a free-eccentricity combination, and $f, g$ are order-unity coefficients tabulated per resonance (with $f<0$, $g>0$). harmonic identifies the resonance $j\!:\!(j\!-\!1)$ nearest each pair's period ratio, then uses the measured amplitudes to constrain the perturbing masses by approximate Bayesian sampling over the mass ratio and free eccentricity. The result is reported in Earth masses, scaled by the host mass (`--mstar`).
