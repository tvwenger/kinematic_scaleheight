# kinematic_scaleheight
Demonstrate the error of the
[Crovisier (1978)](https://ui.adsabs.harvard.edu/abs/1978A%26A....70...43C/abstract)
method, and use various least squares and MCMC methods to kinematically estimate the vertical
distribution of clouds in the Galactic plane

# Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Simulating Data](#simulating-data)
  - [Least Squares](#least-squares)
    - [Crovisier Method](#crovisier-method)
    - [Corrected Least Squares](#corrected-least-squares)
  - [MCMC](#mcmc)
    - [Moment Ratio](#moment-ratio)
    - [Marginalized Distance](#marginalized-distance)
    - [Full Distance](#full-distance)
  - [Other Distributions](#other-distributions)
- [Caveats](#caveats)
- [Issues and Contributing](#issues-and-contributing)
- [License and Copyright](#license-and-copyright)

# Installation
```bash
conda create --name kinematic_scaleheight -c conda-forge pymc==5.8.2
conda activate kinematic_scaleheight
pip install --upgrade git+https://github.com/tvwenger/kinematic_scaleheight.git
pip install --upgrade git+https://github.com/tvwenger/pymc-experimental.git@chi_rv
```

# Usage

## Simulating Data
The `gen_synthetic_sample` function in `simulate.py` generates a synthetic sample of clouds
with which to test the subsequent methods.
Clouds are drawn from a uniform distribution in the Galactic plane and from a given vertical
distribution perpendicular to the Galactic plane. The Sun is assumed to be in the Galactic mid-plane.
The parameters that govern the Galactic rotation model are drawn from a multivariate normal distribution
fit to the posterior distribution of the
[Reid et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...885..131R/abstract)
`A5` model.

Available vertical distributions and their shape parameters include `gaussian` (standard deviation),
`exponential` (scale height), and `rectangular` (half-width).

```python
from kinematic_scaleheight.simulate import gen_synthetic_sample
glong, glat, vlsr, truths = gen_synthetic_sample(
    500, # sample size
    distribution='gaussian', # vertical distribution shape
    shape=100.0, # shape parameter for the distribution (pc)
    vlsr_err=5.0, # random noise added to observed LSR velocities (km/s)
    d_max=2000.0, # maximum distance of the clouds (pc)
    b_min=10.0, # minimum Galactic latitude (deg)
    b_max=90.0, # maximum Galactic latitude (deg)
    seed=1234, # random seed
)
```

The `truths` dictionary contains the "true" parameters that were used to generate the
simulated observations. These parameters include the true distance of each cloud,
the passed shape parameter, the first raw moment of the `|z|` distribution,
the ratio of the third to the second raw moments of the
`|z|` distribution, and the "true" values for the Galactic rotation model.

```python
print(truths.keys())
# dict_keys(['distance', 'distribution', 'shape', 'mom1_abs_z', 'mom3_mom2_abs_z_ratio', 'vlsr_err', 'R0', 'Usun', 'Vsun', 'Wsun', 'a2', 'a3'])
```

## Least Squares

### Crovisier Method
The `crovisier` function in `leastsq.py` uses the least squares method of Crovisier (1978) to
estimate the first raw moment of the vertical distribution of clouds. Given observations of
the positions and LSR velocities of the clouds, this method assigns each cloud to the expectation
value of the distance and then minimizes the squared difference between the observed LSR velocity
and the predicted LSR velocity at that distance, using a local approximation for the Galactic rotation
model (via Oort's A constant).

Note that, as demonstrated in Wenger et al. (in prep.), Crovisier (1978) incorrectly
derived the relationship between the first raw moment of the distance distribution
and the shape of the vertical distribution. Therefore, the Crovisier (1978) method
yields an incorrect result (the first raw moment of the vertical distribution)
when the sample is truncated in Galactic latitude.

```python
from kinematic_scaleheight.leastsq import crovisier
params, errors, vlsr_rms = crovisier(
    glong, # Galactic longitude of clouds (deg)
    glat, # Galactic latitude of clouds (deg)
    vlsr, # LSR velocities of clouds (km/s)
    oortA = 15.3, # Oort's A constant (km/s/kpc)
)
# params contains least-squares fit for (mom1_abs_z [pc], Usun [km/s], Vsun [km/s], Wsun [km/s])
print(params) # [155.70584276  -0.47117138  -0.70529303   0.6415374 ]
# errors contains the standard deviations
print(errors) # [1.06470679 0.06694245 0.0671935  0.1382609 ]
# vlsr_rms is the rms LSR velocity residual (km/s)
print(vlsr_rms) # 5.545071292001762

print(f"Expected: {truths['mom1_abs_z']}") # Expected: 79.78845608028654
print(f"Result: {params[0]}") # Result: 155.70584275602502
```

### Corrected Least Squares

The function `leastsq` corrects the Crovisier (1978) error by performing a
similar analysis and returning the actual measurable quantity: the
ratio between the third and second raw moments of the vertical distribution.

```python
from kinematic_scaleheight.leastsq import leastsq
params, errors, vlsr_rms = leastsq(
    glong, # Galactic longitude of clouds (deg)
    glat, # Galactic latitude of clouds (deg)
    vlsr, # LSR velocities of clouds (km/s)
    R0 = truths['R0'], # Galactocentric radius of the Sun (kpc)
    a2 = truths['a2'], # rotation curve parameter
    a3 = truths['a3'], # rotation curve parameter
)
# params contains least-squares fit for (mom3_mom2_abs_z_ratio [pc], Usun [km/s], Vsun [km/s], Wsun [km/s])
print(params) # array([160.1175338 ,  10.75369242,  15.7962179 ,   7.09344625])
# errors contains the standard deviations
print(errors) # array([1.08453171, 0.06694472, 0.06729067, 0.13826084])
# vlsr_rms is the rms LSR velocity residual (km/s)
print(vlsr_rms) # 5.492385979515315

print(f"Expected: {truths['mom3_mom2_abs_z_ratio']}") # Expected: 159.57691216057307
print(f"Result: {params[0]}") # Result: 160.11753379581967
```

## MCMC

There are a few different ways to approach this problem in a Bayesian way.

### Moment Ratio

First, we can simply infer the ratio of the third to second raw
moments of the `|z|` distribution, as in the least squares method. The
class `MomentRatioModel` in `mcmc.py` samples the *marginal* posterior distribution,
marginalized over the unknown distances of each cloud. As in the least squares
method, we assign each cloud to its expected distance for the inferred moment ratio.
The priors on the Galactic rotation model are taken from a multivariate normal
distribution fit to the posterior samples of the Reid et al. (2019) A5 model.
The prior for the moment ratio is a `k=2` gamma distribution with a user-supplied
mode.

```python
from kinematic_scaleheight.mcmc import MomentRatioModel
model = MomentRatioModel(
    glong, # Galactic longitudes of clouds (deg)
    glat, # Galactic latitudes of clouds (deg)
    vlsr, # LSR velocities of clouds (km/s)
    prior_mom3_mom2_abs_z_ratio=50.0, # mode of the moment ratio prior (pc)
    prior_vlsr_err=10.0, # standard deviation of the LSR velocity error prior (km/s)
)

# prior predictive checks
!mkdir example
prior, prior_predictive = model.plot_predictive(
    "prior", # prior predictive
    50, # prior predictive samples
    truths=truths, # optional truths dictionary
    plot_prefix="example/moment_ratio_", # plot filename prefix
)

# posterior sampling
model.sample(
    init="jitter+adapt_diag", # initialization strategy
    tune=500, # tuning samples
    draws=500, # posterior samples
    cores=4, # number of CPU cores to use
    chains=4, # number of MC chains to run
    target_accept=0.80, # target acceptance rate for sampling
)
#                           mean     sd   hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
# rotcurve[0]              8.168  0.011    8.147    8.189      0.000    0.000    2491.0    1561.0   1.00
# rotcurve[1]             10.594  0.269   10.087   11.087      0.006    0.004    2291.0    1670.0   1.00
# rotcurve[2]             15.714  0.373   14.964   16.359      0.008    0.006    2292.0    1549.0   1.00
# rotcurve[3]              7.670  0.232    7.224    8.083      0.005    0.004    2098.0    1595.0   1.00
# rotcurve[4]              0.959  0.013    0.936    0.986      0.000    0.000    2489.0    1649.0   1.00
# rotcurve[5]              1.610  0.003    1.605    1.616      0.000    0.000    1599.0    1558.0   1.00
# mom3_mom2_abs_z_ratio  161.332  6.002  150.503  172.537      0.123    0.087    2357.0    1546.0   1.00
# vlsr_err                 5.521  0.177    5.187    5.852      0.003    0.002    3283.0    1407.0   1.01

print(f"Expected: {truths['mom3_mom2_abs_z_ratio']}") # Expected: 159.57691216057307

# posterior predictive check
posterior, posterior_predictive = model.plot_predictive(
    "posterior", # posterior predictive
    50, # posterior predictive samples
    truths=truths, # optional truths dictionary
    plot_prefix="example/moment_ratio_", # plot filename prefix
)

# corner plot
model.plot_corner(
    truths=truths, # optional truths dictionary
    plot_prefix="example/moment_ratio_", # plot filename prefix
)
```

### Marginalized Distance

Alternatively, if we assume the shape of the `|z|` distribution, then we can
infer the shape parameter of this distribution directly. The `MarginalDistanceModel`
class in `mcmc.py` does just that, but still samples the *marginal* posterior distribution,
marginalized over the unknown distances of each cloud. The distributions and shape
parameters are the same as in `simulate.py`. The prior on the shape parameter is
a `k=2` gamma distribution with a user-supplied mode.

```python
from kinematic_scaleheight.mcmc import MarginalDistanceModel
model = MarginalDistanceModel(
    glong, # Galactic longitudes of clouds (deg)
    glat, # Galactic latitudes of clouds (deg)
    vlsr, # LSR velocities of clouds (km/s)
    distribution="gaussian", # assumed z distribution
    prior_shape=50.0, # mode of the shape prior (pc)
    prior_vlsr_err=10.0, # standard deviation of the LSR velocity error prior (km/s)
)

# prior predictive checks
!mkdir example
prior, prior_predictive = model.plot_predictive(
    "prior", # prior predictive
    50, # prior predictive samples
    truths=truths, # optional truths dictionary
    plot_prefix="example/marginal_distance_", # plot filename prefix
)

# posterior sampling
model.sample(
    init="jitter+adapt_diag", # initialization strategy
    tune=500, # tuning samples
    draws=500, # posterior samples
    cores=4, # number of CPU cores to use
    chains=4, # number of MC chains to run
    target_accept=0.80, # target acceptance rate for sampling
)
#                 mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
# rotcurve[0]    8.168  0.011   8.147    8.187      0.000    0.000    1946.0    1618.0    1.0
# rotcurve[1]   10.588  0.277  10.079   11.130      0.006    0.004    2464.0    1737.0    1.0
# rotcurve[2]   15.702  0.360  15.027   16.368      0.008    0.006    1849.0    1744.0    1.0
# rotcurve[3]    7.672  0.237   7.257    8.127      0.004    0.003    2867.0    1722.0    1.0
# rotcurve[4]    0.960  0.013   0.936    0.983      0.000    0.000    2268.0    1742.0    1.0
# rotcurve[5]    1.610  0.003   1.605    1.616      0.000    0.000    1758.0    1408.0    1.0
# shape        101.202  3.892  94.192  108.702      0.075    0.053    2708.0    1486.0    1.0
# vlsr_err       5.521  0.173   5.206    5.861      0.004    0.003    2417.0    1570.0    1.0

print(f"Expected: {truths['shape']}") # Expected: 100.0

# posterior predictive check
posterior, posterior_predictive = model.plot_predictive(
    "posterior", # posterior predictive
    50, # posterior predictive samples
    truths=truths, # optional truths dictionary
    plot_prefix="example/marginal_distance_", # plot filename prefix
)

# corner plot
model.plot_corner(
    truths=truths, # optional truths dictionary
    plot_prefix="example/marginal_distance_", # plot filename prefix
)
```

### Full Distance

Finally, we can again infer the shape parameter of the `|z|` distribution
directly, but here we sample the full posterior distribution, including the
distances to each cloud. We can therefore relax the assumption that the
clouds are not truncated in distance (see [Caveats](#caveats)).

```python
from kinematic_scaleheight.mcmc import FullDistanceModel
model = FullDistanceModel(
    glong, # Galactic longitudes of clouds (deg)
    glat, # Galactic latitudes of clouds (deg)
    vlsr, # LSR velocities of clouds (km/s)
    distribution="gaussian", # assumed z distribution
    prior_shape=50.0, # mode of the shape prior (pc)
    prior_vlsr_err=10.0, # standard deviation of the LSR velocity error prior (km/s)
    d_max=2000.0, # truncate distance distribution at this distance (pc)
)

# prior predictive checks
!mkdir example
prior, prior_predictive = model.plot_predictive(
    "prior", # prior predictive
    50, # prior predictive samples
    truths=truths, # optional truths dictionary
    plot_prefix="example/full_distance_", # plot filename prefix
)

# posterior sampling
# Note that we use init="adapt_diag" since jittering does not play nice with truncated
# distributions. Also note that sampling can, in general, be more challenging since
# the distances are not well constrained kinematically in the solar neighborhood.
# Hence, here we draw more samples.
model.sample(
    init="adapt_diag", # initialization strategy
    tune=1000, # tuning samples
    draws=1000, # posterior samples
    cores=4, # number of CPU cores to use
    chains=4, # number of MC chains to run
    target_accept=0.80, # target acceptance rate for sampling
)
#                 mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
# rotcurve[0]    8.168  0.011   8.147    8.187      0.000    0.000    2510.0    2264.0   1.00
# rotcurve[1]   10.569  0.272  10.071   11.079      0.006    0.004    2389.0    2250.0   1.00
# rotcurve[2]   15.821  0.364  15.095   16.462      0.009    0.006    1739.0    2111.0   1.00
# rotcurve[3]    7.662  0.233   7.236    8.110      0.004    0.003    3160.0    2645.0   1.00
# rotcurve[4]    0.959  0.014   0.933    0.984      0.000    0.000    2522.0    2015.0   1.00
# rotcurve[5]    1.610  0.003   1.604    1.616      0.000    0.000    1971.0    2206.0   1.00
# shape        102.914  4.368  94.847  111.016      0.293    0.208     221.0     679.0   1.01
# vlsr_err       4.828  0.185   4.484    5.166      0.005    0.003    1514.0    2346.0   1.00

print(f"Expected: {truths['shape']}") # Expected: 100.0

# We can inspect the posterior statistics for the distances, too
import arviz as az
summary = az.summary(model.trace)
print(summary[7:13])
#                  mean       sd   hdi_3%   hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
# distance[0]   757.101  330.021  153.388  1354.544     10.428    8.865    1278.0     663.0    1.0
# distance[1]   312.937  132.326   66.282   545.195      2.844    2.242    2430.0    1877.0    1.0
# distance[2]   819.522  336.496  236.501  1461.699      8.884    7.135    1598.0    1012.0    1.0
# distance[3]   523.986  226.338  114.376   914.196      6.828    5.842    1416.0     803.0    1.0
# distance[4]   703.863  220.044  294.453  1117.108      4.751    3.742    2270.0    1409.0    1.0
# distance[5]  1071.387  384.358  334.118  1750.720     10.962    9.175    1229.0     743.0    1.0

# posterior predictive check
posterior, posterior_predictive = model.plot_predictive(
    "posterior", # posterior predictive
    50, # posterior predictive samples
    truths=truths, # optional truths dictionary
    plot_prefix="example/full_distance_", # plot filename prefix
)

# corner plot
model.plot_corner(
    truths=truths, # optional truths dictionary
    plot_prefix="example/full_distance_", # plot filename prefix
)
```

## Other Distributions

In the preceeding examples, we simulated a "gaussian" distribution. Here we demonstrate
the results for the other supported distributions.

### Exponential

Note that an "exponential" distribution has a long tail, and thus the assumption that
`d_max sin(b_min) >> mom1_abs_z` becomes less accurate for small `d_max` or small `b_min` 
(see [Caveats](#caveats)). The
least squares and marginalized distance methods will yield incorrect results in this case.
Here we've increased `d_max` to ensure the approximation is more valid, but still the
least squares and marginalized distance methods suffer from the bias.

```python
# Generate data
from kinematic_scaleheight.simulate import gen_synthetic_sample
glong, glat, vlsr, truths = gen_synthetic_sample(
    500, # sample size
    distribution='exponential', # vertical distribution shape
    shape=100.0, # shape parameter for the distribution (pc)
    vlsr_err=5.0, # random noise added to observed LSR velocities (km/s)
    d_max=5000.0, # maximum distance of the clouds (pc)
    b_min=10.0, # minimum Galactic latitude (deg)
    b_max=90.0, # maximum Galactic latitude (deg)
    seed=1234, # random seed
)

# Corrected least-squares
from kinematic_scaleheight.leastsq import leastsq
params, errors, vlsr_rms = leastsq(
    glong, # Galactic longitude of clouds (deg)
    glat, # Galactic latitude of clouds (deg)
    vlsr, # LSR velocities of clouds (km/s)
    R0 = truths['R0'], # Galactocentric radius of the Sun (kpc)
    a2 = truths['a2'], # rotation curve parameter
    a3 = truths['a3'], # rotation curve parameter
)
# mom3_mom2_abs_z_ratio (pc)
print(f"Expected: {truths['mom3_mom2_abs_z_ratio']}") # Expected: 300.0
print(f"Result: {params[0]}") # Result: Result: 284.2074519228318
# N.B. Notice the difference due to the poor assumption

# Moment ratio MCMC
from kinematic_scaleheight.mcmc import MomentRatioModel
model = MomentRatioModel(
    glong, # Galactic longitudes of clouds (deg)
    glat, # Galactic latitudes of clouds (deg)
    vlsr, # LSR velocities of clouds (km/s)
    prior_mom3_mom2_abs_z_ratio=100.0, # mode of the moment ratio prior (pc)
    prior_vlsr_err=10.0, # standard deviation of the LSR velocity error prior (km/s)
)
model.sample(
    init="jitter+adapt_diag", # initialization strategy
    tune=500, # tuning samples
    draws=500, # posterior samples
    cores=4, # number of CPU cores to use
    chains=4, # number of MC chains to run
    target_accept=0.80, # target acceptance rate for sampling
)
#                           mean     sd   hdi_3%  hdi_97%  mcse_mean  
# mom3_mom2_abs_z_ratio  288.521  8.682  272.043  304.364      0.159    0.113    2995.0    1476.0    1.0
# vlsr_err                 7.824  0.237    7.358    8.242      0.005    0.003    2463.0    1739.0    1.0
# N.B. Better, but still biased

# Marginalized distance MCMC
from kinematic_scaleheight.mcmc import MarginalDistanceModel
model = MarginalDistanceModel(
    glong, # Galactic longitudes of clouds (deg)
    glat, # Galactic latitudes of clouds (deg)
    vlsr, # LSR velocities of clouds (km/s)
    distribution="exponential", # assumed z distribution
    prior_shape=100.0, # mode of the shape prior (pc)
    prior_vlsr_err=10.0, # standard deviation of the LSR velocity error prior (km/s)
)
model.sample(
    init="jitter+adapt_diag", # initialization strategy
    tune=500, # tuning samples
    draws=500, # posterior samples
    cores=4, # number of CPU cores to use
    chains=4, # number of MC chains to run
    target_accept=0.80, # target acceptance rate for sampling
)
#                mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
# shape        96.327  2.822  91.005  101.463      0.052    0.037    2959.0    1614.0    1.0
# vlsr_err      7.826  0.240   7.393    8.303      0.004    0.003    2905.0    1404.0    1.0
# N.B. Again, still biased

# Full distance MCMC
from kinematic_scaleheight.mcmc import FullDistanceModel
model = FullDistanceModel(
    glong, # Galactic longitudes of clouds (deg)
    glat, # Galactic latitudes of clouds (deg)
    vlsr, # LSR velocities of clouds (km/s)
    distribution="exponential", # assumed z distribution
    prior_shape=50.0, # mode of the shape prior (pc)
    prior_vlsr_err=10.0, # standard deviation of the LSR velocity error prior (km/s)
    d_max=5000.0, # truncate distance distribution at this distance (pc)
)
# N.B. Need more samples for convergence
model.sample(
    init="adapt_diag", # initialization strategy
    tune=5000, # tuning samples
    draws=5000, # posterior samples
    cores=8, # number of CPU cores to use
    chains=8, # number of MC chains to run
    target_accept=0.80, # target acceptance rate for sampling
)
#                mean      sd   hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
# shape        100.568  3.814  93.310  107.904      0.280    0.198     185.0     358.0   1.05
# vlsr_err       5.118  0.266   4.609    5.607      0.004    0.003    4231.0   12620.0   1.00
```

Notice that the `r_hat` statistic is still large for `shape`. We need
even more posterior samples to ensure convergence.

### Rectangular

```python
# Generate data
from kinematic_scaleheight.simulate import gen_synthetic_sample
glong, glat, vlsr, truths = gen_synthetic_sample(
    1000, # sample size
    distribution='rectangular', # vertical distribution shape
    shape=100.0, # shape parameter for the distribution (pc)
    vlsr_err=5.0, # random noise added to observed LSR velocities (km/s)
    d_max=2000.0, # maximum distance of the clouds (pc)
    b_min=10.0, # minimum Galactic latitude (deg)
    b_max=90.0, # maximum Galactic latitude (deg)
    seed=1234, # random seed
)

# Corrected least-squares
from kinematic_scaleheight.leastsq import leastsq
params, errors, vlsr_rms = leastsq(
    glong, # Galactic longitude of clouds (deg)
    glat, # Galactic latitude of clouds (deg)
    vlsr, # LSR velocities of clouds (km/s)
    R0 = truths['R0'], # Galactocentric radius of the Sun (kpc)
    a2 = truths['a2'], # rotation curve parameter
    a3 = truths['a3'], # rotation curve parameter
)
# mom3_mom2_abs_z_ratio (pc)
print(f"Expected: {truths['mom3_mom2_abs_z_ratio']}") # Expected: 75.0
print(f"Result: {params[0]}") # Result: 72.94068819947938

# Moment ratio MCMC
from kinematic_scaleheight.mcmc import MomentRatioModel
model = MomentRatioModel(
    glong, # Galactic longitudes of clouds (deg)
    glat, # Galactic latitudes of clouds (deg)
    vlsr, # LSR velocities of clouds (km/s)
    prior_mom3_mom2_abs_z_ratio=300.0, # mode of the moment ratio prior (pc)
    prior_vlsr_err=10.0, # standard deviation of the LSR velocity error prior (km/s)
)
model.sample(
    init="jitter+adapt_diag", # initialization strategy
    tune=500, # tuning samples
    draws=500, # posterior samples
    cores=4, # number of CPU cores to use
    chains=4, # number of MC chains to run
    target_accept=0.80, # target acceptance rate for sampling
)
#                          mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
# mom3_mom2_abs_z_ratio  74.053  4.062  66.495   81.834      0.081    0.057    2521.0    1709.0    1.0
# vlsr_err                4.953  0.110   4.725    5.143      0.002    0.002    2637.0    1503.0    1.0

# Marginal distance MCMC
from kinematic_scaleheight.mcmc import MarginalDistanceModel
model = MarginalDistanceModel(
    glong, # Galactic longitudes of clouds (deg)
    glat, # Galactic latitudes of clouds (deg)
    vlsr, # LSR velocities of clouds (km/s)
    distribution="rectangular", # assumed z distribution
    prior_shape=300.0, # mode of the shape prior (pc)
    prior_vlsr_err=10.0, # standard deviation of the LSR velocity error prior (km/s)
)
model.sample(
    init="jitter+adapt_diag", # initialization strategy
    tune=500, # tuning samples
    draws=500, # posterior samples
    cores=4, # number of CPU cores to use
    chains=4, # number of MC chains to run
    target_accept=0.80, # target acceptance rate for sampling
)
#                mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
# shape        98.583  5.232  89.052  109.152      0.106    0.075    2425.0    1531.0    1.0
# vlsr_err      4.959  0.113   4.760    5.189      0.002    0.002    2521.0    1473.0    1.0

# Full distance MCMC
from kinematic_scaleheight.mcmc import FullDistanceModel
model = FullDistanceModel(
    glong, # Galactic longitudes of clouds (deg)
    glat, # Galactic latitudes of clouds (deg)
    vlsr, # LSR velocities of clouds (km/s)
    distribution="rectangular", # assumed z distribution
    prior_shape=50.0, # mode of the shape prior (pc)
    prior_vlsr_err=10.0, # standard deviation of the LSR velocity error prior (km/s)
    d_max=2000.0, # truncate distance distribution at this distance (pc)
)
model.sample(
    init="adapt_diag", # initialization strategy
    tune=1000, # tuning samples
    draws=1000, # posterior samples
    cores=4, # number of CPU cores to use
    chains=4, # number of MC chains to run
    target_accept=0.80, # target acceptance rate for sampling
)
#                mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
# shape        98.320  5.288  88.672  108.373      0.068    0.048    6073.0    2649.0    1.0
# vlsr_err      4.890  0.112   4.683    5.101      0.001    0.001    7093.0    2859.0    1.0
```

# Caveats

Except for the full distance MCMC method, the other methods inherently assume that
`d_max sin(b_min) >> mom1_abs_z`. That is, they assume that the effect of a
truncated distance distribution is hardly noticible at the lowest Galactic latitude.
For low-latitude clouds or large scale heights of the vertical distribution, this
assumption is no longer true!

The full distance MCMC method properly handles a truncated distance distribution,
but sampling the posterior distribution may still be challenging as `d_max sin(b_min)`
approaches `mom1_abs_z`.

# Issues and Contributing

Anyone is welcome to submit issues or contribute to the development
of this software via [Github](https://github.com/tvwenger/kinematic_scaleheight).

# License and Copyright

Copyright (c) 2023 Trey Wenger

GNU General Public License v3 (GNU GPLv3)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
