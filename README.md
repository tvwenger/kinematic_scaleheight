# kinematic_scaleheight

[![DOI](https://zenodo.org/badge/656757083.svg)](https://zenodo.org/doi/10.5281/zenodo.10818723)

Infer the vertical distribution of HI absorption detections in the
solar neighborhood, using the least squares analysis of [Crovisier
(1978)](https://ui.adsabs.harvard.edu/abs/1978A%26A....70...43C/abstract)
as well as a Bayesian model.

See [Wenger et al. (2024)]() for more details.

# Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Simulating Data](#simulating-data)
  - [Least Squares](#least-squares)
    - [Crovisier Method](#crovisier-method)
    - [Corrected Least Squares](#corrected-least-squares)
  - [MCMC](#mcmc)
    - [Moment Ratio](#moment-ratio)
    - [Shape](#shape)
  - [Other Distributions](#other-distributions)
  - [Model Comparison](#model-comparison)
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
`A5` model. Additionally, interloping ("outlier") clouds are drawn from a broader velocity
distribution.

Available vertical distributions and their shape parameters include `gaussian` (standard deviation),
`exponential` (scale height), and `rectangular` (half-width).

```python
from kinematic_scaleheight.simulate import gen_synthetic_sample
glong, glat, vlsr, truths = gen_synthetic_sample(
    300, # sample size
    distribution='gaussian', # vertical distribution shape
    shape=100.0, # shape parameter for the distribution (pc)
    vlsr_err=5.0, # random noise added to observed LSR velocities (km/s)
    b_min=10.0, # minimum Galactic latitude (deg)
    b_max=90.0, # maximum Galactic latitude (deg)
    outlier_vlsr_sigma=30.0, # width of LSR velocity distribution for outliers (km/s)
    outlier_frac=0.1, # fraction of sample that are outliers
    seed=1234, # random seed
    verbose=True, # print helpful information
)
"""
Simulating 300 clouds up to d_max = 5758.770 pc
Added 22 clouds (22/300) in iteration 0
Added 19 clouds (41/300) in iteration 1
Added 23 clouds (64/300) in iteration 2
Added 24 clouds (88/300) in iteration 3
Added 24 clouds (112/300) in iteration 4
Added 23 clouds (135/300) in iteration 5
Added 23 clouds (158/300) in iteration 6
Added 17 clouds (175/300) in iteration 7
Added 28 clouds (203/300) in iteration 8
Added 23 clouds (226/300) in iteration 9
Added 26 clouds (252/300) in iteration 10
Added 29 clouds (281/300) in iteration 11
Added 20 clouds (301/300) in iteration 12
Simulation complete. Trimming sample to 300 clouds
"""
```

The `truths` dictionary contains the "true" parameters that were used to generate the
simulated observations. These parameters include the "true" distance of each cloud,
a boolean flag to indicate if a data point is from the outlier population,
the passed distribution, the passed shape parameter, the maximum distance allowed
for the simulated clouds, the first raw moment of the `|z|` distribution,
the ratio of the third to the second raw moments of the
`|z|` distribution, the true LSR velocity error, the true outlier fraction and velocity
distribution width, and the "true" values for the Galactic rotation model parameters.

```python
print(truths.keys())
# dict_keys(['distance', 'outlier', 'distribution', 'shape', 'd_max', 'mom1_abs_z', 'mom3_mom2_abs_z_ratio', 'vlsr_err', 'outlier_frac', 'outlier_vlsr_sigma', 'R0', 'Usun', 'Vsun', 'Wsun', 'a2', 'a3'])
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

Here we drop the known outliers to test the methods.

```python
from kinematic_scaleheight.leastsq import crovisier
params, errors, vlsr_rms = crovisier(
    glong[~truths['outlier']], # Galactic longitude of clouds (deg)
    glat[~truths['outlier']], # Galactic latitude of clouds (deg)
    vlsr[~truths['outlier']], # LSR velocities of clouds (km/s)
    oortA = 15.3, # Oort's A constant (km/s/kpc)
)
# params contains least-squares fit for
# (mom1_abs_z [pc], Usun [km/s], Vsun [km/s], Wsun [km/s], nodal_deviation [deg])
print(params) # [169.01403745   0.51634831  -0.18137725  -1.05070969   1.53262918]
# errors contains the parameter standard deviation estimates
print(errors) # [7.62922783 0.50414248 0.52096358 1.06562422 1.43799352]
# vlsr_rms is the rms LSR velocity residual (km/s)
print(vlsr_rms) # 5.605146497847245

print(f"Expected: {truths['mom1_abs_z']}") # Expected: 79.78845608028654
print(f"Result: {params[0]}") # Result: 169.0140374479736
```

### Corrected Least Squares

The function `leastsq` corrects the Crovisier (1978) error by performing a
similar analysis and returning the actual measurable quantity: the
ratio between the third and second raw moments of the vertical distribution.
This method also uses an updated, non-local Galactic rotation model.

Again, we drop the known outliers to demonstrate the accuracy of the method.

```python
from kinematic_scaleheight.leastsq import leastsq
params, errors, vlsr_rms = leastsq(
    glong[~truths['outlier']], # Galactic longitude of clouds (deg)
    glat[~truths['outlier']], # Galactic latitude of clouds (deg)
    vlsr[~truths['outlier']], # LSR velocities of clouds (km/s)
    R0 = truths['R0'], # Galactocentric radius of the Sun (kpc)
    a2 = truths['a2'], # rotation curve parameter
    a3 = truths['a3'], # rotation curve parameter
)
# params contains least-squares fit for (mom3_mom2_abs_z_ratio [pc], Usun [km/s], Vsun [km/s], Wsun [km/s])
print(params) # [169.72950204   9.75167973  15.31139737   8.74598833]
# errors contains the standard deviation estimates
print(errors) # [7.6596202  0.50514601 0.52233199 1.06721765]
# vlsr_rms is the rms LSR velocity residual (km/s)
print(vlsr_rms) # 5.6165309797944065

print(f"Expected: {truths['mom3_mom2_abs_z_ratio']}") # Expected: 159.57691216057307
print(f"Result: {params[0]}") # Result: 169.72950203558383
```

## MCMC

There are two Bayesian ways to approach this problem.

### Moment Ratio

First, we can simply infer the ratio of the third to second raw
moments of the `|z|` distribution, as in the least squares method. The
class `MomentRatioModel` in `mcmc.py` samples the *marginal* posterior distribution,
marginalized over the latent distances of each cloud.
The priors on the Galactic rotation model are taken from a multivariate normal
distribution fit to the posterior samples of the Reid et al. (2019) A5 model.
The prior for the moment ratio is a `k=2` gamma distribution with a user-supplied
mode.

```python
from kinematic_scaleheight.mcmc import MomentRatioModel
gaussian_momratio_model = MomentRatioModel(
    glong, # Galactic longitudes of clouds (deg)
    glat, # Galactic latitudes of clouds (deg)
    vlsr, # LSR velocities of clouds (km/s)
    prior_mom3_mom2_abs_z_ratio=50.0, # mode of the moment ratio prior (pc)
    prior_vlsr_err=10.0, # standard deviation of the LSR velocity error prior (km/s)
    prior_outlier_vlsr_sigma=50.0, # standard deviation of the LSR velocity outlier prior (km/s)
)

# prior predictive checks
!mkdir example
gaussian_momratio_vlsr_prior = gaussian_momratio_model.vlsr_predictive(
    "prior", # prior predictive
    50, # prior predictive samples
    fname="example/gaussian_moment_ratio_vlsr_prior.pdf", # plot filename
    seed=1234, # random seed
)

# posterior sampling
gaussian_momratio_model.sample(
    init="jitter+adapt_diag", # initialization strategy
    tune=500, # tuning samples
    draws=500, # posterior samples
    cores=4, # number of CPU cores to use
    chains=4, # number of MC chains to run
    target_accept=0.80, # target acceptance rate for sampling
    seed=1234, # random seed
)
"""
Auto-assigning NUTS sampler...
Initializing NUTS using jitter+adapt_diag...
Multiprocess sampling (4 chains in 4 jobs)
NUTS: [rotcurve, outlier_vlsr_sigma, mom3_mom2_abs_z_ratio, vlsr_err, w]
Sampling 4 chains for 500 tune and 500 draw iterations (2_000 + 2_000 draws total) took 8 seconds.chains, 0 divergences]
                          mean     sd   hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
rotcurve[0]              8.166  0.011    8.147    8.186      0.000    0.000    2641.0    1331.0    1.0
rotcurve[1]             10.216  0.315    9.642   10.819      0.006    0.004    3047.0    1532.0    1.0
rotcurve[2]             15.234  0.551   14.183   16.272      0.011    0.008    2664.0    1514.0    1.0
rotcurve[3]              7.769  0.244    7.330    8.247      0.004    0.003    3705.0    1478.0    1.0
rotcurve[4]              0.964  0.013    0.938    0.986      0.000    0.000    3385.0    1913.0    1.0
rotcurve[5]              1.612  0.003    1.606    1.618      0.000    0.000    2577.0    1482.0    1.0
mom3_mom2_abs_z_ratio  163.506  8.934  147.181  180.331      0.155    0.110    3285.0    1687.0    1.0
vlsr_err                 5.495  0.346    4.879    6.187      0.006    0.005    2949.0    1749.0    1.0
w[0]                     0.863  0.035    0.803    0.929      0.001    0.000    2627.0    1429.0    1.0
w[1]                     0.137  0.035    0.071    0.197      0.001    0.000    2627.0    1429.0    1.0
outlier_vlsr_sigma      21.855  3.269   16.214   27.812      0.067    0.050    2664.0    1598.0    1.0
"""
# Note:
# rotcurve[0] = R0 (kpc)
# rotcurve[1] = Usun (km/s)
# rotcurve[2] = Vsun (km/s)
# rotcurve[3] = Wsun (km/s)
# rotcurve[4] = a2
# rotcurve[5] = a3
# w[0] = non-outlier fraction
# w[1] = outlier fraction

print(f"Expected: {truths['mom3_mom2_abs_z_ratio']}") # Expected: 159.57691216057307

# posterior predictive check
gaussian_momratio_vlsr_posterior = gaussian_momratio_model.vlsr_predictive(
    "posterior", # posterior predictive
    50, # posterior predictive samples
    fname="example/gaussian_moment_ratio_vlsr_posterior.pdf", # plot filename
    seed=1234, # random seed
)

# posterior latent outlier probability
gaussian_momratio_outlier_posterior = gaussian_momratio_model.outlier_predictive(
    50, # posterior predictive samples
    truths=truths, # truths dictionary
    fname="example/gaussian_moment_ratio_outlier_posterior.pdf", # plot filename
    prob=0.5, # posterior outlier probability threshold
    seed=1234, # random seed
)

# posterior latent distance probability
gaussian_momratio_distance_posterior = gaussian_momratio_model.distance_predictive(
    50, # posterior predictive samples
    distribution="gaussian", # assumed distribution
    truths=truths, # truths dictionary
    fname="example/gaussian_moment_ratio_distance_posterior.pdf", # plot filename
    seed=1234, # random seed
)

# corner plot
gaussian_momratio_model.plot_corner(
    truths=truths, # optional truths dictionary
    fname="example/gaussian_moment_ratio_corner.pdf", # plot filename
)
```

### Shape

Alternatively, if we assume the shape of the `|z|` distribution, then we can
infer the shape parameter of this distribution directly. The `MarginalDistanceModel`
class in `mcmc.py` does just that, but still samples the *marginal* posterior distribution,
marginalized over the latent distances of each cloud. The distributions and shape
parameters are the same as in `simulate.py`. The prior on the shape parameter is
a `k=2` gamma distribution with a user-supplied mode.

```python
from kinematic_scaleheight.mcmc import ShapeModel
gaussian_shape_model = ShapeModel(
    glong, # Galactic longitudes of clouds (deg)
    glat, # Galactic latitudes of clouds (deg)
    vlsr, # LSR velocities of clouds (km/s)
    distribution="gaussian", # assumed z distribution
    prior_shape=50.0, # mode of the shape prior (pc)
    prior_vlsr_err=10.0, # standard deviation of the LSR velocity error prior (km/s)
    prior_outlier_vlsr_sigma=50.0, # standard deviation of the LSR velocity outlier prior (km/s)
)

# prior predictive checks
gaussian_shape_vlsr_prior = gaussian_shape_model.vlsr_predictive(
    "prior", # prior predictive
    50, # prior predictive samples
    fname="example/gaussian_shape_vlsr_prior.pdf", # plot filename
    seed=1234, # random seed
)

# posterior sampling
gaussian_shape_model.sample(
    init="jitter+adapt_diag", # initialization strategy
    tune=500, # tuning samples
    draws=500, # posterior samples
    cores=4, # number of CPU cores to use
    chains=4, # number of MC chains to run
    target_accept=0.80, # target acceptance rate for sampling
    seed=1234, # random seed
)
"""
Auto-assigning NUTS sampler...
Initializing NUTS using jitter+adapt_diag...
Multiprocess sampling (4 chains in 4 jobs)
NUTS: [rotcurve, outlier_vlsr_sigma, shape, vlsr_err, w]
Sampling 4 chains for 500 tune and 500 draw iterations (2_000 + 2_000 draws total) took 8 seconds.chains, 0 divergences]
                       mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
rotcurve[0]           8.166  0.011   8.145    8.185      0.000    0.000    3038.0    1555.0    1.0
rotcurve[1]          10.206  0.328   9.622   10.854      0.006    0.004    3424.0    1518.0    1.0
rotcurve[2]          15.239  0.546  14.232   16.215      0.012    0.008    2177.0    1806.0    1.0
rotcurve[3]           7.765  0.245   7.326    8.258      0.004    0.003    3671.0    1594.0    1.0
rotcurve[4]           0.964  0.013   0.938    0.988      0.000    0.000    2895.0    1531.0    1.0
rotcurve[5]           1.612  0.003   1.605    1.618      0.000    0.000    2465.0    1819.0    1.0
shape               102.876  5.576  93.559  114.307      0.089    0.063    3983.0    1428.0    1.0
vlsr_err              5.491  0.322   4.881    6.083      0.007    0.005    2302.0    1798.0    1.0
w[0]                  0.862  0.034   0.800    0.925      0.001    0.000    2502.0    1504.0    1.0
w[1]                  0.138  0.034   0.075    0.200      0.001    0.000    2502.0    1504.0    1.0
outlier_vlsr_sigma   21.677  3.079  16.348   27.475      0.058    0.042    2916.0    1864.0    1.0
"""

print(f"Expected: {truths['shape']}") # Expected: 100.0

# posterior predictive check
gaussian_shape_vlsr_posterior = gaussian_shape_model.vlsr_predictive(
    "posterior", # posterior predictive
    50, # posterior predictive samples
    fname="example/gaussian_shape_vlsr_posterior.pdf", # plot filename
    seed=1234, # random seed
)

# posterior latent outlier probability
gaussian_shape_outlier_posterior = gaussian_shape_model.outlier_predictive(
    50, # posterior predictive samples
    truths=truths, # truths dictionary
    fname="example/gaussian_shape_outlier_posterior.pdf", # plot filename
    prob=0.5, # posterior outlier probability threshold
    seed=1234, # random seed
)

# posterior latent distance probability
# N.B. Note that we do not pass a distribution here, since one is set
# for this model at initialization
gaussian_shape_distance_posterior = gaussian_shape_model.distance_predictive(
    50, # posterior predictive samples
    truths=truths, # truths dictionary
    fname="example/gaussian_shape_distance_posterior.pdf", # plot filename
    seed=1234, # random seed
)

# corner plot
gaussian_shape_model.plot_corner(
    truths=truths, # optional truths dictionary
    fname="example/gaussian_shape_corner.pdf", # plot filename
)
```

## Other Distributions

In the preceeding examples, we simulated a "gaussian" distribution. Here we demonstrate
the results for the other supported distributions.

### Exponential

```python
# Generate data
from kinematic_scaleheight.simulate import gen_synthetic_sample
glong, glat, vlsr, truths = gen_synthetic_sample(
    300, # sample size
    distribution='exponential', # vertical distribution shape
    shape=100.0, # shape parameter for the distribution (pc)
    vlsr_err=5.0, # random noise added to observed LSR velocities (km/s)
    b_min=10.0, # minimum Galactic latitude (deg)
    b_max=90.0, # maximum Galactic latitude (deg)
    outlier_vlsr_sigma=30.0, # width of LSR velocity distribution for outliers (km/s)
    outlier_frac=0.1, # fraction of sample that are outliers
    seed=1234, # random seed
    verbose=True, # print helpful information
)
"""
Simulating 300 clouds up to d_max = 5758.770 pc
Added 40 clouds (40/300) in iteration 0
Added 40 clouds (80/300) in iteration 1
Added 52 clouds (132/300) in iteration 2
Added 35 clouds (167/300) in iteration 3
Added 37 clouds (204/300) in iteration 4
Added 50 clouds (254/300) in iteration 5
Added 39 clouds (293/300) in iteration 6
Added 40 clouds (333/300) in iteration 7
Simulation complete. Trimming sample to 300 clouds
"""

# Corrected least-squares
from kinematic_scaleheight.leastsq import leastsq
params, errors, vlsr_rms = leastsq(
    glong[~truths['outlier']], # Galactic longitude of clouds (deg)
    glat[~truths['outlier']], # Galactic latitude of clouds (deg)
    vlsr[~truths['outlier']], # LSR velocities of clouds (km/s)
    R0 = truths['R0'], # Galactocentric radius of the Sun (kpc)
    a2 = truths['a2'], # rotation curve parameter
    a3 = truths['a3'], # rotation curve parameter
)
# mom3_mom2_abs_z_ratio (pc)
print(f"Expected: {truths['mom3_mom2_abs_z_ratio']}") # Expected: 300.0
print(f"Result: {params[0]}") # Result: 312.10982400921495

# Moment ratio MCMC
from kinematic_scaleheight.mcmc import MomentRatioModel
exponential_momratio_model = MomentRatioModel(
    glong, # Galactic longitudes of clouds (deg)
    glat, # Galactic latitudes of clouds (deg)
    vlsr, # LSR velocities of clouds (km/s)
    prior_mom3_mom2_abs_z_ratio=100.0, # mode of the moment ratio prior (pc)
    prior_vlsr_err=10.0, # standard deviation of the LSR velocity error prior (km/s)
    prior_outlier_vlsr_sigma=50.0, # standard deviation of the LSR velocity outlier prior (km/s)
)
exponential_momratio_model.sample(
    init="jitter+adapt_diag", # initialization strategy
    tune=500, # tuning samples
    draws=500, # posterior samples
    cores=4, # number of CPU cores to use
    chains=4, # number of MC chains to run
    target_accept=0.80, # target acceptance rate for sampling
    seed=1234, # random seed
)
"""
Auto-assigning NUTS sampler...
Initializing NUTS using jitter+adapt_diag...
Multiprocess sampling (4 chains in 4 jobs)
NUTS: [rotcurve, outlier_vlsr_sigma, mom3_mom2_abs_z_ratio, vlsr_err, w]
Sampling 4 chains for 500 tune and 500 draw iterations (2_000 + 2_000 draws total) took 9 seconds.chains, 0 divergences]
                          mean      sd   hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
rotcurve[0]              8.168   0.011    8.146    8.187      0.000    0.000    2963.0    1139.0    1.0
rotcurve[1]             10.488   0.360    9.817   11.175      0.006    0.004    3723.0    1582.0    1.0
rotcurve[2]             16.922   0.775   15.477   18.362      0.017    0.012    2013.0    1355.0    1.0
rotcurve[3]              7.684   0.231    7.300    8.155      0.004    0.003    3540.0    1594.0    1.0
rotcurve[4]              0.955   0.014    0.926    0.978      0.000    0.000    2602.0    1428.0    1.0
rotcurve[5]              1.606   0.004    1.599    1.613      0.000    0.000    1965.0    1514.0    1.0
mom3_mom2_abs_z_ratio  293.828  16.219  264.313  324.500      0.292    0.206    3061.0    1448.0    1.0
vlsr_err                 7.244   0.598    6.146    8.360      0.014    0.010    1794.0    1709.0    1.0
w[0]                     0.841   0.038    0.766    0.906      0.001    0.001    1982.0    1408.0    1.0
w[1]                     0.159   0.038    0.094    0.234      0.001    0.001    1982.0    1408.0    1.0
outlier_vlsr_sigma      34.177   4.429   26.939   42.600      0.091    0.067    2648.0    1574.0    1.0
"""

# Shape MCMC
from kinematic_scaleheight.mcmc import ShapeModel
exponential_shape_model = ShapeModel(
    glong, # Galactic longitudes of clouds (deg)
    glat, # Galactic latitudes of clouds (deg)
    vlsr, # LSR velocities of clouds (km/s)
    distribution="exponential", # assumed z distribution
    prior_shape=100.0, # mode of the shape prior (pc)
    prior_vlsr_err=10.0, # standard deviation of the LSR velocity error prior (km/s)
    prior_outlier_vlsr_sigma=50.0, # standard deviation of the LSR velocity outlier prior (km/s)
)
exponential_shape_model.sample(
    init="jitter+adapt_diag", # initialization strategy
    tune=500, # tuning samples
    draws=500, # posterior samples
    cores=4, # number of CPU cores to use
    chains=4, # number of MC chains to run
    target_accept=0.80, # target acceptance rate for sampling
    seed=1234, # random seed
)
"""
Auto-assigning NUTS sampler...
Initializing NUTS using jitter+adapt_diag...
Multiprocess sampling (4 chains in 4 jobs)
NUTS: [rotcurve, outlier_vlsr_sigma, shape, vlsr_err, w]
Sampling 4 chains for 500 tune and 500 draw iterations (2_000 + 2_000 draws total) took 9 seconds.chains, 0 divergences]
                      mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
rotcurve[0]          8.168  0.011   8.148    8.189      0.000    0.000    3554.0    1571.0    1.0
rotcurve[1]         10.490  0.361   9.794   11.159      0.006    0.004    3288.0    1475.0    1.0
rotcurve[2]         16.915  0.762  15.512   18.348      0.017    0.012    2014.0    1693.0    1.0
rotcurve[3]          7.691  0.248   7.198    8.125      0.004    0.003    3408.0    1398.0    1.0
rotcurve[4]          0.955  0.014   0.928    0.979      0.000    0.000    2904.0    1279.0    1.0
rotcurve[5]          1.606  0.004   1.599    1.613      0.000    0.000    1947.0    1480.0    1.0
shape               98.390  5.206  88.772  108.581      0.094    0.066    3082.0    1701.0    1.0
vlsr_err             7.283  0.601   6.156    8.337      0.013    0.009    2044.0    1708.0    1.0
w[0]                 0.843  0.038   0.774    0.913      0.001    0.001    2190.0    1743.0    1.0
w[1]                 0.157  0.038   0.087    0.226      0.001    0.001    2190.0    1743.0    1.0
outlier_vlsr_sigma  34.168  4.718  25.837   42.693      0.100    0.072    2583.0    1618.0    1.0
"""
```

### Rectangular

```python
# Generate data
from kinematic_scaleheight.simulate import gen_synthetic_sample
glong, glat, vlsr, truths = gen_synthetic_sample(
    300, # sample size
    distribution='rectangular', # vertical distribution shape
    shape=100.0, # shape parameter for the distribution (pc)
    vlsr_err=5.0, # random noise added to observed LSR velocities (km/s)
    b_min=10.0, # minimum Galactic latitude (deg)
    b_max=90.0, # maximum Galactic latitude (deg)
    outlier_vlsr_sigma=30.0, # width of LSR velocity distribution for outliers (km/s)
    outlier_frac=0.1, # fraction of sample that are outliers
    seed=1234, # random seed
    verbose=True, # print helpful information
)
"""
Simulating 300 clouds up to d_max = 5758.770 pc
Added 10 clouds (10/300) in iteration 0
Added 8 clouds (18/300) in iteration 1
Added 6 clouds (24/300) in iteration 2
Added 7 clouds (31/300) in iteration 3
Added 13 clouds (44/300) in iteration 4
Added 4 clouds (48/300) in iteration 5
Added 10 clouds (58/300) in iteration 6
Added 7 clouds (65/300) in iteration 7
Added 9 clouds (74/300) in iteration 8
Added 6 clouds (80/300) in iteration 9
Added 3 clouds (83/300) in iteration 10
Added 9 clouds (92/300) in iteration 11
Added 14 clouds (106/300) in iteration 12
Added 7 clouds (113/300) in iteration 13
Added 6 clouds (119/300) in iteration 14
Added 5 clouds (124/300) in iteration 15
Added 5 clouds (129/300) in iteration 16
Added 5 clouds (134/300) in iteration 17
Added 10 clouds (144/300) in iteration 18
Added 5 clouds (149/300) in iteration 19
Added 10 clouds (159/300) in iteration 20
Added 9 clouds (168/300) in iteration 21
Added 11 clouds (179/300) in iteration 22
Added 6 clouds (185/300) in iteration 23
Added 6 clouds (191/300) in iteration 24
Added 7 clouds (198/300) in iteration 25
Added 8 clouds (206/300) in iteration 26
Added 6 clouds (212/300) in iteration 27
Added 4 clouds (216/300) in iteration 28
Added 7 clouds (223/300) in iteration 29
Added 5 clouds (228/300) in iteration 30
Added 3 clouds (231/300) in iteration 31
Added 5 clouds (236/300) in iteration 32
Added 9 clouds (245/300) in iteration 33
Added 10 clouds (255/300) in iteration 34
Added 9 clouds (264/300) in iteration 35
Added 5 clouds (269/300) in iteration 36
Added 4 clouds (273/300) in iteration 37
Added 3 clouds (276/300) in iteration 38
Added 8 clouds (284/300) in iteration 39
Added 9 clouds (293/300) in iteration 40
Added 4 clouds (297/300) in iteration 41
Added 3 clouds (300/300) in iteration 42
Simulation complete. Trimming sample to 300 clouds
"""

# Corrected least-squares
from kinematic_scaleheight.leastsq import leastsq
params, errors, vlsr_rms = leastsq(
    glong[~truths['outlier']], # Galactic longitude of clouds (deg)
    glat[~truths['outlier']], # Galactic latitude of clouds (deg)
    vlsr[~truths['outlier']], # LSR velocities of clouds (km/s)
    R0 = truths['R0'], # Galactocentric radius of the Sun (kpc)
    a2 = truths['a2'], # rotation curve parameter
    a3 = truths['a3'], # rotation curve parameter
)
# mom3_mom2_abs_z_ratio (pc)
print(f"Expected: {truths['mom3_mom2_abs_z_ratio']}") # Expected: 75.0
print(f"Result: {params[0]}") # Result: 70.08395890730624

# Moment ratio MCMC
from kinematic_scaleheight.mcmc import MomentRatioModel
rectangular_momratio_model = MomentRatioModel(
    glong, # Galactic longitudes of clouds (deg)
    glat, # Galactic latitudes of clouds (deg)
    vlsr, # LSR velocities of clouds (km/s)
    prior_mom3_mom2_abs_z_ratio=100.0, # mode of the moment ratio prior (pc)
    prior_vlsr_err=10.0, # standard deviation of the LSR velocity error prior (km/s)
    prior_outlier_vlsr_sigma=50.0, # standard deviation of the LSR velocity outlier prior (km/s)
)
rectangular_momratio_model.sample(
    init="jitter+adapt_diag", # initialization strategy
    tune=500, # tuning samples
    draws=500, # posterior samples
    cores=4, # number of CPU cores to use
    chains=4, # number of MC chains to run
    target_accept=0.80, # target acceptance rate for sampling
    seed=1234, # random seed
)
"""
Auto-assigning NUTS sampler...
Initializing NUTS using jitter+adapt_diag...
Multiprocess sampling (4 chains in 4 jobs)
NUTS: [rotcurve, outlier_vlsr_sigma, mom3_mom2_abs_z_ratio, vlsr_err, w]
Sampling 4 chains for 500 tune and 500 draw iterations (2_000 + 2_000 draws total) took 9 seconds.chains, 0 divergences]
                         mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
rotcurve[0]             8.167  0.011   8.146    8.186      0.000    0.000    2968.0    1252.0   1.00
rotcurve[1]            10.260  0.305   9.706   10.851      0.005    0.004    3256.0    1510.0   1.00
rotcurve[2]            16.266  0.491  15.258   17.115      0.010    0.007    2522.0    1601.0   1.00
rotcurve[3]             7.708  0.234   7.291    8.179      0.004    0.003    3398.0    1596.0   1.00
rotcurve[4]             0.959  0.013   0.933    0.983      0.000    0.000    2773.0    1301.0   1.00
rotcurve[5]             1.608  0.003   1.602    1.614      0.000    0.000    2671.0    1539.0   1.00
mom3_mom2_abs_z_ratio  70.249  8.165  54.545   85.840      0.139    0.099    3438.0    1179.0   1.01
vlsr_err                5.038  0.261   4.562    5.515      0.005    0.003    2952.0    1861.0   1.00
w[0]                    0.898  0.023   0.856    0.942      0.000    0.000    3302.0    1770.0   1.00
w[1]                    0.102  0.023   0.058    0.144      0.000    0.000    3302.0    1770.0   1.00
outlier_vlsr_sigma     34.394  5.543  24.633   44.500      0.113    0.085    2773.0    1398.0   1.00
"""

# Shape MCMC
from kinematic_scaleheight.mcmc import ShapeModel
rectangular_shape_model = ShapeModel(
    glong, # Galactic longitudes of clouds (deg)
    glat, # Galactic latitudes of clouds (deg)
    vlsr, # LSR velocities of clouds (km/s)
    distribution="rectangular", # assumed z distribution
    prior_shape=100.0, # mode of the shape prior (pc)
    prior_vlsr_err=10.0, # standard deviation of the LSR velocity error prior (km/s)
    prior_outlier_vlsr_sigma=50.0, # standard deviation of the LSR velocity outlier prior (km/s)
)
rectangular_shape_model.sample(
    init="jitter+adapt_diag", # initialization strategy
    tune=500, # tuning samples
    draws=500, # posterior samples
    cores=4, # number of CPU cores to use
    chains=4, # number of MC chains to run
    target_accept=0.80, # target acceptance rate for sampling
    seed=1234, # random seed
)
"""
Auto-assigning NUTS sampler...
Initializing NUTS using jitter+adapt_diag...
Multiprocess sampling (4 chains in 4 jobs)
NUTS: [rotcurve, outlier_vlsr_sigma, shape, vlsr_err, w]
Sampling 4 chains for 500 tune and 500 draw iterations (2_000 + 2_000 draws total) took 9 seconds.chains, 0 divergences]
                      mean      sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
rotcurve[0]          8.167   0.011   8.148    8.189      0.000    0.000    2954.0    1705.0   1.00
rotcurve[1]         10.271   0.316   9.662   10.851      0.006    0.004    3148.0    1707.0   1.00
rotcurve[2]         16.246   0.478  15.397   17.205      0.010    0.007    2389.0    1408.0   1.01
rotcurve[3]          7.711   0.232   7.274    8.123      0.004    0.003    3071.0    1888.0   1.01
rotcurve[4]          0.959   0.013   0.935    0.984      0.000    0.000    2851.0    1627.0   1.00
rotcurve[5]          1.608   0.003   1.602    1.614      0.000    0.000    2214.0    1464.0   1.00
shape               93.342  10.318  74.753  112.936      0.191    0.135    2929.0    1676.0   1.00
vlsr_err             5.040   0.262   4.565    5.509      0.005    0.003    3040.0    1452.0   1.00
w[0]                 0.898   0.024   0.855    0.942      0.000    0.000    2740.0    1843.0   1.00
w[1]                 0.102   0.024   0.058    0.145      0.000    0.000    2740.0    1843.0   1.00
outlier_vlsr_sigma  34.193   5.279  24.346   43.348      0.107    0.079    2699.0    1710.0   1.00
"""
```

## Model Comparison

Here we demonstrate how we can use the MCMC posterior samples to determine which
distribution best represents the data.

```python
# Generate data from a Gaussian distribution
from kinematic_scaleheight.simulate import gen_synthetic_sample
glong, glat, vlsr, truths = gen_synthetic_sample(
    300, # sample size
    distribution='gaussian', # vertical distribution shape
    shape=100.0, # shape parameter for the distribution (pc)
    vlsr_err=5.0, # random noise added to observed LSR velocities (km/s)
    b_min=10.0, # minimum Galactic latitude (deg)
    b_max=90.0, # maximum Galactic latitude (deg)
    outlier_vlsr_sigma=30.0, # width of LSR velocity distribution for outliers (km/s)
    outlier_frac=0.1, # fraction of sample that are outliers
    seed=1234, # random seed
    verbose=True, # print helpful information
)
"""
Simulating 300 clouds up to d_max = 5758.770 pc
Added 22 clouds (22/300) in iteration 0
Added 19 clouds (41/300) in iteration 1
Added 23 clouds (64/300) in iteration 2
Added 24 clouds (88/300) in iteration 3
Added 24 clouds (112/300) in iteration 4
Added 23 clouds (135/300) in iteration 5
Added 23 clouds (158/300) in iteration 6
Added 17 clouds (175/300) in iteration 7
Added 28 clouds (203/300) in iteration 8
Added 23 clouds (226/300) in iteration 9
Added 26 clouds (252/300) in iteration 10
Added 29 clouds (281/300) in iteration 11
Added 20 clouds (301/300) in iteration 12
Simulation complete. Trimming sample to 300 clouds
"""

# Infer shape parameter assuming Gaussian distribution
from kinematic_scaleheight.mcmc import ShapeModel
gaussian_shape_model = ShapeModel(
    glong, # Galactic longitudes of clouds (deg)
    glat, # Galactic latitudes of clouds (deg)
    vlsr, # LSR velocities of clouds (km/s)
    distribution="gaussian", # assumed z distribution
    prior_shape=50.0, # mode of the shape prior (pc)
    prior_vlsr_err=10.0, # standard deviation of the LSR velocity error prior (km/s)
    prior_outlier_vlsr_sigma=50.0, # standard deviation of the LSR velocity outlier prior (km/s)
)
gaussian_shape_model.sample(
    init="jitter+adapt_diag", # initialization strategy
    tune=500, # tuning samples
    draws=500, # posterior samples
    cores=4, # number of CPU cores to use
    chains=4, # number of MC chains to run
    target_accept=0.80, # target acceptance rate for sampling
    seed=1234, # random seed
)
"""
Auto-assigning NUTS sampler...
Initializing NUTS using jitter+adapt_diag...
Multiprocess sampling (4 chains in 4 jobs)
NUTS: [rotcurve, outlier_vlsr_sigma, shape, vlsr_err, w]
Sampling 4 chains for 500 tune and 500 draw iterations (2_000 + 2_000 draws total) took 9 seconds.chains, 0 divergences]
                       mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
rotcurve[0]           8.166  0.011   8.145    8.185      0.000    0.000    3038.0    1555.0    1.0
rotcurve[1]          10.206  0.328   9.622   10.854      0.006    0.004    3424.0    1518.0    1.0
rotcurve[2]          15.239  0.546  14.232   16.215      0.012    0.008    2177.0    1806.0    1.0
rotcurve[3]           7.765  0.245   7.326    8.258      0.004    0.003    3671.0    1594.0    1.0
rotcurve[4]           0.964  0.013   0.938    0.988      0.000    0.000    2895.0    1531.0    1.0
rotcurve[5]           1.612  0.003   1.605    1.618      0.000    0.000    2465.0    1819.0    1.0
shape               102.876  5.576  93.559  114.307      0.089    0.063    3983.0    1428.0    1.0
vlsr_err              5.491  0.322   4.881    6.083      0.007    0.005    2302.0    1798.0    1.0
w[0]                  0.862  0.034   0.800    0.925      0.001    0.000    2502.0    1504.0    1.0
w[1]                  0.138  0.034   0.075    0.200      0.001    0.000    2502.0    1504.0    1.0
outlier_vlsr_sigma   21.677  3.079  16.348   27.475      0.058    0.042    2916.0    1864.0    1.0
"""

# Infer shape parameter assuming exponential distribution
from kinematic_scaleheight.mcmc import ShapeModel
exponential_shape_model = ShapeModel(
    glong, # Galactic longitudes of clouds (deg)
    glat, # Galactic latitudes of clouds (deg)
    vlsr, # LSR velocities of clouds (km/s)
    distribution="exponential", # assumed z distribution
    prior_shape=100.0, # mode of the shape prior (pc)
    prior_vlsr_err=10.0, # standard deviation of the LSR velocity error prior (km/s)
    prior_outlier_vlsr_sigma=50.0, # standard deviation of the LSR velocity outlier prior (km/s)
)
exponential_shape_model.sample(
    init="jitter+adapt_diag", # initialization strategy
    tune=500, # tuning samples
    draws=500, # posterior samples
    cores=4, # number of CPU cores to use
    chains=4, # number of MC chains to run
    target_accept=0.80, # target acceptance rate for sampling
    seed=1234, # random seed
)
"""
Auto-assigning NUTS sampler...
Initializing NUTS using jitter+adapt_diag...
Multiprocess sampling (4 chains in 4 jobs)
NUTS: [rotcurve, outlier_vlsr_sigma, shape, vlsr_err, w]
Sampling 4 chains for 500 tune and 500 draw iterations (2_000 + 2_000 draws total) took 8 seconds.chains, 0 divergences]
The rhat statistic is larger than 1.01 for some parameters. This indicates problems during sampling. See https://arxiv.org/abs/1903.08008 for details
                      mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
rotcurve[0]          8.167  0.011   8.144    8.186      0.000    0.000    3207.0    1432.0   1.00
rotcurve[1]         10.214  0.335   9.599   10.854      0.006    0.004    3698.0    1544.0   1.00
rotcurve[2]         15.227  0.560  14.199   16.273      0.012    0.008    2349.0    1472.0   1.00
rotcurve[3]          7.777  0.238   7.359    8.235      0.004    0.003    3877.0    1615.0   1.00
rotcurve[4]          0.963  0.013   0.939    0.989      0.000    0.000    1963.0    1269.0   1.01
rotcurve[5]          1.612  0.003   1.606    1.618      0.000    0.000    2321.0    1516.0   1.00
shape               54.906  2.944  49.561   60.523      0.050    0.035    3471.0    1475.0   1.00
vlsr_err             5.479  0.313   4.887    6.056      0.006    0.004    2738.0    1538.0   1.00
w[0]                 0.862  0.034   0.793    0.917      0.001    0.000    2559.0    1501.0   1.00
w[1]                 0.138  0.034   0.083    0.207      0.001    0.000    2559.0    1501.0   1.00
outlier_vlsr_sigma  21.750  3.262  16.343   27.793      0.067    0.050    2700.0    1063.0   1.00
"""

# Infer shape parameter assuming rectangular distribution
from kinematic_scaleheight.mcmc import ShapeModel
rectangular_shape_model = ShapeModel(
    glong, # Galactic longitudes of clouds (deg)
    glat, # Galactic latitudes of clouds (deg)
    vlsr, # LSR velocities of clouds (km/s)
    distribution="rectangular", # assumed z distribution
    prior_shape=100.0, # mode of the shape prior (pc)
    prior_vlsr_err=10.0, # standard deviation of the LSR velocity error prior (km/s)
    prior_outlier_vlsr_sigma=50.0, # standard deviation of the LSR velocity outlier prior (km/s)
)
rectangular_shape_model.sample(
    init="jitter+adapt_diag", # initialization strategy
    tune=500, # tuning samples
    draws=500, # posterior samples
    cores=4, # number of CPU cores to use
    chains=4, # number of MC chains to run
    target_accept=0.80, # target acceptance rate for sampling
    seed=1234, # random seed
)
"""
Auto-assigning NUTS sampler...
Initializing NUTS using jitter+adapt_diag...
Multiprocess sampling (4 chains in 4 jobs)
NUTS: [rotcurve, outlier_vlsr_sigma, shape, vlsr_err, w]
Sampling 4 chains for 500 tune and 500 draw iterations (2_000 + 2_000 draws total) took 9 seconds.chains, 0 divergences]
                       mean      sd   hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
rotcurve[0]           8.166   0.011    8.144    8.186      0.000    0.000    3179.0    1582.0   1.00
rotcurve[1]          10.207   0.335    9.537   10.796      0.006    0.004    2913.0    1406.0   1.00
rotcurve[2]          15.234   0.537   14.263   16.219      0.011    0.008    2566.0    1528.0   1.01
rotcurve[3]           7.761   0.236    7.316    8.196      0.004    0.003    3432.0    1343.0   1.00
rotcurve[4]           0.963   0.013    0.939    0.989      0.000    0.000    3290.0    1709.0   1.01
rotcurve[5]           1.612   0.003    1.605    1.617      0.000    0.000    2344.0    1490.0   1.01
shape               218.496  11.838  196.722  241.011      0.206    0.146    3295.0    1361.0   1.00
vlsr_err              5.489   0.330    4.892    6.122      0.006    0.004    2931.0    1774.0   1.00
w[0]                  0.861   0.035    0.798    0.921      0.001    0.000    2841.0    1589.0   1.00
w[1]                  0.139   0.035    0.079    0.202      0.001    0.000    2841.0    1589.0   1.00
outlier_vlsr_sigma   21.750   3.268   16.003   27.720      0.070    0.051    2507.0    1570.0   1.00
"""

# Leave-one-out (LOO) cross-validation
import arviz as az
compare_loo = az.compare({
    "gaussian": gaussian_shape_model.trace,
    "exponential": exponential_shape_model.trace,
    "rectangular": rectangular_shape_model.trace,
})
print(compare_loo)
"""
             rank     elpd_loo     p_loo  elpd_diff        weight         se       dse  warning scale
rectangular     0 -1042.460891  5.012141   0.000000  1.000000e+00  19.133027  0.000000    False   log
gaussian        1 -1042.478919  5.070481   0.018029  0.000000e+00  19.168408  0.066034    False   log
exponential     2 -1042.501990  5.074365   0.041099  3.330669e-16  19.163474  0.113451    False   log
"""
```

In this example, each of the three distributions is equally likely. The distance uncertainties due
to the uncertain Galactic rotation curve wash out any evidence that distinguishes between the three
models.

# Issues and Contributing

Anyone is welcome to submit issues or contribute to the development
of this software via [Github](https://github.com/tvwenger/kinematic_scaleheight).

# License and Copyright

Copyright (c) 2023-2024 Trey Wenger

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
