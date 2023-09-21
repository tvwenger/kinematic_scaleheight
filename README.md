# kinematic_scaleheight
Use MCMC methods to kinematically estimate the vertical distribution of clouds in the Galactic plane

## Installation
```bash
conda create --name kinematic_scaleheight -c conda-forge pymc==5.8.2
conda activate kinematic_scaleheight
pip install --upgrade git+https://github.com/tvwenger/kinematic_scaleheight.git
pip install --upgrade git+https://github.com/tvwenger/pymc-experimental.git@chi_rv
```

## Usage
`simulate.py` is capable of generating a synthetic sample of clouds with which to test the MCMC methods.

```python
from kinematic_scaleheight.simulate import gen_synthetic_sample
glong, glat, vlsr, truths = gen_synthetic_sample(
    500, # sample size
    100.0, # standard deviation of the Gaussian vertical distribution of clouds (pc)
    vlsr_err=5.0, # noise added to observed LSR velocities (km/s)
    d_max=2000.0, # maximum distance of the clouds (pc)
    b_min=10.0, # minimum Galactic latitude (deg)
    b_max=90.0, # maximum Galactic latitude (deg)
    seed=1234, # random seed
)
print(truths.keys()) # dict_keys(['distance', 'sigma_z', 'vlsr_err', 'R0', 'Usun', 'Vsun', 'Wsun', 'a2', 'a3'])
```

`leastsq.py` uses the least squares analysis of Crovisier et al. (1978) to estimate
the standard deviation of the vertical distribution of clouds.

```python
from kinematic_scaleheight.leastsq import crovisier
params, errors, vlsr_rms = crovisier(
    glong, # Galactic longitude of clouds (deg)
    glat, # Galactic latitude of clouds (deg)
    vlsr, # LSR velocities of clouds (km/s)
    oortA = 15.3, # Oort's A constant (km/s/kpc)
    corrected = False, # apply Wenger et al. (2023) correction
)
# params contains least-squares fit for (sigma_z [pc], Usun [km/s], Vsun [km/s], Wsun [km/s])
print(params) # [195.14833432  -0.47117138  -0.70529308   0.64153728]
# errors contains the standard deviations
print(errors) # [1.33441208 0.06694245 0.0671935  0.1382609 ]
# vlsr_rms is the rms LSR velocity residual (km/s)
print(vlsr_rms) # 5.787341705253839

# With Wenger et al. (2023) correction
params, errors, vlsr_rms = crovisier(glong, glat, vlsr, oortA=15.3, corrected=True)
print(params) # [97.57416702 -0.47117137 -0.70529307  0.64153743]
print(errors) # [0.66720604 0.06694245 0.0671935  0.13826091]
print(vlsr_rms) # 5.787341701811527
```

`mcmc.py` performs the posterior sampling to determine the posterior distribution
of the various model parameters, including the standard deviation of the
vertical distribution of clouds.

```python
# drop sources in direction of Galactic center/anti-center, for which
# distances are poorly constrained kinematically. The sampler will converge
# *much* faster
import numpy as np
bad = (glong < 15.0) + (np.abs(glong - 180.0) < 20.0) + (glong > 345.0)
glong = glong[~bad]
glat = glat[~bad]
vlsr = vlsr[~bad]
truths['distance'] = truths['distance'][~bad]
print(len(glong)) # 388

from kinematic_scaleheight.mcmc import Model
model = Model(
    glong, # Galactic longitudes of clouds (deg)
    glat, # Galactic latitudes of clouds (deg)
    vlsr, # LSR velocities of clouds (km/s)
    prior_sigma_z=50.0, # mode of the sigma_z (Gamma distribution) prior (pc)
    prior_vlsr_err=10.0, # standard deviation of the LSR velocity error (half-normal distribution) prior (km/s)
    d_max=5000.0, # maximum distance of the clouds to consider (pc)
)

# prior predictive checks
!mkdir example
model.plot_predictive(
    "prior", # prior predictive
    50, # prior predictive samples
    truths=truths, # optional truths dictionary
    plot_prefix="example/", # plot filename prefix
)
```

![Longitude-Velocity prior](https://raw.githubusercontent.com/tvwenger/kinematic_scaleheight/main/example/lv_prior.png)
![Longitude-Distance prior](https://raw.githubusercontent.com/tvwenger/kinematic_scaleheight/main/example/ld_prior.png)

```python
# posterior sampling
model.sample(
    init="adapt_diag", # initialization strategy
    tune=1000, # tuning samples
    draws=1000, # posterior samples
    cores=8, # number of CPU cores to use
    chains=8, # number of MC chains to run
    target_accept=0.95, # target acceptance rate for sampling
)
```

The output is appended below. Notice that there are some divergences because the distances are
hard to constrain kinematically. The parameters of interest appear converged, however.

```
Auto-assigning NUTS sampler...
Initializing NUTS using adapt_diag...
Multiprocess sampling (8 chains in 8 jobs)
NUTS: [rotcurve, sigma_z, distance, vlsr_err]
Sampling 8 chains for 1_000 tune and 1_000 draw iterations (8_000 + 8_000 draws total) took 56 seconds.
There were 203 divergences after tuning. Increase `target_accept` or reparameterize.
                mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
rotcurve[0]    8.167  0.011   8.147    8.189      0.000    0.000   10460.0    5570.0    1.0
rotcurve[1]   10.405  0.328   9.775   11.013      0.004    0.002    8668.0    5695.0    1.0
rotcurve[2]   15.878  0.368  15.179   16.568      0.005    0.003    6495.0    6051.0    1.0
rotcurve[3]    7.689  0.233   7.228    8.113      0.002    0.002    9336.0    5825.0    1.0
rotcurve[4]    0.959  0.013   0.933    0.982      0.000    0.000    9582.0    5891.0    1.0
rotcurve[5]    1.609  0.003   1.604    1.615      0.000    0.000    7207.0    5843.0    1.0
sigma_z      101.340  4.382  93.089  109.449      0.113    0.080    1505.0    2733.0    1.0
vlsr_err       4.940  0.225   4.512    5.347      0.003    0.002    4812.0    5576.0    1.0
```

```python
# posterior predictive check
model.plot_predictive(
    "posterior", # posterior predictive
    50, # posterior predictive samples
    truths=truths, # optional truths dictionary
    plot_prefix="example/", # plot filename prefix
)
```

![Longitude-Velocity posterior](https://raw.githubusercontent.com/tvwenger/kinematic_scaleheight/main/example/lv_posterior.png)
![Longitude-Distance posterior](https://raw.githubusercontent.com/tvwenger/kinematic_scaleheight/main/example/ld_posterior.png)

```python
# corner plot
model.plot_corner(
    truths=truths, # optional truths dictionary
    plot_prefix="example/", # plot filename prefix
)
```

![Corner](https://raw.githubusercontent.com/tvwenger/kinematic_scaleheight/main/example/corner.png)