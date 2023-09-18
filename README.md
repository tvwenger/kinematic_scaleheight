# kinematic_scaleheight
Use MCMC methods to kinematically estimate the vertical distribution of clouds in the Galactic plane

## Installation
```bash
conda create --name kinematic_scaleheight -c conda-forge pymc
conda activate kinematic_scaleheight
pip install git+https://github.com/tvwenger/kinematic_scaleheight.git
pip install git+https://github.com/tvwenger/pymc-experimental.git@chi_rv
```

## Usage
`simulate.py` is capable of generating a synthetic sample of clouds with which to test the MCMC methods.

```python
from kinematic_scaleheight.simulate import gen_synthetic_sample
glong, glat, vlsr, truths = gen_synthetic_sample(
    200, # sample size
    100.0, # standard deviation of the Gaussian vertical distribution of clouds (pc)
    vlsr_err=10.0, # noise added to observed LSR velocities (km/s)
    d_max=1000.0, # maximum distance of the clouds (pc)
    b_min=0.0, # minimum Galactic latitude (deg)
    b_max=90.0, # maximum Galactic latitude (deg)
    seed=1234, # random seed
)
print(truths.keys()) # dict_keys(['distance', 'sigma_z', 'vlsr_err', 'R0', 'Usun', 'Vsun', 'Wsun', 'a2', 'a3'])
```

`mcmc.py` performs the posterior samplling.

```python
from kinematic_scaleheight.mcmc import Model
model = Model(
    glong, # Galactic longitudes of clouds (deg)
    glat, # Galactic latitudes of clouds (deg)
    vlsr, # LSR velocities of clouds (km/s)
    prior_sigma_z=100.0, # half-width of sigma_z prior (pc)
    prior_vlsr_err=10.0, # half-width of LSR velocity error prior (km/s)
    d_max=1000.0, # maximum distance of the clouds to consider (pc)
)

# prior predictive check
!mkdir example
model.plot_predictive(
    "prior", # prior predictive
    50, # prior predictive samples
    truths=truths, # optional truths dictionary
    plot_prefix="example/", # plot filename prefix
)

# posterior sampling
model.sample(
    init="adapt_diag", # initialization strategy
    tune=500, # tuning samples
    draws=500, # posterior samples
    cores=4, # number of CPU cores to use
    chains=4, # number of MC chains to run
    target_accept=0.8, # target acceptance rate for sampling
)

# posterior predictive check
model.plot_predictive(
    "posterior", # posterior predictive
    50, # posterior predictive samples
    truths=truths, # optional truths dictionary
    plot_prefix="example/", # plot filename prefix
)

# corner plot
model.plot_corner(
    truths=truths, # optional truths dictionary
    plot_prefix="example/", # plot filename prefix
)
```