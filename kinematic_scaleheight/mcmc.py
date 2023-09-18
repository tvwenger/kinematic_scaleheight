"""
mcmc.py
Use MCMC methods to estimate the vertical distribution of a population
of clouds in the Galactic disk.

Copyright(C) 2023 by
Trey V. Wenger; tvwenger@gmail.com

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

Trey Wenger - June 2023
"""

import os
import pickle
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import corner

from pymc_experimental.distributions import Chi

from .rotation import reid19_vlsr


class Model:
    """
    Storage and utilities for MCMC model
    """

    def __init__(
        self,
        glong,
        glat,
        vlsr,
        prior_sigma_z=100.0,
        prior_vlsr_err=10.0,
        d_max=1000.0,
    ):
        """
        Initialize a new model.

        Inputs:
            glong :: 1-D array of scalars (deg)
                Galactic longitude of clouds
            glat :: 1-D array of scalars (deg)
                Galactic latitude of clouds
            vlsr :: 1-D array of scalars (km/s)
                LSR velocities of clouds
            prior_sigma_z :: scalar (pc)
                Width of half-normal prior distribution on sigma_z
            prior_vlsr_err :: scalar (km/s)
                Width of half-normal prior distribution on vlsr_err
            d_max :: scalar (pc)
                Truncate distance prior at this distance

        Returns: model
            model :: Model
                New model instance
        """
        self.glong = glong
        self.glat = glat
        self.vlsr = vlsr
        self.trace = None

        # wrap longitude for plotting
        self.wrap_glong = glong % 360.0
        wrap = self.wrap_glong > 180.0
        self.wrap_glong[wrap] = self.wrap_glong[wrap] - 360.0

        # sample size
        self.size = len(glong)

        if len(glat) != self.size or len(vlsr) != self.size:
            raise ValueError("Shape mismatch between glong, glat, and vlsr")

        # Load Reid et al. (2019) kernel density estimate mean
        # and covariance for the rotation curve parameters
        fname = os.path.join(os.path.dirname(__file__), "data/reid19_mv.pkl")
        with open(fname, "rb") as f:
            reid19_mean, reid19_cov = pickle.load(f)

        # Define model
        with pm.Model() as self.model:
            # multivariate-normal distribution in rotation curve parameters
            R0, Usun, Vsun, Wsun, a2, a3 = pm.MvNormal(
                "rotcurve", mu=reid19_mean, cov=reid19_cov
            )

            # scale height
            sigma_z = pm.Gamma("sigma_z", alpha=2.0, beta=1.0 / prior_sigma_z)

            # distance
            distance = pm.Truncated(
                "distance",
                Chi.dist(
                    df=3, sigma=sigma_z / pm.math.sin(pm.math.abs(np.deg2rad(glat)))
                ),
                upper=d_max,
                shape=(self.size,),
            )

            # velocity
            vlsr_err = pm.HalfNormal("vlsr_err", sigma=prior_vlsr_err)
            vlsr_mu = reid19_vlsr(
                glong,
                glat,
                distance / 1000.0,  # kpc
                R0=R0,
                a2=a2,
                a3=a3,
                Usun=Usun,
                Vsun=Vsun,
                Wsun=Wsun,
            )
            _ = pm.Normal("vlsr", mu=vlsr_mu, sigma=vlsr_err, observed=vlsr)

    def plot_predictive(self, predtype, num, truths=None, plot_prefix=""):
        """
        Generate prior or posterior predictive samples and plots. The plots are:
        name                description
        lv_{predtype}.pdf        Longitude-velocity predictive
        ld_{predtype}.pdf        Longitude-distance predictive

        Inputs:
            predtype :: string
                One of "prior" or "posterior"
            num :: integer
                Number of predictive samples to generate
            truths :: dictionary
                If not None, the dictionary should include the key
                'distance' with the true distance for each target.
                These values are incorporated to the relevant plots.
            plot_prefix :: string
                Save plots with names like {plot_prefix}{name}.pdf

        Returns: predictive
            predictive :: arviz.InferenceData
                Prior/posterior and prior/posterior predictive samples
        """
        # prior predictive samples
        with self.model:
            if predtype == "prior":
                trace = pm.sample_prior_predictive(samples=num)
                samples = trace.prior
                predictive = trace.prior_predictive
            elif predtype == "posterior":
                # thin the posterior samples to requested sample size
                total_samples = len(self.trace.posterior.chain) * len(
                    self.trace.posterior.draw
                )
                thin = total_samples // num
                trace = pm.sample_posterior_predictive(
                    self.trace.sel(draw=slice(None, None, thin))
                )
                samples = self.trace.posterior
                predictive = trace.posterior_predictive

        # longitude-velocity
        fig, ax = plt.subplots()
        for chain in predictive.chain:
            for draw in predictive.draw:
                ax.plot(
                    predictive["vlsr"].sel(chain=chain, draw=draw),
                    self.wrap_glong,
                    ".",
                    alpha=0.1,
                    markersize=1,
                )
        ax.plot(self.vlsr, self.wrap_glong, "r.", markersize=1, label="Observed")
        ax.set_xlabel(r"$V_{\rm LSR}$ (km s$^{-1}$)")
        ax.set_ylabel(r"Galactic Longitude (deg)")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(f"{plot_prefix}lv_{predtype}.pdf", bbox_inches="tight")
        plt.close(fig)

        # longitude-distance
        fig, ax = plt.subplots()
        for chain in predictive.chain:
            for draw in predictive.draw:
                ax.plot(
                    samples["distance"].sel(chain=chain, draw=draw),
                    self.wrap_glong,
                    "k.",
                    alpha=0.1,
                    markersize=1,
                )
        if truths is not None:
            ax.plot(
                truths["distance"],
                self.wrap_glong,
                "r.",
                markersize=1,
                label="Truth",
            )
            ax.legend(loc="best")
        ax.set_xlabel(r"Distance (pc)")
        ax.set_ylabel(r"Galactic Longitude (deg)")
        fig.tight_layout()
        fig.savefig(f"{plot_prefix}ld_{predtype}.pdf", bbox_inches="tight")
        plt.close(fig)

    def sample(
        self,
        tune=1000,
        draws=1000,
        cores=4,
        chains=4,
        init="adapt_diag",
        target_accept=0.8,
    ):
        """
        Generate posterior samples of the model.

        Inputs:
            tune :: integer
                Number of tuning samples
            draws :: integer
                Number of posterior samples
            cores :: integer
                Number of CPU cores to use in parallel
            chains :: integer
                Number of MC chains
            init :: string
                Initialization method. Default is the pymc default: "jitter+adapt_diag"
            target_accept :: scalar
                Target accept fraction

        Returns: Nothing
        """
        # sample posterior
        with self.model:
            self.trace = pm.sample(
                init=init,
                tune=tune,
                draws=draws,
                cores=cores,
                chains=chains,
                target_accept=target_accept,
            )
        print(
            pm.summary(
                self.trace,
                var_names=["rotcurve", "sigma_z", "vlsr_err"],
            )
        )

    def plot_corner(self, truths=None, plot_prefix=""):
        """
        Generate corner plot of posterior samples.

        Inputs:
            truths :: dictionary
                Truths dictionary
            plot_prefix :: string
                Save plot with name like {plot_prefix}corner.pdf

        Returns: Nothing
        """
        truth_vals = None
        if truths is not None:
            truth_vals = [
                truths["R0"],
                truths["Usun"],
                truths["Vsun"],
                truths["Wsun"],
                truths["a2"],
                truths["a3"],
                truths["sigma_z"],
                truths["vlsr_err"],
            ]
        fig = corner.corner(
            self.trace, var_names=["rotcurve", "sigma_z", "vlsr_err"], truths=truth_vals
        )
        fig.savefig(f"{plot_prefix}corner.pdf", bbox_inches="tight")
        plt.close(fig)
