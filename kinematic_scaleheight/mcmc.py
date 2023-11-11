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
Trey Wenger - November 2023 - add outlier component, remove d_max,
    remove FullDistanceModel
"""

import os
import pickle
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import corner

from pymc_experimental.distributions import Chi

from kinematic_scaleheight.rotation import reid19_vlsr


class BaseModel:
    """
    Base model for MCMC analyses
    """

    def __init__(
        self,
        glong,
        glat,
        vlsr,
        prior_outlier_vlsr_sigma,
    ):
        """
        Initialize a new BaseModel instance.

        Inputs:
            glong :: 1-D array of scalars (deg)
                Galactic longitude of clouds
            glat :: 1-D array of scalars (deg)
                Galactic latitude of clouds
            vlsr :: 1-D array of scalars (km/s)
                LSR velocities of clouds
            prior_outlier_vlsr_sigma :: scalar (km/s)
                Width of half-normal distribution prior on outlier_vlsr_sigma

        Returns: model
            model :: BaseModel
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
        with pm.Model(coords={"data": range(self.size)}) as self.model:
            # multivariate-normal distribution in rotation curve parameters
            # order: R0, Usun, Vsun, Wsun, a2, a3
            _ = pm.MvNormal("rotcurve", mu=reid19_mean, cov=reid19_cov)

            # outlier component
            outlier_vlsr_sigma = pm.HalfNormal(
                "outlier_vlsr_sigma", sigma=prior_outlier_vlsr_sigma
            )
            outlier_vlsr_mu = 0.0
            self.outlier_vlsr_dist = pm.Normal.dist(
                mu=outlier_vlsr_mu, sigma=outlier_vlsr_sigma
            )

    def vlsr_predictive(self, predtype, num, fname=None, seed=1234):
        """
        Generate prior or posterior predictive LSR velocity samples and plot.

        Inputs:
            predtype :: string
                One of "prior" or "posterior"
            num :: integer
                Number of predictive samples to generate
            fname :: string
                If not None, generate plot to this filename
            seed :: integer
                Random seed

        Returns: trace
            trace :: arviz.InferenceData
                Predictive samples
        """
        # prior predictive samples
        with self.model:
            if predtype == "prior":
                trace = pm.sample_prior_predictive(samples=num, random_seed=seed)
                predictive = trace.prior_predictive
            elif predtype == "posterior":
                # thin the posterior samples to requested sample size
                total_samples = len(self.trace.posterior.chain) * len(
                    self.trace.posterior.draw
                )
                thin = total_samples // num
                trace = pm.sample_posterior_predictive(
                    self.trace.sel(draw=slice(None, None, thin)), random_seed=seed
                )
                predictive = trace.posterior_predictive
            else:
                raise ValueError(f"invalid predtype {predtype}")

        if fname is not None:
            fig, ax = plt.subplots(figsize=(6, 8))
            for chain in predictive.chain:
                for draw in predictive.draw:
                    ax.plot(
                        predictive["vlsr"].sel(chain=chain, draw=draw),
                        self.wrap_glong,
                        ".",
                        alpha=0.1,
                        markersize=1,
                    )
            ax.plot(self.vlsr, self.wrap_glong, "r.", markersize=1, label="Data")
            max_vlsr = np.abs(self.vlsr).max()
            ax.set_xlim(-1.5 * max_vlsr, 1.5 * max_vlsr)
            ax.set_ylim(-200.0, 200.0)
            ax.set_xlabel(r"$V_{\rm LSR}$ (km s$^{-1}$)")
            ax.set_ylabel(r"Galactic Longitude (deg)")
            ax.legend(loc="best", fontsize=10)
            fig.tight_layout()
            fig.savefig(fname, bbox_inches="tight")
            plt.close(fig)

        return trace

    def sample(
        self,
        tune=1000,
        draws=1000,
        cores=4,
        chains=4,
        init="adapt_diag",
        target_accept=0.8,
        seed=1234,
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
                Initialization method.
            target_accept :: scalar
                Target accept fraction
            seed :: integer
                Random seed

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
                random_seed=seed,
            )
        print(
            pm.summary(
                self.trace,
                var_names=self.var_names,
            )
        )

    def plot_corner(self, truths=None, fname="corner.pdf"):
        """
        Generate corner plot of posterior samples.

        Inputs:
            truths :: dictionary
                Truths dictionary. If None, do not identify truths.
            fname :: string
                Plot filename

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
                truths["shape"]
                if "shape" in self.var_names
                else truths["mom3_mom2_abs_z_ratio"],
                truths["vlsr_err"],
                1.0 - truths["outlier_frac"],
                truths["outlier_frac"],
                truths["outlier_vlsr_sigma"],
            ]
        fig = corner.corner(self.trace, var_names=self.var_names, truths=truth_vals)
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)

    def outlier_predictive(self, num, truths=None, fname=None, prob=0.5, seed=1234):
        """
        Draw posterior samples for latent outlier classification.

        Inputs:
            num :: integer
                Number of predictive samples to generate
            truths :: dictionary
                Truths dictionary. If None, do not identify truths
            fname :: string
                If not None, generate longitude-velocity diagram and
                identify outliers.
            prob :: scalar
                Outliers are classified where the outlier likelihood exceeds
                this probability.
            seed :: integer
                Random seed
        Returns: trace
            trace :: arviz.InferenceData
                Posterior predictive
        """
        # add component likelihood to model
        if "outlier" not in self.model.named_vars:
            with self.model:
                model_logp = pm.math.log(self.model.w[0]) + pm.logp(
                    self.model_vlsr_dist, self.vlsr
                )
                outlier_logp = pm.math.log(self.model.w[1]) + pm.logp(
                    self.outlier_vlsr_dist, self.vlsr
                )
                log_probs = pm.math.concatenate([[model_logp], [outlier_logp]], axis=0)
                _ = pm.Categorical("outlier", logit_p=log_probs.T, dims="data")

        # sample
        with self.model:
            # thin the posterior samples to requested sample size
            total_samples = len(self.trace.posterior.chain) * len(
                self.trace.posterior.draw
            )
            thin = total_samples // num
            trace = pm.sample_posterior_predictive(
                self.trace.sel(draw=slice(None, None, thin)),
                var_names=["outlier"],
                random_seed=seed,
            )

        if fname is not None:
            # identify outliers
            outlier = (
                (trace.posterior_predictive.mean(("chain", "draw")) > prob)
                .to_array()
                .data
            )[0]
            fig, ax = plt.subplots(figsize=(6, 8))
            if truths is None:
                ax.plot(self.vlsr, self.wrap_glong, "k.", label="Data")
            else:
                ax.plot(
                    self.vlsr[~truths["outlier"]],
                    self.wrap_glong[~truths["outlier"]],
                    "k.",
                    label="Non-outlier Data",
                )
                ax.plot(
                    self.vlsr[truths["outlier"]],
                    self.wrap_glong[truths["outlier"]],
                    "r.",
                    label="Outlier Data",
                )
            ax.plot(
                self.vlsr[outlier],
                self.wrap_glong[outlier],
                "bo",
                label="Identified Outliers",
                alpha=0.5,
            )
            max_vlsr = np.abs(self.vlsr).max()
            ax.set_xlim(-1.5 * max_vlsr, 1.5 * max_vlsr)
            ax.set_ylim(-200.0, 200.0)
            ax.set_xlabel(r"$V_{\rm LSR}$ (km s$^{-1}$)")
            ax.set_ylabel(r"Galactic Longitude (deg)")
            ax.legend(loc="best", fontsize=10)
            fig.tight_layout()
            fig.savefig(fname, bbox_inches="tight")
            plt.close(fig)

        return trace

    def distance_predictive(self, num, truths=None, fname=None, seed=1234):
        """
        Draw posterior samples for latent distances assuming a distribution.

        Inputs:
            num :: integer
                Number of predictive samples to generate
            truths :: dictionary
                Truths dictionary. If None, do not identify truths
            fname :: string
                If not None, generate longitude-distance diagram
            seed :: integer
                Random seed
        Returns: trace
            trace :: arviz.InferenceData
                Posterior predictive
        """
        # add component likelihood to model
        if "distance" not in self.model.named_vars:
            with self.model:
                if self.distribution == "gaussian":
                    dist_sigma = self.model.shape / pm.math.sin(
                        pm.math.abs(np.deg2rad(self.glat))
                    )
                    _ = Chi("distance", df=3.0, sigma=dist_sigma, dims="data")
                elif self.distribution == "exponential":
                    dist_beta = (
                        pm.math.sin(pm.math.abs(np.deg2rad(self.glat)))
                        / self.model.shape
                    )
                    _ = pm.Gamma("distance", alpha=3.0, beta=dist_beta, dims="data")
                elif self.distribution == "rectangular":
                    dist_scale = self.model.shape / pm.math.sin(
                        pm.math.abs(np.deg2rad(self.glat))
                    )
                    distance3 = pm.Uniform(
                        "distance3", lower=0.0, upper=dist_scale**3.0, dims="data"
                    )
                    _ = pm.Deterministic("distance", distance3 ** (1 / 3), dims="data")
                else:
                    raise NotImplementedError(f"{self.distribution} not available")

        # sample
        with self.model:
            # thin the posterior samples to requested sample size
            total_samples = len(self.trace.posterior.chain) * len(
                self.trace.posterior.draw
            )
            thin = total_samples // num
            trace = pm.sample_posterior_predictive(
                self.trace.sel(draw=slice(None, None, thin)),
                var_names=["distance"],
                random_seed=seed,
            )

        if fname is not None:
            fig, ax = plt.subplots(figsize=(6, 8))
            for chain in trace.posterior_predictive.chain:
                for draw in trace.posterior_predictive.draw:
                    ax.plot(
                        trace.posterior_predictive["distance"].sel(
                            chain=chain, draw=draw
                        ),
                        self.wrap_glong,
                        ".",
                        alpha=0.1,
                        markersize=1,
                    )
            ax.plot(
                truths["distance"],
                self.wrap_glong,
                "r.",
                markersize=1,
                label="Truth",
            )
            ax.set_ylim(-200.0, 200.0)
            ax.set_xlabel(r"Distance (pc)")
            ax.set_ylabel(r"Galactic Longitude (deg)")
            ax.legend(loc="best", fontsize=10)
            fig.tight_layout()
            fig.savefig(fname, bbox_inches="tight")
            plt.close(fig)

        return trace


class MomentRatioModel(BaseModel):
    """
    Use MCMC to infer the ratio of the third to second raw moments of
    the vertical distribution of clouds.
    """

    def __init__(
        self,
        glong,
        glat,
        vlsr,
        prior_mom3_mom2_abs_z_ratio=100.0,
        prior_vlsr_err=10.0,
        prior_outlier_vlsr_sigma=50.0,
    ):
        """
        Initialize a new MomentRatioModel instance.

        Inputs:
            glong :: 1-D array of scalars (deg)
                Galactic longitude of clouds
            glat :: 1-D array of scalars (deg)
                Galactic latitude of clouds
            vlsr :: 1-D array of scalars (km/s)
                LSR velocities of clouds
            prior_mom3_mom2_abs_z_ratio :: scalar (pc)
                Mode of the k=2 gamma distribution prior on the
                moment ratio.
            prior_vlsr_err :: scalar (km/s)
                Width of half-normal distribution prior on vlsr_err
            prior_outlier_vlsr_sigma :: scalar (km/s)
                Width of half-normal distribution prior on outlier_vlsr_sigma

        Returns: model
            model :: MomentRatioModel
                New model instance
        """
        super().__init__(glong, glat, vlsr, prior_outlier_vlsr_sigma)
        self.var_names = [
            "rotcurve",
            "mom3_mom2_abs_z_ratio",
            "vlsr_err",
            "w",
            "outlier_vlsr_sigma",
        ]

        # Define model
        with self.model:
            # moment ratio
            mom3_mom2_abs_z_ratio = pm.Gamma(
                "mom3_mom2_abs_z_ratio",
                alpha=2.0,
                beta=1.0 / prior_mom3_mom2_abs_z_ratio,
            )

            # distance expectation
            mom0_distance = pm.Deterministic(
                "mom0_distance",
                mom3_mom2_abs_z_ratio / pm.math.sin(pm.math.abs(np.deg2rad(glat))),
                dims="data",
            )

            # velocity
            vlsr_err = pm.HalfNormal("vlsr_err", sigma=prior_vlsr_err)
            vlsr_mu = reid19_vlsr(
                glong,
                glat,
                mom0_distance / 1000.0,  # kpc
                R0=self.model.rotcurve[0],
                Usun=self.model.rotcurve[1],
                Vsun=self.model.rotcurve[2],
                Wsun=self.model.rotcurve[3],
                a2=self.model.rotcurve[4],
                a3=self.model.rotcurve[5],
            )
            self.model_vlsr_dist = pm.Normal.dist(mu=vlsr_mu, sigma=vlsr_err)

            # observed mixture
            w = pm.Dirichlet("w", a=np.ones(2))
            _ = pm.Mixture(
                "vlsr",
                w=w,
                comp_dists=[self.model_vlsr_dist, self.outlier_vlsr_dist],
                observed=vlsr,
                dims="data",
            )

    def distance_predictive(
        self, num, distribution="gaussian", truths=None, fname=None, seed=1234
    ):
        """
        Draw posterior samples for latent distances assuming a distribution.

        Inputs:
            num :: integer
                Number of predictive samples to generate
            distribution :: string
                One of "gaussian", "exponential", or "rectangular", the
                assumed vertical distribution of clouds
            truths :: dictionary
                Truths dictionary. If None, do not identify truths
            fname :: string
                If not None, generate longitude-distance diagram
            seed :: integer
                Random seed

        Returns: trace
            trace :: arviz.InferenceData
                Posterior predictive
        """
        self.distribution = distribution

        # add shape parameter to model
        if "shape" not in self.model.named_vars:
            with self.model:
                if distribution == "gaussian":
                    _ = pm.Deterministic(
                        "shape",
                        self.model.mom3_mom2_abs_z_ratio / (2.0 * np.sqrt(2.0 / np.pi)),
                    )
                elif distribution == "exponential":
                    _ = pm.Deterministic(
                        "shape", self.model.mom3_mom2_abs_z_ratio / 3.0
                    )
                elif distribution == "rectangular":
                    _ = pm.Deterministic(
                        "shape", self.model.mom3_mom2_abs_z_ratio * 4.0 / 3.0
                    )
                else:
                    raise NotImplementedError(f"{distribution} not available")

        return super().distance_predictive(num, truths=truths, fname=fname, seed=seed)


class ShapeModel(BaseModel):
    """
    Use MCMC to infer the shape parameter for a given vertical distribution
    of clouds, marginaling over latent distances.
    """

    def __init__(
        self,
        glong,
        glat,
        vlsr,
        distribution="gaussian",
        prior_shape=100.0,
        prior_vlsr_err=10.0,
        prior_outlier_vlsr_sigma=50.0,
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
            distribution :: string
                One of "gaussian", "exponential", or "rectangular", the
                assumed vertical distribution of clouds
            prior_shape :: scalar (pc)
                Mode of the k=2 gamma distribution prior on the
                shape of the vertical distribution of clouds.
                    Gaussian :: standard deviation
                    Exponential :: scale height
                    Rectangular :: half-width
            prior_vlsr_err :: scalar (km/s)
                Width of half-normal distribution prior on vlsr_err
            prior_outlier_vlsr_sigma :: scalar (km/s)
                Width of half-normal distribution prior on outlier_vlsr_sigma

        Returns: model
            model :: MarginalDistanceModel
                New model instance
        """
        super().__init__(glong, glat, vlsr, prior_outlier_vlsr_sigma)
        self.var_names = ["rotcurve", "shape", "vlsr_err", "w", "outlier_vlsr_sigma"]
        self.distribution = distribution

        # Define model
        with self.model:
            # shape parameter
            shape = pm.Gamma("shape", alpha=2.0, beta=1.0 / prior_shape)

            if distribution == "gaussian":
                mom3_mom2_abs_z_ratio = 2.0 * np.sqrt(2.0 / np.pi) * shape
            elif distribution == "exponential":
                mom3_mom2_abs_z_ratio = 3.0 * shape
            elif distribution == "rectangular":
                mom3_mom2_abs_z_ratio = 3.0 * shape / 4.0
            else:
                raise NotImplementedError(f"{distribution} not available")

            # distance expectation value
            mom0_distance = pm.Deterministic(
                "mom0_distance",
                mom3_mom2_abs_z_ratio / pm.math.sin(pm.math.abs(np.deg2rad(glat))),
                dims="data",
            )

            # velocity
            vlsr_err = pm.HalfNormal("vlsr_err", sigma=prior_vlsr_err)
            vlsr_mu = reid19_vlsr(
                glong,
                glat,
                mom0_distance / 1000.0,  # kpc
                R0=self.model.rotcurve[0],
                Usun=self.model.rotcurve[1],
                Vsun=self.model.rotcurve[2],
                Wsun=self.model.rotcurve[3],
                a2=self.model.rotcurve[4],
                a3=self.model.rotcurve[5],
            )
            self.model_vlsr_dist = pm.Normal.dist(mu=vlsr_mu, sigma=vlsr_err)

            # observed mixture
            w = pm.Dirichlet("w", a=np.ones(2))
            _ = pm.Mixture(
                "vlsr",
                w=w,
                comp_dists=[self.model_vlsr_dist, self.outlier_vlsr_dist],
                observed=vlsr,
                dims="data",
            )
