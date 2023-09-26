"""
simulate.py
Generate a synthetic sample of clouds to test the methods.

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
from scipy.stats import multivariate_normal

from kinematic_scaleheight.rotation import reid19_vlsr


def gen_synthetic_sample(
    n,
    distribution="gaussian",
    shape=100.0,
    vlsr_err=10.0,
    d_max=1000.0,
    b_min=10.0,
    b_max=30.0,
    seed=1234,
):
    """
    Generate a sample of synthetic observations between b_min and b_max.

    Inputs:
        n :: integer
            Sample size
        distribution :: string
            One of "gaussian", "exponential", or "rectangular", the
            vertical distribution of clouds
        shape :: scalar (pc)
            The shape parameter of the distribution.
            Gaussian :: standard deviation
            Exponential :: scale height
            Rectangular :: half-width
        vlsr_err :: scalar (km/s)
            Noise added to LSR velocity.
        d_max :: scalar (pc)
            Maximum distance
        b_min, b_max :: scalars (deg)
            Minimum and maximum absolute Galactic latitude
        seed :: integer
            Random seed

    Returns: (glong, glat, vlsr, truths)
        glong, glat :: scalars (deg)
            Galactic longitude and latitude
        vlsr :: scalar (km/s)
            LSR velocity
        truths :: dictionary
            Dictionary containing truths values
    """
    rng = np.random.default_rng(seed)

    # Load Reid et al. (2019) kernel density estimate mean
    # and covariance for the rotation curve parameters
    fname = os.path.join(os.path.dirname(__file__), "data/reid19_mv.pkl")
    with open(fname, "rb") as f:
        reid19_mean, reid19_cov = pickle.load(f)
    reid19_mv = multivariate_normal(mean=reid19_mean, cov=reid19_cov)

    # Generate a single sample of rotation curve parameters
    R0, Usun, Vsun, Wsun, a2, a3 = reid19_mv.rvs(random_state=rng)

    # storage for data
    glong = np.empty(0)
    glat = np.empty(0)
    vlsr = np.empty(0)
    distance = np.empty(0)

    while len(glong) < n:
        # Heliocentric position. Draw 10x sample size
        # since we will have to throw away some
        X = rng.uniform(-d_max, d_max, size=10 * n)
        Y = rng.uniform(-d_max, d_max, size=10 * n)
        if distribution == "gaussian":
            Z = rng.normal(0.0, shape, size=10 * n)
            mom1_abs_z = np.sqrt(2.0 / np.pi) * shape
            mom3_mom2_abs_z_ratio = 2.0 * np.sqrt(2.0 / np.pi) * shape
        elif distribution == "exponential":
            sign = rng.integers(0, 1, endpoint=True, size=10 * n) * 2 - 1
            Z = sign * rng.exponential(shape, size=10 * n)
            mom1_abs_z = shape
            mom3_mom2_abs_z_ratio = 3.0 * shape
        elif distribution == "rectangular":
            Z = rng.uniform(-shape, shape, size=10 * n)
            mom1_abs_z = shape / 2.0
            mom3_mom2_abs_z_ratio = 3.0 / 4.0 * shape
        else:
            raise NotImplementedError(f"{distribution} not available")

        # Heliocentric distance
        new_distance = np.sqrt(X**2.0 + Y**2.0 + Z**2.0)

        # Galactic coordinates
        new_glong = np.rad2deg(np.arctan2(Y, X)) % 360.0
        new_glat = np.rad2deg(np.arcsin(Z / new_distance))

        # LSR velocity from rotation curve
        new_vlsr = reid19_vlsr(
            new_glong,
            new_glat,
            new_distance / 1000.0,  # kpc
            R0=R0,
            a2=a2,
            a3=a3,
            Usun=Usun,
            Vsun=Vsun,
            Wsun=Wsun,
        )
        new_vlsr += rng.normal(0.0, vlsr_err, size=new_vlsr.size)

        # check longitude and latitude limits
        good = (
            (np.abs(new_glat) > b_min)
            & (np.abs(new_glat) < b_max)
            & (new_distance < d_max)
        )

        # save
        glong = np.concatenate((glong, new_glong[good]))
        glat = np.concatenate((glat, new_glat[good]))
        vlsr = np.concatenate((vlsr, new_vlsr[good]))
        distance = np.concatenate((distance, new_distance[good]))

    # trim to size
    glong = glong[:n]
    glat = glat[:n]
    vlsr = vlsr[:n]
    distance = distance[:n]

    # store truths separately
    truths = {
        "distance": distance,
        "distribution": distribution,
        "shape": shape,
        "mom1_abs_z": mom1_abs_z,
        "mom3_mom2_abs_z_ratio": mom3_mom2_abs_z_ratio,
        "vlsr_err": vlsr_err,
        "R0": R0,
        "Usun": Usun,
        "Vsun": Vsun,
        "Wsun": Wsun,
        "a2": a2,
        "a3": a3,
    }

    # trim to size
    return glong[:n], glat[:n], vlsr[:n], truths
