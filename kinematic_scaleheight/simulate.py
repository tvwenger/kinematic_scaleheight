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
from .rotation import reid19_vlsr


def gen_synthetic_sample(
    n,
    sigma_z,
    vlsr_err=10.0,
    glat_err=0.1,
    d_max=1000.0,
    b_min=10.0,
    b_max=30.0,
    seed=1234,
):
    """
    Generate a sample of synthetic observations. Exclude targets
    within 15 degrees of the Galactic center, 20 degrees of the
    Galactic anti-center, or outside of (b_min, b_max)

    Inputs:
        n :: integer
            Sample size
        sigma_z :: scalar (pc)
            Standard deviation of vertical distribution
        vlsr_err :: scalar (km/s)
            Noise added to LSR velocity.
        glat_err :: scalar (deg)
            Noise added to the Galactic latitude
        d_max :: scalar (pc)
            Maximum mid-plane distance
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
        # Heliocentric midplane distance
        midplane_dist = d_max * np.sqrt(rng.uniform(0, 1, size=10 * n))

        # GC is at (R0, 0, 0)
        new_glong = rng.uniform(0.0, 360.0, size=10 * n)
        X = midplane_dist * np.cos(np.deg2rad(new_glong))
        Y = midplane_dist * np.sin(np.deg2rad(new_glong))

        # Gaussian distribution in Z
        Z = rng.normal(0.0, sigma_z, size=10 * n)

        # true distance
        new_distance = np.sqrt(X**2.0 + Y**2.0 + Z**2.0)

        # Galactic latitude
        new_glat = np.rad2deg(np.arcsin(Z / new_distance))
        new_glat += rng.normal(0.0, glat_err, size=10 * n)

        # LSR velocity from rotation curve
        new_vlsr = reid19_vlsr(
            new_glong,
            new_glat,
            midplane_dist / 1000.0,  # kpc
            Z / 1000.0,  # kpc
            R0=R0,
            a2=a2,
            a3=a3,
            Usun=Usun,
            Vsun=Vsun,
            Wsun=Wsun,
        ).eval()
        new_vlsr += rng.normal(0.0, vlsr_err, size=10 * n)

        # check longitude and latitude limits
        good = (
            (new_glong > 15.0)
            & ((new_glong < 160.0) | (new_glong > 200.0))
            & (new_glong < 345.0)
            & (np.abs(new_glat) > b_min)
            & (np.abs(new_glat) < b_max)
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
        "sigma_z": sigma_z,
        "vlsr_err": vlsr_err,
        "glat_err": glat_err,
        "R0": R0,
        "Usun": Usun,
        "Vsun": Vsun,
        "Wsun": Wsun,
        "a2": a2,
        "a3": a3,
    }

    # trim to size
    return glong[:n], glat[:n], vlsr[:n], truths
