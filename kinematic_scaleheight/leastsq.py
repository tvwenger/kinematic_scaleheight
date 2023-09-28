"""
leastsq.py
Use a least-squares method to estimate the vertical distribution of a population
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

Trey Wenger - September 2023
"""

import numpy as np
from scipy import optimize

from kinematic_scaleheight.rotation import reid19_vlsr


def calc_vlsr(glong, glat, distance, Usun, Vsun, Wsun, glong0, oortA=15.3):
    """
    Use Oort's A constant to estimate LSR velocity in the
    solar neighborhood.

    Inputs:
        glong :: 1-D array of scalars
            Galactic longitude (deg)
        glat :: 1-D array of scalars
            Galatic latitude (deg)
        distance :: 1-D array of scalars
            Distance (kpc)
        Usun, Vsun, Wsun :: scalars
            Solar motion w.r.t. LSR (km/s)
        glong0 :: scalar (deg)
            Nodal deviation
        oortA :: scalar
            Oort's A constant (km/s/kpc)

    Returns: vlsr
        vlsr :: 1-D array of scalars
            LSR velocity (km/s)
    """
    cos_glong = np.cos(np.deg2rad(glong))
    sin_glong = np.sin(np.deg2rad(glong))
    sin_2glong_offset = np.sin(2.0 * np.deg2rad(glong - glong0))
    cos_glat = np.cos(np.deg2rad(glat))
    sin_glat = np.sin(np.deg2rad(glat))
    vlsr = distance * oortA * sin_2glong_offset * cos_glat**2.0
    vlsr += (Usun * cos_glong + Vsun * sin_glong) * cos_glat + Wsun * sin_glat
    return vlsr


def crovisier(glong, glat, vlsr, e_vlsr, oortA=15.3):
    """
    Estimate the first raw moment of the |z| distribution of some clouds using
    the least-squares method of Crovisier (1978).

    Inputs:
        glong :: 1-D array of scalars
            Galactic longitude (deg)
        glat :: 1-D array of scalars
            Galatic latitude (deg)
        vlsr, e_vlsr :: 1-D arrays of scalars
            LSR velocity and uncertainties (km/s)
        oortA :: scalar
            Oort's A constant (km/s/kpc)

    Returns: params, errors, vlsr_rms
        params :: 1-D array of scalars
            The least-squares optimal values for
                mom1_abs_z :: first raw moment of the |z| distribution (pc)
                Usun, Vsun, Wsun :: solar motion w.r.t. LSR (km/s)
                glong0 :: Nodal deviation (deg)
        errors :: 1-D array of scalars
            Standard deviations
        vlsr_rms :: scalar
            The rms LSR velocity error
    """

    # Cost function
    def loss(params):
        mom1_abs_z, Usun, Vsun, Wsun, glong0 = params
        mom1_distance = mom1_abs_z / np.sin(np.abs(np.deg2rad(glat)))
        model_vlsr = calc_vlsr(
            glong, glat, mom1_distance / 1000.0, Usun, Vsun, Wsun, glong0, oortA=oortA
        )
        return (vlsr - model_vlsr) / e_vlsr

    # optimize
    x0 = (100.0, 0.0, 0.0, 0.0, 0.0)
    params, pcov, *_ = optimize.leastsq(loss, x0=x0, full_output=True)
    errors = np.sqrt(np.diag(pcov))

    mom1_abs_z, Usun, Vsun, Wsun, glong0 = params
    mom1_distance = mom1_abs_z / np.sin(np.abs(np.deg2rad(glat)))
    model_vlsr = calc_vlsr(
        glong, glat, mom1_distance / 1000.0, Usun, Vsun, Wsun, glong0, oortA=oortA
    )
    vlsr_rms = np.sqrt(np.mean((vlsr - model_vlsr) ** 2.0))

    return params, errors, vlsr_rms


def leastsq(
    glong,
    glat,
    vlsr,
    e_vlsr,
    R0=8.1660,
    a2=0.977,
    a3=1.623,
):
    """
    Estimate the ratio of the third raw moment to the second raw moment of the
    |z| distribution of some clouds using a least-squares method.

    Inputs:
        glong :: 1-D array of scalars
            Galactic longitude (deg)
        glat :: 1-D array of scalars
            Galatic latitude (deg)
        vlsr, e_vlsr :: 1-D arrays of scalars
            LSR velocity and uncertainties (km/s)
        R0 :: scalar (kpc)
            Solar Galactocentric radius
        a2, a3 :: scalar
            Parameters that define rotation curve

    Returns: params, errors, vlsr_rms
        params :: 1-D array of scalars
            The least-squares optimal values for
                mom3_mom2_abs_z_ratio :: ratio of third to second raw moments of the |z| distribution (pc)
                Usun, Vsun, Wsun :: solar motion w.r.t. LSR (km/s)
        errors :: 1-D array of scalars
            Standard deviations
        vlsr_rms :: scalar
            The rms LSR velocity error
    """

    # Cost function
    def loss(params):
        mom3_mom2_abs_z_ratio, Usun, Vsun, Wsun = params
        mom1_distance = mom3_mom2_abs_z_ratio / np.sin(np.abs(np.deg2rad(glat)))
        model_vlsr = reid19_vlsr(
            glong,
            glat,
            mom1_distance / 1000.0,
            R0=R0,
            a2=a2,
            a3=a3,
            Usun=Usun,
            Vsun=Vsun,
            Wsun=Wsun,
        )
        return (vlsr - model_vlsr) / e_vlsr

    # optimize
    x0 = (100.0, 0.0, 0.0, 0.0)
    params, pcov, *_ = optimize.leastsq(loss, x0=x0, full_output=True)
    errors = np.sqrt(np.diag(pcov))

    mom3_mom2_abs_z_ratio, Usun, Vsun, Wsun = params
    mom1_distance = mom3_mom2_abs_z_ratio / np.sin(np.abs(np.deg2rad(glat)))
    model_vlsr = reid19_vlsr(
        glong,
        glat,
        mom1_distance / 1000.0,
        R0=R0,
        a2=a2,
        a3=a3,
        Usun=Usun,
        Vsun=Vsun,
        Wsun=Wsun,
    )
    vlsr_rms = np.sqrt(np.mean((vlsr - model_vlsr) ** 2.0))

    return params, errors, vlsr_rms
