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
from scipy.optimize import leastsq


def calc_vlsr(glong, glat, distance, Usun, Vsun, Wsun, oortA=15.3):
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
        oortA :: scalar
            Oort's A constant (km/s/kpc)

    Returns: vlsr
        vlsr :: 1-D array of scalars
            LSR velocity (km/s)
    """
    cos_glong = np.cos(np.deg2rad(glong))
    sin_glong = np.sin(np.deg2rad(glong))
    sin_2glong = np.sin(2.0 * np.deg2rad(glong))
    cos_glat = np.cos(np.deg2rad(glat))
    sin_glat = np.sin(np.deg2rad(glat))
    vlsr = distance * oortA * sin_2glong * cos_glat**2.0
    vlsr += (Usun * cos_glong + Vsun * sin_glong) * cos_glat + Wsun * sin_glat
    return vlsr


def crovisier(glong, glat, vlsr, oortA=15.3, corrected=False):
    """
    Estimate the first moment of the |z| distribution of some clouds using
    the least-squares method of Crovisier (1978).

    Inputs:
        glong :: 1-D array of scalars
            Galactic longitude (deg)
        glat :: 1-D array of scalars
            Galatic latitude (deg)
        vlsr :: 1-D array of scalars
            LSR velocity (km/s)
        oortA :: scalar
            Oort's A constant (km/s/kpc)
        corrected :: boolean
            If True, apply correction to mom1_distance

    Returns: params, errors, vlsr_rms
        params :: 1-D array of scalars
            The least-squares optimal values for
                mom1_abs_z :: first moment of the |z| distribution (pc)
                Usun, Vsun, Wsun :: solar motion w.r.t. LSR (km/s)
        errors :: 1-D array of scalars
            Standard deviations
        vlsr_rms :: scalar
            The rms LSR velocity error
    """

    # Cost function
    def loss(params):
        sigma_z, Usun, Vsun, Wsun = params
        mom1_distance = (
            np.sqrt(2.0 / np.pi) * sigma_z / np.sin(np.abs(np.deg2rad(glat)))
        )
        if corrected:
            mom1_distance *= 2.0
        mom1_vlsr = calc_vlsr(
            glong, glat, mom1_distance / 1000.0, Usun, Vsun, Wsun, oortA=oortA
        )
        return vlsr - mom1_vlsr

    # optimize
    x0 = (100.0, 0.0, 0.0, 0.0)
    params, pcov, *_ = leastsq(loss, x0=x0, full_output=True)
    errors = np.sqrt(np.diag(pcov))

    mom1_abs_z, Usun, Vsun, Wsun = params
    mom1_distance = mom1_abs_z / np.sin(np.abs(np.deg2rad(glat)))
    if corrected:
        mom1_distance *= 2.0
    model_vlsr = calc_vlsr(
        glong, glat, mom1_distance / 1000.0, Usun, Vsun, Wsun, oortA=oortA
    )
    vlsr_rms = np.sqrt(np.mean((vlsr - model_vlsr) ** 2.0))

    return params, errors, vlsr_rms
