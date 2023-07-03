"""
rotation.py
Utilities related to the Reid et al. (2019) Galactic rotation curve.

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

import numpy as np
import pytensor.tensor as pt

#
# Reid+2019 A5 rotation model parameters
#
__R0 = 8166.0  # pc
__Usun = 10.449  # km/s
__Vsun = 12.092  # km/s
__Wsun = 7.729  # km/s
__a2 = 0.977
__a3 = 1.623

#
# IAU defined LSR
#
__Ustd = 10.27  # km/s
__Vstd = 15.32  # km/s
__Wstd = 7.74  # km/s


def reid19_theta(R, R0=__R0, a2=__a2, a3=__a3):
    """
    Calculate the Reid et al. (2019) circular rotation speed at a
    given Galactocentric radius.

    Inputs:
        R :: scalar (kpc)
            Galactocentric radius
        R0 :: scalar (kpc)
            Solar Galactocentric radius
        a2, a3 :: scalar
            Parameters that define rotation curve

    Returns: theta
        theta :: scalar (km/s)
            Circular rotation speed
    """
    rho = R / (a2 * R0)
    lam = (a3 / 1.5) ** 5.0
    loglam = np.log10(lam)
    term1 = 200.0 * lam**0.41
    term2 = np.sqrt(
        0.8 + 0.49 * loglam + 0.75 * np.exp(-0.4 * lam) / (0.47 + 2.25 * lam**0.4)
    )
    term3 = (0.72 + 0.44 * loglam) * 1.97 * rho**1.22 / (rho**2.0 + 0.61) ** 1.43
    term4 = 1.6 * np.exp(-0.4 * lam) * rho**2.0 / (rho**2.0 + 2.25 * lam**0.4)
    theta = term1 / term2 * np.sqrt(term3 + term4)
    return theta


def reid19_vlsr(
    glong,
    glat,
    midplane_dist,
    Z,
    R0=__R0,
    a2=__a2,
    a3=__a3,
    Usun=__Usun,
    Vsun=__Vsun,
    Wsun=__Wsun,
):
    """
    Calculate the Reid et al. (2019) rotation curve LSR velocity
    at a given position.

    Inputs:
        glong, glat :: scalars (deg)
            Galactic longitude and latitude
        midplane_dist :: scalar (kpc)
            Mid-plane distance (dist * cos(glat))
        Z :: scalar (kpc)
            Height above plane (dist * sin(glat))
        R0 :: scalar (kpc)
            Solar Galactocentric radius
        a2, a3 :: scalar
            Parameters that define rotation curve
        Usun, Vsun, Wsun :: scalars (km/s)
            Solar motion relative to the LSR

    Returns: vlsr
        vlsr :: scalar (km/s)
            LSR velocity
    """
    cos_glong = np.cos(np.deg2rad(glong))
    sin_glong = np.sin(np.deg2rad(glong))
    cos_glat = np.cos(np.deg2rad(glat))
    sin_glat = np.sin(np.deg2rad(glat))

    # Barycentric cartesian coordinates. GC is at (R0, 0, 0)
    X = midplane_dist * cos_glong
    Y = midplane_dist * sin_glong
    distance = np.sqrt(midplane_dist**2.0 + Z**2.0)

    # calculate Galactocentric radius via law of cosines
    R = np.sqrt(R0**2.0 + midplane_dist**2.0 - 2.0 * R0 * midplane_dist * cos_glong)

    # calculate Galactocentric azimuth
    az = pt.arctan2(midplane_dist * sin_glong, R0 - midplane_dist * cos_glong)
    sin_az = np.sin(az)
    cos_az = np.cos(az)

    # Circular velocity
    theta = reid19_theta(R, R0=R0, a2=a2, a3=a3)
    theta0 = reid19_theta(R0, R0=R0, a2=a2, a3=a3)

    # Cartesian velocities in Galactocentric frame
    vXg = theta * sin_az
    vYg = theta * cos_az

    # Cartesian velocities in the barycentric frame.
    vXb = vXg - Usun
    vYb = vYg - theta0 - Vsun
    vZb = -Wsun

    # Barycentric velocity
    vbary = (X * vXb + Y * vYb + Z * vZb) / distance

    # IAU-defined LSR
    vlsr = (
        vbary + (__Ustd * cos_glong + __Vstd * sin_glong) * cos_glat + __Wstd * sin_glat
    )
    return vlsr
