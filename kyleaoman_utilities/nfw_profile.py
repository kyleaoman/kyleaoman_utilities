import numpy as np
from astropy.cosmology import WMAP7
from kyleaoman_utilities.cosmology.mass_concentration_redshift import c
import astropy.constants as C
import astropy.units as U

# UNITS ASSUMED: kpc, Msun, km/s; also works if other units provided explicitly

# NFW PROFILE PARAMETERS DERIVED FROM MASS (M below is M200c)


class NFW(object):

    def __init__(self, M200, z=0., CP=WMAP7, c200=None):
        try:
            self.M200 = M200.to(U.solMass)
        except AttributeError:
            self.M200 = M200 * U.solMass  # assume solar masses
        self.CP = CP
        self.z = z
        if c200 is None:
            self.c200 = c(self.M200, z=self.z, CP=self.CP)
        else:
            self.c200 = c200
        self.Rs = 1.63E-2 * U.kpc * U.solMass ** (-1. / 3.) * \
            (np.power(self.CP.h, -2. / 3.) / self.c200) \
            * np.power(self.M200, 1. / 3.)
        self.rho0 = np.power(self.CP.h, 2) * 278. * (200. / 3.) \
            * np.power(self.c200, 3) / \
            (np.log(1 + self.c200) - self.c200 /
             (1 + self.c200)) * U.solMass * U.kpc ** -3
        self.rmax = 2.1626 * self.Rs
        self.vmax = self.vc(self.rmax)
        self.R200 = self.Rs * self.c200
        return

    def vc(self, r):
        retval = np.power(
            4. * np.pi * C.G * self.rho0 * np.power(self.Rs, 3) /
            r * (np.log(1 + r / self.Rs) - 1. / (1 + self.Rs / r)),
            .5)
        return retval.to(U.km * U.s ** -1)

    def Mltr(self, r):
        retval = 4. * np.pi * self.rho0 * np.power(self.Rs, 3) * \
            (np.log(1 + r / self.Rs) - 1. / (self.Rs / r + 1))
        return retval.to(U.solMass)

    def rho(self, r):
        retval = self.rho0 / ((r / self.Rs) * np.power(1 + r / self.Rs, 2))
        return retval.to(U.solMass * U.kpc ** -3)

    def dlogrho(self, r):
        return -(1 + 2 / (self.Rs / r + 1)).to(U.dimensionless_unscaled)

    def accel(self, r):
        retval = C.G * self.Mltr(r) / np.power(r, 2)
        return retval.to(U.km * U.s ** -2)

    def pot(self, r, M, pm=0., h=0.704):
        retval = - 4. * np.pi * C.G * self.rho0 * np.power(self.Rs, 2) * \
            np.log(1 + r / self.Rs) / (r / self.Rs)
        return retval.to(U.kpc * U.km / U.s ** -2)
