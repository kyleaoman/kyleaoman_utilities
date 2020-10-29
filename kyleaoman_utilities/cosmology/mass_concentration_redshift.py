import numpy as np
from scipy.interpolate import interp1d as interpolate
from scipy.optimize import fsolve
import os.path
from astropy.cosmology import WMAP7, Planck13

prefix = os.path.dirname(__file__) + '/data/'


class c(object):

    def __init__(self, CP=WMAP7, tableCP=None):
        self.CP = CP
        if tableCP is None:
            self.tableCP = self.CP

    def _load_tables(self, tableCP=WMAP7):
        if self.tableCP not in (WMAP7, Planck13):
            raise ValueError('Supported cosmologies: WMAP7, Planck13.')
        name = {WMAP7: 'WMAP7', Planck13: 'Planck13'}[self.tableCP]
        sigma0_tab, M_tab_s = \
            np.genfromtxt(prefix + 'sigma0_' + name + '.dat').T
        M_tab_c, c_tab = np.genfromtxt(prefix + 'Mc_z0_' + name + '.dat').T
        M_tab_s = np.power(10, M_tab_s) * 1E10 / self.CP.h
        M_tab_c = M_tab_c * 1E10 / self.CP.h
        self.sigma0_tab = sigma0_tab
        self.M_tab_s = M_tab_s
        self.M_tab_c = M_tab_c
        self.c_tab = c_tab
        return

    def _D(self, z):
        Ez = np.sqrt(self.CP.Ode0 + self.CP.Om0 * np.power(1 + z, 3))
        OmegaL_z = self.CP.Ode0 / np.power(Ez, 2)
        OmegaM_z = 1 - OmegaL_z
        g0 = 2.5 * self.CP.Om0 / (
            np.power(self.CP.Om0, 4. / 7.) -
            self.CP.Ode0 +
            (1. + .5 * self.CP.Om0) * (1 + self.CP.Ode0 / 70.)
        )
        gz = 2.5 * OmegaM_z / (
            np.power(OmegaM_z, 4. / 7.) -
            OmegaL_z +
            (1. + .5 * OmegaM_z) * (1 + OmegaL_z / 70.)
        )
        return gz / (g0 * (1 + z))

    def _sigma(self, M0):
        f_sigma = interpolate(self.M_tab_s, self.sigma0_tab)
        return f_sigma(M0)

    def _zf(self, M0):
        f = 0.068

        def f_zf(x):
            self._D(x) - np.power(
                np.power(self._D(0), -1) +
                (.477 / (.15 * np.power(12 * np.pi, 2. / 3.) *
                         np.power(self.CP.Om0, 0.0055)))
                * np.sqrt(2 * (np.power(self._sigma(f * M0), 2) -
                               np.power(self._sigma(M0), 2))),
                -1
            )
        return fsolve(f_zf, 2.)

    def __call__(self, Mz, z=0):
        if z == 0:
            M0 = Mz
        else:
            def f_M0(x):
                self._sigma(x) - self._sigma(Mz) * self._D(z) / self._D(0)
            M0 = fsolve(f_M0, Mz)
        f_c = interpolate(self.M_tab_c, self.c_tab)
        return f_c(M0)
