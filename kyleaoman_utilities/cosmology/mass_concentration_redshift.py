import numpy as np
from scipy.interpolate import interp1d as interpolate
from scipy.optimize import fsolve
import os.path
from astropy.cosmology import WMAP7, Planck13

prefix = os.path.dirname(__file__) + '/data/'


def _load_tables(CP=WMAP7):
    if CP not in (WMAP7, Planck13):
        raise ValueError('Supported cosmologies: WMAP7, Planck13.')
    name = {WMAP7: 'WMAP7', Planck13: 'Planck13'}[CP]
    sigma0_tab, M_tab_s = \
        np.genfromtxt(prefix + 'sigma0_' + name + '.dat').T
    M_tab_c, c_tab = np.genfromtxt(prefix + 'Mc_z0_' + name + '.dat').T
    M_tab_s = np.power(10, M_tab_s) * 1E10 / CP.h
    M_tab_c = M_tab_c * 1E10 / CP.h
    return sigma0_tab, M_tab_s, M_tab_c, c_tab


def _D(z, CP=WMAP7):
    Ez = np.sqrt(CP.Ode0 + CP.Om0 * np.power(1 + z, 3))
    OmegaL_z = CP.Ode0 / np.power(Ez, 2)
    OmegaM_z = 1 - OmegaL_z
    g0 = 2.5 * CP.Om0 / (
        np.power(CP.Om0, 4. / 7.) -
        CP.Ode0 +
        (1. + .5 * CP.Om0) * (1 + CP.Ode0 / 70.)
    )
    gz = 2.5 * OmegaM_z / (
        np.power(OmegaM_z, 4. / 7.) -
        OmegaL_z +
        (1. + .5 * OmegaM_z) * (1 + OmegaL_z / 70.)
    )
    return gz / (g0 * (1 + z))


def _sigma(M0, CP=WMAP7):
    sigma0_tab, M_tab_s, M_tab_c, c_tab = _load_tables(CP=CP)
    f_sigma = interpolate(M_tab_s, sigma0_tab)
    return f_sigma(M0)


def _zf(M0, CP=WMAP7):
    f = 0.068

    def f_zf(x):
        _D(x, CP=CP) - np.power(
            np.power(_D(0, CP=CP), -1) +
            (.477 / (.15 * np.power(12 * np.pi, 2. / 3.) *
                     np.power(CP.Om0, 0.0055)))
            * np.sqrt(2 * (np.power(_sigma(f * M0), 2) -
                           np.power(_sigma(M0), 2))),
            -1
        )
    return fsolve(f_zf, 2.)


def c(Mz, z=0, CP=WMAP7):
    if z == 0:
        M0 = Mz
    else:
        def f_M0(x):
            _sigma(x, CP=CP) - _sigma(Mz, CP=CP) * _D(z, CP=CP) / _D(0, CP=CP)
        M0 = fsolve(f_M0, Mz)
    sigma0_tab, M_tab_s, M_tab_c, c_tab = _load_tables(CP=CP)
    f_c = interpolate(M_tab_c, c_tab)
    return f_c(M0)
