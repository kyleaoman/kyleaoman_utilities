"""concentration-mass-redshift relations and mean mass accretion histories in the Planck cosmology. All input masses should be passed as Msun (no h)."""

import numpy as np
from scipy.interpolate import interp1d as interpolate
from scipy.optimize import fsolve

import os
try:
    prefix = os.path.dirname(__file__)+'/data/'
except NameError:
    prefix = './data/'

cosmology = {}

def set_cosmology(cosmo='Planck'):
    if cosmo == 'Planck':
        cosmology['OmegaL'] = 0.693
        cosmology['OmegaM'] = 0.307
        cosmology['Omegab'] = 0.04825
        cosmology['h'] = 0.6777
        cosmology['sigma8'] = 0.8288
        cosmology['ns'] = 0.9611
        sigma0_file =  prefix+"sigma0_Planck.dat"
        Mc_file = prefix+"Mc_z0_Planck.dat"

    elif cosmo == 'WMAP7':
        cosmology['OmegaL'] = 0.729
        cosmology['OmegaM'] = 0.2707
        cosmology['Omegab'] = 0.0451
        cosmology['h'] = 0.703
        cosmology['sigma8'] = 0.809
        cosmology['ns'] = 0.966
        sigma0_file = prefix+"sigma0_WMAP7.dat"
        Mc_file = prefix+"Mc_z0_WMAP7.dat"

    sigma0_tab,M_tab_s = np.genfromtxt(sigma0_file).T
    M_tab_s = np.power(10,M_tab_s) * 1E10 / cosmology['h']
    M_tab_c,c_tab = np.genfromtxt(Mc_file).T
    M_tab_c = M_tab_c * 1E10 / cosmology['h']
    cosmology['M_tab_s'] = M_tab_s
    cosmology['sigma0_tab'] = sigma0_tab
    cosmology['M_tab_c'] = M_tab_c
    cosmology['c_tab'] = c_tab
            

def D(z):
    try:
        Ez = np.sqrt(cosmology['OmegaL'] + cosmology['OmegaM'] * np.power(1 + z, 3))
        OmegaL_z = cosmology['OmegaL']/np.power(Ez, 2)
        OmegaM_z = 1 - OmegaL_z
        g0 = 2.5 * cosmology['OmegaM'] / (np.power(cosmology['OmegaM'], 4. / 7.) - cosmology['OmegaL'] + (1. + .5 * cosmology['OmegaM']) * (1 + cosmology['OmegaL'] / 70.))
        gz = 2.5 * OmegaM_z / (np.power(OmegaM_z, 4. / 7.) - OmegaL_z + (1. + .5 * OmegaM_z) * (1 + OmegaL_z / 70.))
        return gz / (g0 * (1 + z))
    except KeyError:
        set_cosmology()
        return D(z)

def zf(M0):
    try:
        f = 0.068
        f_zf = lambda x: D(x) - np.power(np.power(D(0), -1) + (.477 / (.15 * np.power(12 * np.pi, 2. / 3.) * np.power(cosmology['OmegaM'], 0.0055))) * np.sqrt(2 * (np.power(sigma(f * M0), 2) - np.power(sigma(M0), 2))), -1)
        return fsolve(f_zf, 2.)
    except KeyError:
        set_cosmology()
        return zf(M0)

def chi(M0):
    try:
        return 1.211 + 1.858 * np.log10(1 + zf(M0)) + 0.308 * np.power(cosmology['OmegaL'], 2) - 0.032 * np.log10(M0 * cosmology['h'] / 1E11)
    except KeyError:
        set_cosmology()
        return chi(M0)

def sigma(M0):
    try:
        f_sigma = interpolate(cosmology['M_tab_s'], cosmology['sigma0_tab'])
        return f_sigma(M0)
    except KeyError:
        set_cosmology()
        return sigma(M0)

def c(Mz, z=0):
    try:
        if z == 0:
            M0 = Mz
        else:
            f_M0 = lambda x: sigma(x) - sigma(Mz) * D(z) / D(0)
            M0 = fsolve(f_M0, Mz)
        f_c = interpolate(cosmology['M_tab_c'], cosmology['c_tab'])
        return f_c(M0)
    except KeyError:
        set_cosmology()
        return c(Mz, z=0)
