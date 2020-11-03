import numpy as np
from scipy.optimize import fsolve
import astropy.units as U


def Mstar(Mh, z=0.0, scatter=True):

    if z > 0.1:
        raise ValueError('Parameters only given for low redshift.')
    # note input should be Mvir as defined by Bryan & Norman 1998
    try:
        Mh = Mh.to(U.solMass) / U.solMass
    except AttributeError:
        pass  # assume value was given in Msun

    if scatter:
        log10M1 = 11.513
        log10eps = -1.685
        alpha = -1.740
        delta = 4.335
        gamma = 0.531
    else:
        log10M1 = 11.43
        log10eps = -1.663
        alpha = -1.750
        delta = 4.290
        gamma = 0.595

    def f(x):
        return -np.log10(np.power(10, alpha * x) + 1) \
            + delta * (np.power(np.log10(1 + np.exp(x)), gamma)) \
            / (1 + np.exp(np.power(10, -x)))

    log10Mstar = log10eps + log10M1 + f(np.log10(Mh) - log10M1) - f(0)

    return np.power(10, log10Mstar) * U.solMass


def Mh(Ms, z=0):
    try:
        Ms = Ms.to(U.solMass) / U.solMass
    except AttributeError:
        pass  # assume value was given in Msun

    def f(M):
        return np.log10(Mstar(np.power(10, M), z=z).value) - np.log10(Ms)

    return np.power(10, fsolve(f, x0=np.ones(Ms.shape)*12)) * U.solMass
