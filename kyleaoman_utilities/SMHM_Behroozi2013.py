import numpy as np
from scipy.optimize import fsolve
import astropy.units as U

def Mstar(Mh, z=0.0):
    try:
        Mh = Mh.to(U.solMass) / U.solMass
        had_units = True
    except AttributeError:
        pass #assume value was given in solar masses
    #note input should be Mvir as defined by Bryan & Norman 1998
    a = 1 / (1 + z)
    nu = np.exp(-4 * np.power(a, 2))
    log10eps = -1.777 + (-.006 * (a - 1) + (-0) * z) * nu + (-0.119) * (a - 1)
    log10M1 = 11.513 + (-1.793 * (a - 1) + (-0.251) * z) * nu
    alpha = -1.412 + (0.731 * (a - 1)) * nu
    delta = 3.508 + (2.608 * (a - 1) + (-.043) * z) * nu
    gamma = 0.316 + (1.319 * (a - 1) + 0.279 * z) * nu

    f = lambda x: -np.log10(np.power(10, alpha * x) + 1) + delta * (np.power(np.log10(1 + np.exp(x)), gamma)) / (1 + np.exp(np.power(10, -x)))

    log10Mstar = log10eps + log10M1 + f(np.log10(Mh) - log10M1) - f(0)

    return np.power(10, log10Mstar) * {True: U.solMass, False: 1.}[had_units]

def Mh(Ms, z=0):
    f = lambda M: np.log10(Mstar(np.power(10, M), z=z)) - np.log10(Ms)
    return np.power(10, fsolve(f, x0=np.ones(Ms.shape)*12))
