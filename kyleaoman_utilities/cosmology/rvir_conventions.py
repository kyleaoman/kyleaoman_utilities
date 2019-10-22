import numpy as np
from scipy.optimize import fsolve
from astropy.cosmology import WMAP7, FlatLambdaCDM


def rvir_ratio(DeltaA, rel_toA, DeltaB, rel_toB, CP=WMAP7, c=None):
    if c is None:
        raise ValueError("Provide concentration parameter (kwarg 'c').")
    overdensity_ratio = DeltaA * {'b': CP.Om0, 'c': 1}[rel_toA] / \
        (DeltaB * {'b': CP.Om0, 'c': 1}[rel_toB])

    def f(r_ratio):
        return np.power(r_ratio, 3) - np.power(overdensity_ratio, -1) * \
            (np.log(1 + c) - c / (1 + c)) / (np.log(1 + c / r_ratio) -
                                             c / (r_ratio + c))

    try:
        # these should be broadcastable, or scalar
        guess = np.ones((c * DeltaA * DeltaB).shape)
    except AttributeError:
        guess = 1.0
    return fsolve(f, x0=guess)


def mvir_ratio(DeltaA, rel_toA, DeltaB, rel_toB, CP=WMAP7, c=None):
    overdensity_ratio = DeltaA * {'b': CP.Om0, 'c': 1}[rel_toA] / \
        (DeltaB * {'b': CP.Om0, 'c': 1}[rel_toB])
    r_ratio = rvir_ratio(DeltaA, rel_toA, DeltaB, rel_toB, CP=CP, c=c)
    return overdensity_ratio * np.power(r_ratio, 3)


def Delta_vir(z, CP=WMAP7):
    # Bryan & Norman (1998)
    if not isinstance(CP, FlatLambdaCDM):
        raise ValueError('Delta_vir only valid for flat cosmologies.')
    x = CP.Om0 * np.power(1 + z, 3) * np.power(CP.efunc(z), -2) - 1
    delta_c = 18. * np.power(np.pi, 2) + 82. * x - 39. * np.power(x, 2)
    return delta_c / CP.Om(z)  # delta_b
