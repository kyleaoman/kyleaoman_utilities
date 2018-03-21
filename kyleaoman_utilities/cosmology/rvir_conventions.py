import numpy as np
from scipy.optimize import fsolve
from .params import WMAP7
from .mass_concentration_redshift import c as concentration

def rvir_ratio(DeltaA, rel_toA, DeltaB, rel_toB, CP=WMAP7(), c=None):
    if c is None:
        raise ValueError("Provide concentration parameter (kwarg 'c').")
    overdensity_ratio = DeltaA * {'b': CP.Om, 'c': 1}[rel_toA] / (DeltaB * {'b': CP.Om, 'c': 1}[rel_toB])
    f = lambda r_ratio: np.power(r_ratio, 3) - np.power(overdensity_ratio, -1) * (np.log(1 + c) - c / (1 + c)) / (np.log(1 + c / r_ratio) - c / (r_ratio + c))
    return fsolve(f, x0=1.0)

def mvir_ratio(DeltaA, rel_toA, DeltaB, rel_toB, CP=WMAP7(), c=None):
    overdensity_ratio = DeltaA * {'b': CP.Om, 'c': 1}[rel_toA] / (DeltaB * {'b': CP.Om, 'c': 1}[rel_toB])
    r_ratio = rvir_ratio(DeltaA, rel_toA, DeltaB, rel_toB, CP=CP, c=c)
    return overdensity_ratio * np.power(r_ratio, 3)

def Delta_vir(z, CP=WMAP7()):
    #Bryan & Norman (1998)
    x = -np.power(1 + CP.Om / CP.Ol * np.power(1 + z, 3), -1)
    delta_c = 18. * np.power(np.pi, 2) + 82. * x - 39. * np.power(x, 2)
    return delta_c / CP.Om #delta_b
