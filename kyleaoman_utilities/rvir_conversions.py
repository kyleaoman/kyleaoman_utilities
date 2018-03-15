import numpy as np
from scipy.optimize import fsolve

def rvir_ratio(DeltaA, DeltaB, rel_toA, rel_toB, OmegaM):
    c = ??
    overdensity_ratio = DeltaA * {'b': OmegaM, 'c': 1}[rel_toA] / (DeltaB * {'b': OmegaM, 'c': 1}[rel_toB])
    f = lambda r_ratio: np.power(r_ratio, 3) - np.power(overdensity_ratio, -1) * (np.log(1 + c) - c / (1 + c)) / (np.log(1 + c / r_ratio) - c / (r_ratio + c))
    return fsolve(f, x0=1.0)

def mvir_ratio(DeltaA, DeltaB, rel_toA, rel_toB, OmegaM):
    overdensity_ratio = DeltaA * {'b': OmegaM, 'c': 1}[rel_toA] / (DeltaB * {'b': OmegaM, 'c': 1}[rel_toB])
    r_ratio = rvir_ratio(DeltaA, DeltaB, rel_toA, rel_toB, OmegaM)
    return overdensity_ratio * np.power(r_ratio, 3)

