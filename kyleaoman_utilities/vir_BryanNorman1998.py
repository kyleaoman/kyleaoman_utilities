import numpy as np

#Bryan & Norman (1998)

def Delta_vir(z, Om=.2727, Ol=.7273):
    x = -np.power(1 + Om / Ol * np.power(1 + z, 3), -1)
    delta_c = 18 * np.power(np.pi, 2) + 82 * x - 39 * np.power(x, 2)
    return delta_c / Om #delta_b
