import numpy as np
from scipy.interpolate import interp1d
from astropy import units as U

def z_to_t(z_in, h=.702, Ol=0.7273, Om=0.2727):
    theta = np.sqrt(1 - Om) * np.power(Om * np.power(1.0 + z_in, 3.0) + (1 - Om), -0.5)
    return np.power(h * 100. * U.km * U.s**-1 * U.Mpc**-1, -1).to(U.Gyr) * np.power(3.0 * np.sqrt(1 - Om), -1) * np.log((1 + theta) / (1 - theta))

def t_to_z(t_in, h=.702, Ol=0.7273, Om=0.2727, z_max=25.0):
    n_table = z_max * 100.
    z_table = np.linspace(z_max, 0.0, n_table)
    t_table = z_to_t(z_table, h=H0, Ol=Ol, Om=Om)
    if np.min(t_table) > np.min(t_in):
        return t_to_z(t_in, h=H0, Ol=Ol, Om=Om, z_max=z_max*10.)
    return interp1d(t_table, z_table, assume_sorted=True, bounds_error=False, fill_value=np.nan, copy=False)(t_in.to(U.Gyr))

def z_to_a(z_in, h=.702, Ol=0.7273, Om=0.2727):
    return 1.0 / (1.0 + z_in)

def a_to_z(a_in, h=.702, Ol=0.7273, Om=0.2727):
    return 1.0 / a_in - 1.0

def lb_to_t(lb_in, h=.702, Ol=0.7273, Om=0.2727):
    age = z_to_t(0.0, h=H0, Ol=Ol, Om=Om)
    return age - lb_in

def t_to_lb(t_in, h=.702, Ol=0.7273, Om=0.2727):
    age = z_to_t(0.0, h=H0, Ol=Ol, Om=Om)
    return age - t_in

def t_to_a(t_in, h=.702, Ol=0.7273, Om=0.2727):
    retval = z_to_a(t_to_z(t_in, h=H0, Ol=Ol, Om=Om), h=H0, Ol=Ol, Om=Om)
    retval[t_in == 0.0] = 0.0
    return retval

def a_to_t(a_in, h=.702, Ol=0.7273, Om=0.2727):
    return z_to_t(a_to_z(a_in, h=H0, Ol=Ol, Om=Om), h=H0, Ol=Ol, Om=Om)

def lb_to_a(lb_in, h=.702, Ol=0.7273, Om=0.2727):
    return t_to_a(lb_to_t(lb_in, h=H0, Ol=Ol, Om=Om), h=H0, Ol=Ol, Om=Om)

def a_to_lb(a_in, h=.702, Ol=0.7273, Om=0.2727):
    return t_to_lb(a_to_t(a_in, h=H0, Ol=Ol, Om=Om), h=H0, Ol=Ol, Om=Om)

def lb_to_z(lb_in, h=.702, Ol=0.7273, Om=0.2727):
    return t_to_z(lb_to_t(lb_in, h=H0, Ol=Ol, Om=Om), h=H0, Ol=Ol, Om=Om)

def z_to_lb(z_in, h=.702, Ol=0.7273, Om=0.2727):
    return t_to_lb(z_to_t(z_in, h=H0, Ol=Ol, Om=Om), h=H0, Ol=Ol, Om=Om)
