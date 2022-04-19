import numpy as np
from astropy import units as U, constants as C
from kyleaoman_utilities.cosmology.rvir_conventions import Delta_vir
from astropy.cosmology import FlatLambdaCDM


def ax3(arr):
    _ax3 = (np.argwhere(np.array(arr.shape) == 3)).flatten()
    if (_ax3.size != 1) or (arr.ndim != 2):
        raise ValueError('arr must have shape (3, N) or (N, 3) and N!=3.')
    else:
        return _ax3[0]


class BJHalo(object):

    # constants for all halos
    cosmo = FlatLambdaCDM(70 * U.km / U.s / U.Mpc, 0.3, Ob0=.024 / .7 ** 2)
    Mvir0 = 1.4E12 * U.Msun
    # BJ05 quotes Rvir0=282kpc, but this is incompatible with the value
    # derived from the virial condition, and the quoted Vvir=144km/s.
    # Best to just use the calculated value of Rvir0=289kpc.
    # Rvir0 = 282 * U.kpc
    a0 = 1
    # halo incides from Table 1 of B&J2005
    acs = {1: .375, 2: .287, 3: .388, 4: .393, 5: .214, 6: .232, 7: .385,
           8: .205, 9: .187, 10: .229, 11: .230}

    def __init__(self, Nhalo=1):
        self.ac = self.acs[Nhalo]
        self.Rvir0 = self.Rvir(1)
        return

    def c(self, a):
        return 5.1 * a / self.ac

    def Mvir(self, a):
        return self.Mvir0 * np.exp(-2 * self.ac * (self.a0 / a - 1))

    def Rvir(self, a):
        z = 1 / a - 1
        rhoM = self.cosmo.Om(z) * self.cosmo.critical_density(z)
        Dvir = Delta_vir(z, CP=self.cosmo)
        return np.power(
            self.Mvir(a) * 3 / (4 * np.pi) / rhoM / Dvir,
            1 / 3
        ).to(U.kpc)

    def Mhalo(self, a):
        _c = self.c(a)
        return self.Mvir(a) / (np.log(_c + 1) - _c / (_c + 1))

    def rhalo(self, a):
        return self.Rvir(a) / self.c(a)

    def Mdisk(self, a):
        return 1E11 * U.Msun * self.Mvir(a) / self.Mvir0

    def Msphere(self, a):
        return 3.4E10 * U.Msun * self.Mvir(a) / self.Mvir0

    def Rdisk(self, a):
        return 6.5 * U.kpc * self.Rvir(a) / self.Rvir0

    def Zdisk(self, a):
        return .26 * U.kpc * self.Rvir(a) / self.Rvir0

    def rsphere(self, a):
        return .7 * U.kpc * self.Rvir(a) / self.Rvir0

    def Phi_halo(self, r, a=1):
        _Phi_halo = -C.G * self.Mhalo(a=a) / r \
            * np.log(r / self.rhalo(a=a) + 1)
        return _Phi_halo.to(U.km ** 2 * U.s ** -2)

    def Phi_disk(self, R, Z, a=1):
        _Phi_disk = -C.G * self.Mdisk(a=a) / np.sqrt(
            np.power(R, 2) + np.power(
                self.Rdisk(a=a) + np.sqrt(
                    np.power(Z, 2) + np.power(self.Zdisk(a=a), 2)
                ),
                2
            )
        )
        return _Phi_disk.to(U.km ** 2 * U.s ** -2)

    def Phi_sphere(self, r, a=1):
        _Phi_sphere = -C.G * self.Msphere(a=a) / (r + self.rsphere(a=a))
        return _Phi_sphere.to(U.km ** 2 * U.s ** -2)

    def Phi(self, xyz, a=1):
        _ax3 = ax3(xyz)
        r = np.sqrt(np.sum(np.power(xyz, 2), axis=_ax3))
        R = np.sqrt(np.sum(np.power(np.take(xyz, (0, 1), _ax3), 2), axis=_ax3))
        Z = np.take(xyz, (2, ), _ax3).flatten()
        _Phi_halo = self.Phi_halo(r, a=a)
        _Phi_disk = self.Phi_disk(R, Z, a=a)
        _Phi_sphere = self.Phi_sphere(r, a=a)
        return _Phi_halo + _Phi_disk + _Phi_sphere

    def E(self, xyz, vxyz, a=1):
        if xyz.shape != vxyz.shape:
            raise ValueError('xyz and vxyz must have same shape.')
        _ax3 = ax3(xyz)
        _E = 0.5 * np.sum(np.power(vxyz, 2), axis=_ax3) + self.Phi(xyz, a=a)
        return _E.to(U.km **2 * U.s ** -2)

    def Lz(self, xyz, vxyz):
        if xyz.shape != vxyz.shape:
            raise ValueError('xyz and vxyz must have same shape.')
        _ax3 = ax3(xyz)
        r = np.sqrt(np.sum(np.power(xyz, 2), axis=_ax3))
        R = np.sqrt(np.sum(np.power(np.take(xyz, (0, 1), _ax3), 2), axis=_ax3))
        rhat = xyz / np.expand_dims(r, _ax3)
        zhat = np.array([0, 0, 1])
        phihat = np.cross(rhat, zhat, axis=-_ax3)
        vphi = np.sum(phihat * vxyz, axis=_ax3)
        raise RuntimeError('There is a bug here, Lz=r*v, not r**2*v!'
                           ' Need to fix this and re-make plots.')
        _Lz = np.power(R, 2) * vphi
        return _Lz.to(U.kpc ** 2 * U.km * U.s ** -1)
