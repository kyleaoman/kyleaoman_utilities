from simobj import SimObj
import astropy.units as U
from astropy.coordinates import CartesianRepresentation, \
    SphericalRepresentation, CylindricalRepresentation, \
    CartesianDifferential, SphericalDifferential, CylindricalDifferential
from itertools import product

T = ['g', 'dm', 'b2', 'b3', 's', 'bh']
Ccart = ['x', 'y', 'z']
Ccyl = ['R', 'phi']
Csph = ['r', 'lon', 'lat']


def spec_t(f, t):
    return lambda: f(t)


def spec_c(f, c):
    return lambda t: f(t, c)


class EnhancedSO(SimObj):

    def __init__(
            self,
            obj_id=None,
            snap_id=None,
            mask_type=None,
            mask_args=None,
            mask_kwargs=None,
            configfile=None,
            simfiles_configfile=None,
            simfiles_instance=None,
            verbose=False,
            ncpu=2

    ):
        self._init_enhancers()
        super().__init__(
            obj_id=obj_id,
            snap_id=snap_id,
            mask_type=mask_type,
            mask_args=mask_args,
            mask_kwargs=mask_kwargs,
            configfile=configfile,
            simfiles_configfile=simfiles_configfile,
            simfiles_instance=simfiles_instance,
            verbose=verbose,
            ncpu=ncpu
        )

        return

    def __getattr__(self, key):
        ga = object.__getattribute__
        if key in ga(self, '_enhancers').keys():
            return ga(self, '_enhancers')[key]()
        else:
            try:
                return ga(self, key)
            except AttributeError:
                return super().__getattr__(key)

    def __getitem__(self, key):
        if key in self._enhancers.keys():
            return self._enhancers[key]()
        else:
            return super().__getitem__(key)

    def _init_enhancers(self):
        object.__setattr__(self, '_enhancers', dict())
        for C, F, vF in [
                (Ccart, self._xyz, self._vxyz),
                (Csph, self._sph, self._vsph),
                (Ccyl, self._cyl, self._vcyl)
        ]:
            self._enhancers.update({
                c + '_' + t:
                spec_t(spec_c(F, c), t)
                for t, c in product(T, C)
            })
            self._enhancers.update({
                'v' + c + '_' + t:
                spec_t(spec_c(vF, c), t)
                for t, c in product(T, C)
            })
        return

    def _CR(self, t, diff=False):
        if diff:
            return CartesianRepresentation(
                U.Quantity(self['xyz_' + t], dtype='f8'),
                xyz_axis=1,
                differentials={
                    's': CartesianDifferential(
                        U.Quantity(self['vxyz_' + t], dtype='f8'),
                        xyz_axis=1
                    )
                }
            )
        else:
            return CartesianRepresentation(
                U.Quantity(self['xyz_' + t], dtype='f8'),
                xyz_axis=1
            )

    def _CyR(self, t, diff=False):
        if diff:
            return self._CR(t, diff=diff).represent_as(
                CylindricalRepresentation,
                CylindricalDifferential
            )
        else:
            return self._CR(t, diff=diff).represent_as(
                CylindricalRepresentation
            )

    def _SR(self, t, diff=False):
        if diff:
            return self._CR(t, diff=diff).represent_as(
                SphericalRepresentation,
                SphericalDifferential
            )
        else:
            return self._CR(t, diff=diff).represent_as(
                SphericalRepresentation
            )

    def _xyz(self, t, c):
        return object.__getattribute__(
            self._CR(t),
            c
        )

    def _vxyz(self, t, c):
        return object.__getattribute__(
            self._CR(t, diff=True).differentials['s'],
            'd_' + c
        )

    def _cyl(self, t, c):
        return object.__getattribute__(
            self._CyR(t),
            dict(R='rho').get(c, c)
        )

    def _vcyl(self, t, c):
        return object.__getattribute__(
            self._CyR(t, diff=True).differentials['s'],
            'd_' + dict(R='rho').get(c, c)
        )

    def _sph(self, t, c):
        return object.__getattribute__(
            self._SR(t), dict(r='distance').get(c, c)
        )

    def _vsph(self, t, c):
        return object.__getattribut__(
            self._SR(t, diff=True).differentials['s'],
            'd_' + dict(r='distance'.get(c, c))
        )
