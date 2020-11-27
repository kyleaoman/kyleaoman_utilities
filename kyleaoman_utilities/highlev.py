import h5py
from os import path
import numpy as np
from astropy.cosmology import Planck13 as cosmo, z_at_value
import astropy.units as U
from kyleaoman_utilities.cosmology.rvir_conventions import Delta_vir


def _replace_invalid(arr, val, rep):
    arr[arr == val] = rep
    return arr


def _to_proper(d, a):
    if np.array(a).size == 1:
        return d * a
    shape = [si if si == len(a) else 1 for si in d.shape]
    return d * a.reshape(shape)


def _to_comoving(d, a):
    if np.array(a).size == 1:
        return d / a
    shape = [si if si == len(a) else 1 for si in d.shape]
    return d / a.reshape(shape)


def recentre(xyz, centre=np.zeros(3) * U.Mpc, Lbox=100 * U.Mpc, a=None):
    if a is None:
        raise ValueError('recentre requires comoving coordinates, '
                         'provide (array of) scale factor')
    xyz = _to_comoving(xyz, a)
    if centre.ndim == 1 and np.array(a).size > 1:
        centre = _to_comoving(
            np.repeat(centre[np.newaxis], a.size, axis=0),
            a
        )
    else:
        centre = _to_comoving(centre, a)
    xyz = xyz - centre
    xyz[xyz < -Lbox / 2.] += Lbox
    xyz[xyz > Lbox / 2.] -= Lbox
    return _to_proper(xyz, a)


def M_to_sigma(M, z=0, mode='3D'):
    if mode == '3D':
        ndim_scale = np.sqrt(3)
    elif mode == '1D':
        ndim_scale = 1.
    # return prefac * .00989 * U.km / U.s \
    #     * np.power(M.to(U.Msun).value, 1 / 3)
    # Note: Delta_vir returns the overdensity in units of background,
    # which is Delta_c * Om in B&N98 notation.
    return 0.016742 * U.km * U.s ** -1 * np.power(
        np.power(M.to(U.Msun).value, 2)
        * Delta_vir(z) / Delta_vir(0)
        * cosmo.Om(z) / cosmo.Om0
        * np.power(cosmo.H(z) / cosmo.H0, 2),
        1 / 6
    ) / ndim_scale


def host_mask(HL, Mrange=(0, np.inf), snap=-1, ret_inds=False,
              ret_interp_inds=False):
    if ret_interp_inds and not ret_inds:
        raise ValueError('ret_inds required for ret_interp_inds')
    if snap < 0:
        snap = len(HL.snap_times) + snap
    mask = np.logical_and(
        np.logical_and(
            np.logical_not(HL.SatFlag[:, snap]),
            np.logical_not(HL.ContFlag[:, snap])
        ),
        np.logical_and(
            HL.M200[:, snap] > Mrange[0],
            HL.M200[:, snap] < Mrange[1]
        )
    )
    inds = (np.where(mask)[0], ) if ret_inds else tuple()
    interp_inds = (HL.interpGalaxyRevIndex[inds], ) if ret_interp_inds \
        else tuple()
    return (mask, ) + inds + interp_inds


def sat_mask(HL, host, Rcut=3.35, snap=-1, ret_inds=False,
             ret_interp_inds=False):
    if snap < 0:
        snap = len(HL.snap_times) + snap
    if ret_interp_inds and not ret_inds:
        raise ValueError('ret_inds required for ret_interp_inds')
    xyz = recentre(HL.Centre[:, snap], host.Centre[snap], Lbox=HL.Lbox,
                   a=HL.snap_scales[snap])
    cube_mask = (np.abs(xyz) < Rcut * host.R200[snap]).all(axis=-1)
    sphere_mask = np.sum(np.power(xyz[cube_mask], 2), axis=-1) \
        < np.power(Rcut * host.R200[snap], 2)
    cube_mask[cube_mask] = sphere_mask  # refine to sphere
    mask = cube_mask
    mask[host.ind] = False
    inds = (np.where(mask)[0], ) if ret_inds else tuple()
    interp_inds = (HL.interpGalaxyRevIndex[inds], ) if ret_interp_inds \
        else tuple()
    return (mask, ) + inds + interp_inds


def inter_mask(HL, host, Rcut, Vcut, H=70. * U.km / U.s / U.Mpc, snap=-1,
               ret_inds=False, ret_interp_inds=False):
    if snap < 0:
        snap = len(HL.snap_times) + snap
    if ret_interp_inds and not ret_inds:
        raise ValueError('ret_inds required for ret_interp_inds')
    xyz = recentre(HL.Centre[:, snap], host.Centre[snap], Lbox=HL.Lbox,
                   a=HL.snap_scales[snap])
    vxyz = HL.Velocity[:, snap] - host.Velocity[snap]
    prism_mask = np.logical_and(
        (np.abs(xyz[:, :2]) < Rcut * host.R200[snap]).all(axis=-1),
        np.abs(vxyz[:, 2] + H * xyz[:, 2]) < Vcut * M_to_sigma(host.M200[snap])
    )
    cylinder_mask = np.sum(np.power(xyz[prism_mask][:, :2], 2), axis=-1) \
        < np.power(Rcut * host.R200[snap], 2)
    prism_mask[prism_mask] = cylinder_mask  # refine to cylinder
    cube_mask = (np.abs(xyz) < Rcut * host.R200[snap]).all(axis=-1)
    sphere_mask = np.sum(np.power(xyz[cube_mask], 2), axis=-1) \
        < np.power(Rcut * host.R200[snap], 2)
    cube_mask[cube_mask] = sphere_mask  # refine to sphere
    mask = np.logical_and(
        np.logical_not(cube_mask),  # refined to sphere
        prism_mask  # refined to cylinder
    )
    mask[host.ind] = False
    inds = (np.where(mask)[0], ) if ret_inds else tuple()
    interp_inds = (HL.interpGalaxyRevIndex[inds], ) if ret_interp_inds \
        else tuple()
    return (mask, ) + inds + interp_inds


# could/should generalize to be along a given axis
def find_peris(r, Rcut=3.35):
    minima = np.logical_and(
        np.concatenate((
            np.ones(r.shape[0])[..., np.newaxis],
            r[:, 1:] < r[:, :-1]
        ), axis=1),
        np.concatenate((
            r[:, :-1] < r[:, 1:],
            np.ones(r.shape[0])[..., np.newaxis]
        ), axis=1)
    )
    minima = np.logical_and(minima, r < Rcut)
    minima[:, -1] = False
    return minima


def t_r_firstperi(peris, t, r):
    ifirstperi, jfirstperi = np.nonzero(peris)
    jfirstperi = jfirstperi[np.r_[True, np.diff(ifirstperi) > 0]]
    ifirstperi = ifirstperi[np.r_[True, np.diff(ifirstperi) > 0]]
    tfirstperi = np.zeros(peris.shape[0]) * np.nan * t.unit
    rfirstperi = np.zeros(peris.shape[0]) * np.nan * r.unit
    tfirstperi[peris.any(axis=1)] = t[jfirstperi]
    rfirstperi[peris.any(axis=1)] = r[ifirstperi, jfirstperi]
    return tfirstperi, rfirstperi


def t_firstinfall(r, t, Rcut=3.35):
    inside = r < Rcut
    iinfall, jinfall = np.nonzero(inside)
    jinfall = jinfall[np.r_[True, np.diff(iinfall) > 0]]
    tfirstinfall = np.zeros(inside.shape[0]) * np.nan * t.unit
    tfirstinfall[inside.any(axis=1)] = t[jinfall]
    return tfirstinfall


class _Gal(object):

    def __init__(self, HL, ind, interp_ind=None):

        self.HL = HL
        self.ind = ind
        self.interp_ind = interp_ind
        return

    def __getitem__(self, k):
        if (k in self.HL.interp_keys):
            if (self.interp_ind is not None):
                retval = self.HL[k][self.interp_ind]
                shape = (np.sum(self.interp_ind == -1), 3, 1)
                retval[self.interp_ind == -1] = \
                    np.ones(shape) * np.nan
                return retval
            else:
                raise ValueError('Host requires interp_ind for interpolated'
                                 ' values.')
        elif k in (self.HL.prop_keys | self.HL.pos_keys | self.HL.snep_keys):
            return self.HL[k][self.ind]
        else:
            raise KeyError

    def __getattribute__(self, name):
        if name in object.__getattribute__(self, 'HL').pos_keys | \
           object.__getattribute__(self, 'HL').prop_keys | \
           object.__getattribute__(self, 'HL').snep_keys | \
           object.__getattribute__(self, 'HL').interp_keys:
            return self.__getitem__(name)
        else:
            return object.__getattribute__(self, name)


class Sats(_Gal):

    def __init__(self, HL, ind, interp_ind=None):
        super().__init__(HL, ind, interp_ind=interp_ind)
        return

    def __len__(self):
        # will break if self.ind is not a bool mask
        return np.sum(self.ind)


class Interlopers(_Gal):

    def __init__(self, HL, ind, interp_ind=None):
        super().__init__(HL, ind, interp_ind=interp_ind)
        return

    def __len__(self):
        # will break if self.ind is not a bool mask
        return np.sum(self.ind)


class Host(_Gal):

    def __init__(self, HL, ind, interp_ind=None, f_sat_mask=None,
                 f_inter_mask=None):
        super().__init__(HL, ind, interp_ind=interp_ind)
        if f_sat_mask is not None:
            if interp_ind is not None:
                sm, sat_inds, sat_interp_inds = f_sat_mask(self)
            else:
                sm, sat_inds = f_sat_mask(self)
                sat_interp_inds = None
            self.sats = Sats(
                HL,
                sm,
                interp_ind=sat_interp_inds
            )
        else:
            self.sats = None
        if f_inter_mask is not None:
            if interp_ind is not None:
                im, inter_inds, inter_interp_inds = f_inter_mask(self)
            else:
                im, inter_inds = f_inter_mask(self)
                inter_interp_inds = None
            self.interlopers = Interlopers(
                HL,
                im,
                interp_ind=inter_interp_inds
            )
        return


class HighLev(object):

    Hydrangea_CEs = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                     18, 21, 22, 24, 25, 28, 29)

    Lbox = 3200 * U.Mpc
    h = .6777

    _units = dict(
        M200=U.Msun,
        MBH=U.Msun,
        MDM=U.Msun,
        MGas=U.Msun,
        Mgas30kpc=U.Msun,
        Mstar=U.Msun,
        Mstar30kpc=U.Msun,
        MstarInit=U.Msun,
        Msub=U.Msun,
        R200=U.Mpc,
        SFR=U.Msun * U.yr ** -1,
        StellarHalfMassRad=U.Mpc,
        Vmax=U.km * U.s ** -1,
        VmaxRadius=U.Mpc,
        Centre=U.Mpc,
        Velocity=U.km * U.s ** -1,
        snepCoordinateDispersion=U.Mpc,
        snepCoordinates=U.Mpc,
        snepVelocity=U.km * U.s ** -1,
        snepVelocityDispersion=U.km * U.s ** -1,
        interpInterpolatedPositions=U.Mpc,
        interpInterpolationTimes=U.Gyr,
        MHI=U.Msun,
        MHneutral=U.Msun
    )

    _replacements = dict(
        M200=lambda x: _replace_invalid(x, -1, np.nan),
        MBH=lambda x: _replace_invalid(x, -1, np.nan),
        MDM=lambda x: _replace_invalid(x, -1, np.nan),
        MGas=lambda x: _replace_invalid(x, -1, np.nan),
        Mgas30kpc=lambda x: _replace_invalid(x, -1, np.nan),
        Mstar=lambda x: _replace_invalid(x, -1, np.nan),
        Mstar30kpc=lambda x: _replace_invalid(x, -1, np.nan),
        MstarInit=lambda x: _replace_invalid(x, -1, np.nan),
        Msub=lambda x: _replace_invalid(x, -1, np.nan),
        MHI=lambda x: _replace_invalid(x, -1, np.nan),
        MHneutral=lambda x: _replace_invalid(x, -1, np.nan)
    )

    _is_log = {'M200', 'MBH', 'MDM', 'MGas', 'Mgas30kpc', 'Mstar',
               'Mstar30kpc', 'MstarInit', 'Msub', 'SFR', 'MHI', 'MHneutral'}

    prop_keys = {'CenGal', 'ContFlag', 'M200', 'MBH', 'MDM', 'MGas',
                 'Mgas30kpc', 'Mstar', 'Mstar30kpc', 'MstarInit',
                 'Msub', 'R200', 'SFR', 'SHI', 'SatFlag',
                 'StellarHalfMassRad', 'Vmax', 'VmaxRadius', 'MHI',
                 'MHneutral'}

    pos_keys = {'Centre', 'Velocity'}

    snep_keys = {'snepCoordinateDispersion', 'snepCoordinates',
                 'snepVelocity', 'snepVelocityDispersion'}

    interp_keys = {'interpGalaxy', 'interpGalaxyRevIndex',
                   'interpInterpolatedPositions',
                   'interpInterpolationTimes'}

    def __init__(self, CE):

        self._base_dir = '/virgo/simulations/Hydrangea/10r200/CE-{:.0f}/'\
            'HYDRO/highlev/'.format(CE)
        self._propfile = path.join(self._base_dir, 'FullGalaxyTables.hdf5')
        self._posfile = path.join(self._base_dir, 'GalaxyPositionsSnap.hdf5')
        self._snepfile = path.join(self._base_dir, 'GalaxyPaths.hdf5')
        self._interp_file = path.join(self._base_dir, 'GalaxyCoordinates10Myr.hdf5')
        with h5py.File(self._snepfile, 'r') as f:
            self.snep_redshifts = np.array([
                f['Snepshot_{:04d}'.format(sn_i)].attrs['Redshift'][0]
                for sn_i in f['RootIndex/Basic']
            ])
            self.snap_redshifts = np.array([
                f['Snepshot_{:04d}'.format(sn_i)].attrs['Redshift'][0]
                for sn_i in f['SnapshotIndex']
            ])
            self.snip_redshifts = np.array([
                f['Snepshot_{:04d}'.format(sn_i)].attrs['Redshift'][0]
                for sn_i in f['SnipshotIndex']
            ])
            self.snap_scales = 1 / (1 + self.snap_redshifts)
            self.snap_times = cosmo.age(self.snap_redshifts)
            self.snip_scales = 1 / (1 + self.snip_redshifts)
            self.snip_times = cosmo.age(self.snip_redshifts)
            self.snep_scales = 1 / (1 + self.snep_redshifts)
            self.snep_times = cosmo.age(self.snep_redshifts)
        self._data = dict()
        return

    def _format_data(self, k, data):
        data = np.array(data)
        if k in self._replacements:
            data = self._replacements[k](data)
        if k in self._is_log:
            data = np.power(10, data)
        if k in self._units:
            data *= self._units[k]
        return data

    def _load(self, k, snepset='Basic'):
        if k in self.prop_keys:
            with h5py.File(self._propfile, 'r') as pf:
                self._data[k] = self._format_data(k, pf[k])
        elif k in self.pos_keys:
            with h5py.File(self._posfile, 'r') as pf:
                self._data[k] = self._format_data(k, pf[k])
        elif k in self.snep_keys:
            with h5py.File(self._snepfile, 'r') as pf:
                sneplist = np.array(pf['/RootIndex/'+snepset])
                parts = [np.array(
                    pf['Snepshot_{:04d}/{:s}'.format(s, k[4:])]
                )[:, np.newaxis] for s in sneplist]
                data = np.concatenate(parts, axis=1)
                self._data[k] = self._format_data(k, data)
        elif k in self.interp_keys:
            with h5py.File(self._interp_file, 'r') as pf:
                self._data[k] = self._format_data(k, pf[k[6:]])
        else:
            raise KeyError('Unknown key {:s}.'.format(k))
        if k in {'snepCoordinates', 'snepCoordinateDispersion'}:
            self._data[k] = _to_proper(self._data[k], self.snep_scales) / self.h

    def _load_all(self):
        for k in self.prop_keys | self.pos_keys:
            self._load(k)

    def __contains__(self, key):
        return self._data.__contains__(key)

    def __delitem__(self, key):
        return self._data.__delitem__(key)

    def __eq__(self, value):
        return self._data.__eq__(value)

    def __ge__(self, value):
        return self._data.__ge__(value)

    def __getitem__(self, key):
        if key not in self._data:
            self._load(key)
        return self._data[key]

    def __getattribute__(self, name):
        if name in object.__getattribute__(self, 'pos_keys') | \
           object.__getattribute__(self, 'prop_keys') | \
           object.__getattribute__(self, 'snep_keys') | \
           object.__getattribute__(self, 'interp_keys'):
            return self.__getitem__(name)
        else:
            return object.__getattribute__(self, name)

    def __gt__(self, value):
        return self._data.__gt__(value)

    def __iter__(self):
        return self._data.__iter__()

    def __le__(self, value):
        return self._data.__le__(value)

    def __len__(self):
        return self._data.__len__()

    def __lt__(self, value):
        return self._data.__lt__(value)

    def __ne__(self, value):
        return self._data.__ne__(value)

    def __repr__(self, value):
        return self._data.__repr__()

    def clear(self):
        return self._data.clear()

    def get(self, key, default=None):
        return self._data.get(key, default)

    def items(self):
        self._load_all()
        return self._data.items()

    def keys(self):
        return self.pos_keys | self.prop_keys

    def values(self):
        self._load_all()
        return self._data.values()
