import numpy as np
from astropy import units as U, constants as C


def vcirc(
    xyz,
    m,
    plane="xy",
    soften=0.1 * U.kpc,
    sample_r=np.linspace(0, 50, 101) * U.kpc,
    sample_phi=np.linspace(0, 2 * np.pi, 17)[:-1] * U.rad,
    return_percentiles=False,
    decimate=np.s_[...],
):
    grid_r, grid_phi = np.meshgrid(sample_r, sample_phi)
    grid_x = grid_r * np.cos(grid_phi)
    grid_y = grid_r * np.sin(grid_phi)
    grid_z = np.zeros(grid_x.shape)
    grid_xyz = np.r_[grid_x[np.newaxis], grid_y[np.newaxis], grid_z[np.newaxis]]
    grid_xyz = np.transpose(grid_xyz, (1, 2, 0))
    if plane == "xy":
        pass
    elif plane == "yz":
        xyz = xyz[:, [1, 2, 0]]
    elif plane == "xz":
        xyz = xyz[:, [2, 0, 1]]
    xyz = xyz[decimate, :]
    m = m[decimate]
    grid_dxyz = xyz[:, np.newaxis, np.newaxis, :] - grid_xyz
    grid_dr = np.sqrt(np.sum(np.power(grid_dxyz, 2), axis=-1))
    grid_dr_soft = np.sqrt(np.power(grid_dr, 2) + np.power(soften, 2))
    accel = np.sum(
        C.G
        * m[:, np.newaxis, np.newaxis, np.newaxis]
        * grid_dxyz
        / np.power(grid_dr_soft[..., np.newaxis], 3),
        axis=0,
    )
    grid_rhat = np.array(
        [-np.cos(grid_phi), -np.sin(grid_phi), np.zeros(grid_phi.shape)]
    )
    grid_rhat = np.transpose(grid_rhat, (1, 2, 0))
    accel_r = np.sum(accel * grid_rhat, axis=-1)
    vc_direct = (np.sign(accel_r) * np.sqrt(np.abs(accel_r) * grid_r)).to(U.km / U.s)
    if return_percentiles:
        return (
            np.median(vc_direct, axis=0),
            sample_r,
            np.percentile(vc_direct, 16, axis=0),
            np.percentile(vc_direct, 84, axis=0),
        )
    else:
        return np.median(vc_direct, axis=0), sample_r
