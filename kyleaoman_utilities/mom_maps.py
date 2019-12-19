import numpy as np
import matplotlib.pyplot as pp
from astropy.io import fits
from astropy import wcs
from astropy import units as U
from astropy.constants import m_p
from astropy.coordinates import SkyCoord


def _genmask(mom0, Sigma_lim=19.5):
    return lambda I: np.where(np.log10(mom0.value) > Sigma_lim, I, np.nan) \
        * I.unit


def _getsep(c1, c2):
    return SkyCoord(*c1).separation(SkyCoord(*c2)).to(U.rad)


def mom_maps(fout, fitsfile, crop=0, label='', Sigma_lim=19.5,
             D=3 * U.Mpc, beamsize=6 * U.arcsec, px_scale=3 * U.arcsec,
             vsys=None, max_v=100 * U.km / U.s):

    with fits.open(fitsfile) as f:
        hdu = f[0]
        datacube = hdu.data

    if vsys is None:
        vsys = (D * 70 * U.km / U.s / U.Mpc).to(U.km / U.s)

    world_coords = wcs.WCS(hdu.header).dropaxis(3).dropaxis(2)
    Npixels = datacube.shape[-1]
    Nchannels = datacube.shape[1]

    Vbin_mids = np.array(wcs.WCS(hdu.header).sub(axes=[3]).all_pix2world(
        np.arange(Nchannels), 0))[0] * U.m / U.s
    Vbin_mids = (Vbin_mids - vsys).to(U.km / U.s)
    Wchannel = Vbin_mids[1] - Vbin_mids[0]

    px_centre = (Npixels / 2 - .5 * (not bool(Npixels % 2)),
                 Npixels / 2 - .5 * (not bool(Npixels % 2)))

    px_per_beam = 4 * np.pi * np.power((beamsize / px_scale) / 2, 2).to(
        U.dimensionless_unscaled).value

    datacube = datacube / px_per_beam * U.Jy
    datacube = 2.36E5 * U.Msun * datacube.to(U.Jy).value \
        / np.power(D.to(U.Mpc).value, -2) \
        * Wchannel.to(U.km / U.s).value
    datacube = (datacube / m_p).to(U.dimensionless_unscaled).value \
        / np.power((px_scale * D).to(U.cm, U.dimensionless_angles()), 2)

    mom0 = np.sum(datacube[0], axis=0).to(U.cm ** -2)
    mom1 = (np.sum(datacube[0] * Vbin_mids[..., np.newaxis, np.newaxis],
                   axis=0) / mom0).to(U.km / U.s)
    mom2 = (np.sqrt(np.sum(
        datacube[0]
        * np.power(
            np.tile(
                Vbin_mids,
                (Npixels, Npixels, 1)
            ).transpose(2, 0, 1)
            - mom1[np.newaxis, ...],
            2),
        axis=0) / mom0)).to(U.km / U.s)

    mask = _genmask(mom0, Sigma_lim=Sigma_lim)

    coord_centre = world_coords.all_pix2world(
        *px_centre, 0) * U.deg
    coord_left = world_coords.all_pix2world(
        0 + crop, px_centre[1], 0) * U.deg
    coord_right = world_coords.all_pix2world(
        Npixels - 1 - crop, px_centre[1], 0) * U.deg
    coord_bottom = world_coords.all_pix2world(
        px_centre[0], 0 + crop, 0) * U.deg
    coord_top = world_coords.all_pix2world(
        px_centre[0], Npixels - 1 - crop, 0) * U.deg
    ang_left = _getsep(coord_left, coord_centre)
    ang_right = _getsep(coord_right, coord_centre)
    ang_bottom = _getsep(coord_bottom, coord_centre)
    ang_top = _getsep(coord_top, coord_centre)

    fig = pp.figure(1, figsize=(9, 6))
    pp.clf()
    sps = [fig.add_subplot(1, 3, i, projection=world_coords)
           for i in range(1, 4)]
    for sp in sps:
        for i in [0, 1]:
            sp.coords[i].set_ticklabel(size=6)
        sp.plot(px_centre[0], px_centre[1], marker='x', mec='k')
        sp.coords[0].set_axislabel(r'$\alpha$', size=8, minpad=1.)
        sp.coords[0].set_ticks_position('b')
        sp.coords[1].set_ticks_position('l')
        sp.axis([0 + crop, Npixels - 1 - crop, 0 + crop, Npixels - 1 - crop])
    sps[0].coords[1].set_axislabel(r'$\delta$', size=8, minpad=1.)
    sps[1].coords[1].set_ticklabel(visible=False)
    sps[2].coords[1].set_ticklabel(visible=False)

    pp.sca(sps[0])
    im = sps[0].imshow(
        np.log10(mom0.value),
        origin='lower',
        cmap='gray_r',
        interpolation='nearest',
        vmin=18,
        vmax=22,
        aspect='equal'
    )
    cs = sps[0].contour(
        np.arange(Npixels),
        np.arange(Npixels),
        np.log10(mom0.value),
        [Sigma_lim],
        colors='#FF0000',
        linestyles='solid'
    )
    cb = pp.colorbar(mappable=im, orientation='horizontal', fraction=.04)
    cb.ax.tick_params(axis='both', labelsize=4, length=2.)
    cb.set_label(r'$\log_{10}(\Sigma_{\rm HI}/{\rm atoms}\,{\rm cm}^{-2})$',
                 size=8, labelpad=1.5)
    cb.add_lines(cs)
    sps[0].text(.1, .1, label,
                size=8, ha='left', va='bottom', transform=sps[0].transAxes)

    pp.sca(sps[1])
    im = sps[1].imshow(
        mask(mom1).value,
        origin='lower',
        cmap='RdBu_r',
        interpolation='nearest',
        vmin=-max_v.to(U.km / U.s).value,
        vmax=max_v.to(U.km / U.s).value,
        aspect='equal'
    )
    cb = pp.colorbar(mappable=im, orientation='horizontal', fraction=.04)
    cb.ax.tick_params(axis='both', labelsize=4, length=2.)
    cb.set_label(r'$V-V_{\rm sys}\,[{\rm km}\,{\rm s}^{-1}]$',
                 size=8, labelpad=1.5)
    sps[1].contour(
        np.arange(Npixels),
        np.arange(Npixels),
        mask(mom1),
        cb.ax.get_xticks(),
        colors='#333333',
        linestyles='solid'
    )

    pp.sca(sps[2])
    im = sps[2].imshow(
        np.log10(mask(mom2).value),
        origin='lower',
        cmap='hot',
        interpolation='nearest',
        vmin=np.log10(3.),
        vmax=np.log10(30.),
        aspect='equal'
    )
    cb = pp.colorbar(mappable=im, orientation='horizontal', fraction=.04)
    cb.ax.tick_params(axis='both', labelsize=4, length=2.)
    cb.set_ticks([np.log10(3.), np.log10(5.), np.log10(10.), np.log10(15.),
                  np.log10(20.), np.log10(30.)])
    cb.set_ticklabels(['3', '5', '10', '15', '20', '30'])
    cb.set_label(r'$\sigma\,[{\rm km}\,{\rm s}^{-1}]$', size=8, labelpad=1.5)

    pp.subplots_adjust(wspace=.05, hspace=.25)
    phys_axs = [fig.add_axes(sps[i].get_position()) for i in range(3)]
    for phys_ax in phys_axs:
        phys_ax.set_facecolor('None')
        phys_ax.set_aspect('equal')
        phys_ax.set_xlim(
            -(ang_left * D).to(U.kpc, U.dimensionless_angles()).value,
            (ang_right * D).to(U.kpc, U.dimensionless_angles()).value
        )
        phys_ax.set_ylim(
            -(ang_bottom * D).to(U.kpc, U.dimensionless_angles()).value,
            (ang_top * D).to(U.kpc, U.dimensionless_angles()).value
        )
        phys_ax.tick_params(bottom=False, top=True, left=False, right=True,
                            labelbottom=False, labeltop=True, labelleft=False,
                            labelright=False, labelsize=6)
        phys_ax.set_xlabel(r'$X\,[{\rm kpc}]$', size=8)
        phys_ax.xaxis.set_label_position('top')
    phys_axs[2].tick_params(labelright=True)
    phys_axs[2].set_ylabel(r'$Y\,[{\rm kpc}]$', size=8)
    phys_axs[2].yaxis.set_label_position('right')

    pp.savefig(fout, format='pdf', bbox_inches='tight', pad_inches=.02)
