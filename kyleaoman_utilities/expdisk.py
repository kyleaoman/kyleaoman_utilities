import numpy as np
from scipy.integrate import dblquad
from scipy.special import kv, iv
from scipy.interpolate import CubicSpline


def expdisc_potential_integrand(
    zp,
    a,
    R=0,
    z=0,
    Rd=1,
    G=1,
    Sig0=1,
    zeta=lambda zp, zd=1: np.exp(np.abs(zp) / zd) / 2 / zd,
    zeta_kwargs=dict(zd=1),
):
    """
    Integrand involved in calculating the potential of a thick exponential disc.

    The calculation assumes that the radial an vertical structure is separable, such that
    the function zeta(z) has no dependence on R. The default values are for a "double
    exponential disc" with an exponential decay away from the midplane.

    Parameters
    ----------
    zp: float or array-like
        Integration variable in the vertical direction.

    a: float or array-like
        Integration variable in the radial direction, loosely speaking.

    R: float or array-like, optional
        Radial coordinate for potential evaluation. (Default: 0)

    z: float or array-like, optional
        Vertical coordinate for potential evaluation. (Default: 0)

    Rd: float, optional
        Scale length of exponential disc. (Default: 1)

    G: float, optional
        Gravitational constant. (Default: 1)

    Sig0: float, optional
        Central surface density of exponential disc. (Default: 1)

    zeta: callable, optional
        Function of a single argument (zp) and any number of keyword arguments (see
        zeta_kwargs) describing the vertical structure of the exponential disc. Must not
        depend on the radial coordinate (R). The default is a "double exponential disc".
        (Default: lambda zp, zd=1: np.exp(np.abs(zp) / zd) / 2 / zd)

    zeta_kwargs: dict, optional
        Keyword argiments for the zeta function. The default is the scale height of the
        "double exponential disc". (Default: dict(zd=1))

    Returns
    -------
    out: float or array-like
        Integrand evaluated for given input parameters.

    References
    ----------
    Binney & Tremaine (2008) "Galactic Dynamics 2e" Sec. 2.6.1(c) Eq. 2.170
    """
    return (
        -4
        * G
        * Sig0
        / Rd
        * np.arcsin(
            2
            * a
            / (
                np.sqrt((zp - z) ** 2 + (a + R) ** 2)
                + np.sqrt((zp - z) ** 2 + (a - R) ** 2)
            )
        )
        * a
        * kv(0, a / Rd)
        * zeta(zp, **zeta_kwargs)
    )


def expdisc_vcirc(
    z=0,
    Rd=1,
    G=1,
    Sig0=1,
    zeta=lambda zp, zd=1: np.exp(np.abs(zp) / zd) / 2 / zd,
    zeta_kwargs=dict(zd=1),
    R_samples=np.linspace(0, 5, 50),
):
    """
    Calculation of the circular velocity in an exponential disc with vertical structure.

    The calculation assumes that the radial an vertical structure is separable, such that
    the function zeta(z) has no dependence on R. The default values are for a "double
    exponential disc" with an exponential decay away from the midplane. Returns a function
    that evaluates the circular velocity for the specified fixed layer (height z). The
    integration assumes a disc that is symmetric about the mid-plane. The potential is
    first sampled at the points R_samples (at fixed z) by numerical integration, then a
    cubic spline interpolation is used to evaluate the partial derivative of the potential
    needed to calculate the circular velocity.

    Parameters
    ----------
    z: float, optional
        Height in the disc where circular velocity will be calculated. (Default: 0)

    Rd: float, optional
        Scale length of exponential disc. (Default: 1)

    G: float, optional
        Gravitational constant. (Default: 1)

    Sig0: float, optional
        Central surface density of exponential disc. (Default: 1)

    zeta: callable or str, optional
        Function of a single argument (zp) and any number of keyword arguments (see
        zeta_kwargs) describing the vertical structure of the exponential disc. Must not
        depend on the radial coordinate (R). The default is a "double exponential disc".
        A string "razor" or "spherical" can be provided instead for the razor-thin case
        (B&T eq. 2.165) or sphericalized case (B&T eq. 2.166 & Fig. 2.17), respectively.
        These are only implemented for the mid-plane (z=0) case.
        (Default: lambda zp, zd=1: np.exp(np.abs(zp) / zd) / 2 / zd)

    zeta_kwargs: dict, optional
        Keyword argiments for the zeta function. The default is the scale height of the
        "double exponential disc". Ignored if zeta is "razor" or "spherical".
        (Default: dict(zd=1))

    R_samples: array-like, optional
        Radial locations where the potential will be sampled. Ignored if zeta is "razor"
        or "spherical".
        (Default: np.linspace(0, 5, 50))

    Returns
    -------
    out: callable
        A function of one variable accepting a radius where the circular velocity is to
        be evaluated.

    References
    ----------
    Binney & Tremaine (2008) "Galactic Dynamics 2e" Sec. 2.6.1(c) Eq. 2.170

    """
    if zeta == "razor":
        # razor thin disc (B&T eq. 2.165)
        if z != 0:
            raise ValueError("Razor thin disc only implemented for mid-plane (z=0).")
        return lambda R: np.sqrt(
            4
            * np.pi
            * G
            * Sig0
            * (R / 2 / Rd) ** 2
            * (
                iv(0, R / 2 / Rd) * kv(0, R / 2 / Rd)
                - iv(1, R / 2 / Rd) * kv(1, R / 2 / Rd)
            )
        )
    elif zeta == "spherical":
        # "spherical" exponential disc (B&T eq. 2.166 & Fig. 2.17)
        if z != 0:
            raise ValueError(
                "Sphericalized exponential disc only implemented for mid-plane (z=0)."
            )
        return lambda R: np.sqrt(2 * np.pi * Sig0 * Rd * (1 - np.exp(-R) * (1 + R)) / R)
    Phi = np.array(
        [
            dblquad(
                lambda zp, a: 2
                * expdisc_potential_integrand(
                    zp,
                    a,
                    R=Ri,
                    z=z,
                    Rd=Rd,
                    G=G,
                    Sig0=Sig0,
                    zeta=zeta,
                    zeta_kwargs=zeta_kwargs,
                ),
                0,
                10 * Rd,
                0,
                10 * Rd,
            )
            for Ri in R_samples
        ]
    )

    cs = CubicSpline(R_samples, Phi)

    return lambda R: np.sqrt(R * cs(R, 1))


def Sig0_from_mass(m, Rd):
    """
    Central surface density given mass and scale length.

    Parameters
    ----------
    m: float or array-like
        Mass of exponential disc.

    Rd: float or array-like
        Scale length of exponential disc.

    Returns
    -------
    out: float or array-like
        Central surface density of exponential disc.
    """
    return m / 2 / np.pi / Rd**2
