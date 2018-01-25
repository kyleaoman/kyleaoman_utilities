import numpy as np

def getR(x, y, incl=0., PA=np.radians(270.), x0=0., y0=0.):
    #PA -= 3. * np.pi / 2.
    return np.sqrt(
        np.power((x - x0) * np.cos(PA) + (y - y0) * np.sin(PA), 2) +
        np.power((x - x0) * np.sin(PA) - (y - y0) * np.cos(PA), 2)
        / np.power(np.cos(incl), 2)
    )

#NOTE can use either t, the "mean anomaly (?)": as appears in the parametric equation of an ellipse, or phi, the "eccentric anomaly": simply arctan(y/x) on a grid for instance

def phi_to_t(phi, incl=None, PA=0.):
    #PROVIDE PA ONLY FOR PRODUCING A t_grid, e.g. phi_to_t(phi_grid) where phi_grid = np.arctan2(*np.meshgrid(xs, ys)[::-1])
    if incl is None:
        raise ValueError
    #PA = PA - 3. * np.pi / 2.
    phi = phi - PA
    retval = np.arctan(np.tan(phi) / np.cos(incl)) + np.pi * np.trunc(np.sign(phi) * (np.abs(phi) + np.pi / 2.) / np.pi)
    while (retval < 0.).any():
        retval[retval < 0.] += 2. * np.pi
    while (retval > 2 * np.pi).any():
        retval[retval > 2 * np.pi] -= 2. * np.pi
    return retval

def getx(t=None, phi=None, incl=0., PA=np.radians(270.), x0=0., y0=0., R=1.):
    if t is not None:
        pass
    elif phi is not None:
        t = phi_to_t(phi, incl=incl)
    else:
        raise ValueError
        
    #PA -= 3. * np.pi / 2.
    return R * np.cos(t) * np.cos(PA) - R * np.cos(incl) * np.sin(t) * np.sin(PA) + x0

def gety(t=None, phi=None, incl=0., PA=np.radians(270.), x0=0., y0=0., R=1.):
    if t is not None:
        pass
    elif phi is not None:
        t = phi_to_t(phi, incl=incl)
    else:
        raise ValueError
    
    #PA -= 3 * np.pi / 2.
    return R * np.cos(t) * np.sin(PA) + R * np.cos(incl) * np.sin(t) * np.cos(PA) + y0

def getR_fromphi(t=None, phi=None, incl=0., PA=np.radians(270.), x0=0., y0=0., R=1.):
    args = {'t': t, 'phi': phi, 'incl': incl, 'PA': PA, 'x0': x0, 'y0': y0, 'R': R}
    return np.sqrt(np.power(ellipse_getx(**args), 2) + np.power(ellipse_gety(**args), 2))

"""
#TESTING
import matplotlib.pyplot as pp
incl = np.radians(60.)
PA = np.radians(220.)
x0 = 1.
y0 = 1.

fig = pp.figure(1)
sp = fig.add_subplot(1, 1, 1, aspect='equal')

ts = np.linspace(0, 2 * np.pi, 200, endpoint=True)

for R in [1.,.6,.2]:

    xplot = ellipse_getx(ts, incl=incl, PA=PA, x0=x0, y0=y0, R=R)
    yplot = ellipse_gety(ts, incl=incl, PA=PA, x0=x0, y0=y0, R=R)

    xarr = np.linspace(0, 2, 50)
    yarr = np.linspace(0, 2, 50)

    xgrid, ygrid = np.meshgrid(xarr, yarr)
    Rgrid = np.abs(ellipse_getR(xgrid, ygrid, incl=incl, PA=PA, x0=x0, y0=y0) - R)
    mask = Rgrid > .2

    sp.imshow(mask, origin='lower', extent=[xarr[0], xarr[-1], yarr[0], yarr[-1]], cmap='hot', interpolation='nearest', alpha=.5)
    sp.plot(xplot, yplot, '-b')
pp.show()

"""
