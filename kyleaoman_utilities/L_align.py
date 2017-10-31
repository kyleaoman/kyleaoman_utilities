import numpy as np

def L_align(xyz, vxyz, m, frac=.3, saverot=None, Laxis=(0, 0, 1), xaxis=None):

    transposed = False
    if xyz.ndim != 2:
        raise ValueError('L_align: cannot guess coordinate axis for input with ndim != 2.')
    else if (xyz.shape[0] == 3) and (xyz.shape[1] == 3):
        raise ValueError('L_align: cannot guess coordinate axis for input with shape (3, 3).')
    else if xyz.shape[1] == 3:
        xyz = xyz.T
        vxyz = vxyz.T
        transposed = True

    rsort = np.argsort(np.sum(np.power(xyz, 2), axis=0), kind='quicksort')
    p = m[:, np.newaxis] * vxyz
    L = np.cross(xyz, p)
    p = p[rsort]
    L = L[rsort]
    m = m[rsort]
    mcumul = np.cumsum(m) / np.sum(m)
    Nfrac = np.argmin(np.abs(mcumul - frac))
    Nfrac = np.max([Nfrac, 100]) #use a minimum of 100 particles
    Nfrac = np.min([Nfrac, len(m)]) #unless this exceeds particle count
    p = p[:Nfrac]
    L = L[:Nfrac]
    Ltot = np.sqrt(np.sum(np.power(np.sum(L, axis=1), 2)))
    Lhat = np.sum(L, axis=1) / Ltot
    zhat = Laxis / np.sqrt(np.sum(np.power(Laxis, 2))) #normalized
    xaxis = np.array([1., 1., 1.]) if xaxis is None else np.array(xaxis) #default unlikely Laxis
    xhat = xaxis - xaxis.dot(zhat) * zhat
    xhat = xhat / np.sqrt(np.sum(np.power(xhat, 2))) #normalized
    yhat = np.cross(zhat, xhat) #guarantees right-handedness

    rotmat = np.vstack((xhat, yhat, zhat)) #units will be dropped (which is good)
    np.save(rotmat, saverot)
    return rotmat
