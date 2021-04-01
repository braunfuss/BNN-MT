import numpy as np


def _mat(m):
    return np.array(([[m[0], m[3], m[4]],
                      [m[3], m[1], m[5]],
                      [m[4], m[5], m[2]]]))


def omega_angle(M1x, M2x):
    M1 = _mat(M1x)
    M2 = _mat(M2x)

    n = len(M1)
    cosom = np.zeros(n)

    M1r = M1
    M2r = M2
    omega = 0.5 * (1-((np.sum(M1r*M2r))/(np.sqrt(np.sum(M1r**2))*np.sqrt(np.sum((M2r**2))))))

    return omega
