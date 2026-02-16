import numpy as np
from scipy.signal import cont2discrete

def linearized_cartpole_continuous(M=1.0, m=0.1, l=0.5, g=9.81):
    """
    Linearized cart-pole dynamics about the upright equilibrium (theta ~ 0).
    State: x = [pos, vel, theta, theta_dot]
    Input: u = cart force

    This is a standard small-angle linearization.
    """
    n, mm = 4, 1
    Ac = np.zeros((n, n))
    Bc = np.zeros((n, mm))

    # xdot = v
    Ac[0, 1] = 1.0

    # vdot ≈ u/M + (m*g/M)*theta
    Ac[1, 2] = (m * g) / M
    Bc[1, 0] = 1.0 / M

    # thetadot = omega
    Ac[2, 3] = 1.0

    # omegadot ≈ -(1/(l*M)) u - ((M+m)g/(l*M)) theta
    Ac[3, 2] = -((M + m) * g) / (l * M)
    Bc[3, 0] = -1.0 / (l * M)

    return Ac, Bc

def discretize(Ac, Bc, dt):
    n = Ac.shape[0]
    mm = Bc.shape[1]
    C = np.eye(n)
    D = np.zeros((n, mm))
    Ad, Bd, _, _, _ = cont2discrete((Ac, Bc, C, D), dt)
    return Ad, Bd