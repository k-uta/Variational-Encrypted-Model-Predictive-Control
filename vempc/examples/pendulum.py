"""
Discrete-time linearized cart-pole (inverted pendulum) example.

State:  x = [cart_pos, cart_vel, theta, theta_dot]
         where theta is the pole angle from vertical (radians).
Input:  u = horizontal cart force (scalar)

Dynamics are linearized about the upright equilibrium (theta=0)
using a small-angle approximation.
"""

import numpy as np
from scipy.signal import cont2discrete

# ------------------------------------------------------------------
# Physical parameters
# ------------------------------------------------------------------
M   = 1.0    # cart mass (kg)
m_p = 0.1    # pole mass (kg)
l   = 0.5    # pole half-length (m)
g   = 9.81   # gravitational acceleration (m/s^2)
dt  = 0.1    # sampling period (s)

# ------------------------------------------------------------------
# Continuous-time linearization
# ------------------------------------------------------------------

def linearized_cartpole_continuous(M=M, m=m_p, l=l, g=g):
    """
    Linearized cart-pole dynamics about the upright equilibrium (theta ~ 0).
    State: x = [pos, vel, theta, theta_dot]
    Input: u = cart force
    """
    n, mm = 4, 1
    Ac = np.zeros((n, n))
    Bc = np.zeros((n, mm))

    # xdot = v
    Ac[0, 1] = 1.0

    # vdot = u/M + (m*g/M)*theta
    Ac[1, 2] = (m * g) / M
    Bc[1, 0] = 1.0 / M

    # thetadot = omega
    Ac[2, 3] = 1.0

    # omegadot = -(1/(l*M)) u - ((M+m)*g/(l*M)) theta
    Ac[3, 2] = -((M + m) * g) / (l * M)
    Bc[3, 0] = -1.0 / (l * M)

    return Ac, Bc


def discretize(Ac, Bc, dt=dt):
    n  = Ac.shape[0]
    mm = Bc.shape[1]
    C  = np.eye(n)
    D  = np.zeros((n, mm))
    Ad, Bd, _, _, _ = cont2discrete((Ac, Bc, C, D), dt)
    return Ad, Bd


# ------------------------------------------------------------------
# Discrete-time system matrices (module-level, used by test scripts)
# ------------------------------------------------------------------
Ac, Bc = linearized_cartpole_continuous()
A, B   = discretize(Ac, Bc)

# ------------------------------------------------------------------
# Cost matrices
# ------------------------------------------------------------------
# Angle (index 2) is penalized most heavily to promote stabilization.
Q  = np.diag([1.0, 0.1, 10.0, 1.0])
R  = 0.01 * np.eye(B.shape[1])
Qf = 10.0 * Q

# ------------------------------------------------------------------
# Constraint bounds (relaxed to widen the feasible region)
# ------------------------------------------------------------------
x_max     = 0.5    # cart position limit (m)
v_max     = 1.0    # cart velocity limit (m/s)
theta_max = 0.5    # pole angle limit (rad), ~28 deg; within linearization validity
omega_max = 1.0    # pole angular velocity limit (rad/s)
u_max     = 10.0   # cart force limit (N)

# State constraint matrices: Gx @ x <= hx
x_bound = np.array([x_max, v_max, theta_max, omega_max])
Gx = np.vstack([ np.eye(4), -np.eye(4)])
hx = np.hstack([x_bound, x_bound])

# Input constraint matrices: Gu @ u <= hu
Gu = np.vstack([ np.eye(B.shape[1]), -np.eye(B.shape[1])])
hu = np.full(2 * B.shape[1], u_max)