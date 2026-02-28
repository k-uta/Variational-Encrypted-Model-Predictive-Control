"""
Discrete-time double integrator example.

State:  x = [position, velocity]
Input:  u = force (scalar)
"""

import numpy as np

# Discrete-time system matrices
A = np.array([[1.0, 1.0],
              [0.0, 1.0]])

B = np.array([[0.0],
              [1.0]])


# Cost matrices
Q  = np.eye(A.shape[0])
R  = 10.0 * np.eye(B.shape[1])
Qf = 10.0 * Q

# State constraints: |pos| <= p_max, |vel| <= v_max
p_max = 1.5
v_max = 0.5

# Input constraints: |u| <= u_max
u_max = 0.5

Gu = np.vstack([ np.eye(B.shape[1]),
                -np.eye(B.shape[1])])
hu = np.full(2 * B.shape[1], u_max)


# # No state constraints (for now)
# Gx = np.zeros((0, A.shape[0]))
# hx = np.zeros(0)

x_bound = np.array([p_max, v_max])
Gx = np.vstack([ np.eye(A.shape[0]),
                -np.eye(A.shape[0])])
hx = np.hstack([x_bound, x_bound])