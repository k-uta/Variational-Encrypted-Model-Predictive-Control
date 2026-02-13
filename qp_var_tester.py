"""
Test variational MPC on a simple 2D QP problem.
Compare VEMPC library against CVXPY solution.
"""

import numpy as np
import cvxpy as cp

from vempc.core.mpc import MPCProblem
from vempc.core.constraints import ConstraintPenalty
from vempc.core.variational import VariationalMPC

# for visuals
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def simple_qp_test():
    """
    Test on a simple 2D QP matching your existing test.
    """
    
    np.random.seed(9)
    
    # QP parameters
    x0 = np.array([0.5, -0.3])
    P = np.array([[1.0, 0.2],
                  [0.2, 1.5]])
    S = np.array([[0.8, -0.4],
                  [-0.3, 0.6]])
    H = np.array([[2.0, 0.5],
                  [0.5, 3.0]])
    u_bar = np.array([-4.1, -2.1])
    u_bar_upper = np.array([-0.5, 0.5])
    
    print(f"x0 = {x0}")
    print(f"Constraints: {u_bar} <= U <= {u_bar_upper}")
    
    # cost function
    def J0(U, x0):
        return x0 @ P @ x0 + x0 @ S @ U + 0.5 * U @ H @ U
    
    # Baseline by CVXPY
    U_cvx = cp.Variable(2)
    objective = cp.Minimize(x0 @ P @ x0 + x0 @ S @ U_cvx + 0.5 * cp.quad_form(U_cvx, H))
    constraints = [U_cvx >= u_bar, U_cvx <= u_bar_upper]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    U_star = U_cvx.value
    
    print(f"\nCVXPY solution:")
    print(f"U_star = {U_star}")
    print(f"Cost = {J0(U_star, x0):.6f}")
    
    # --------------------
    # Testing VEMPC library
    # --------------------
    

    # artificial mpc problem for now
    n = 2  # state dimension
    m = 2  # control dimension
    N = 1  # horizon = 1
    
    # Dummy dynamics to instantiate MPC Problem
    A = np.eye(n)
    B = np.eye(m)
    Q = P  # state cost
    R = np.zeros((m, m))
    Qf = np.zeros((n, n))

    mpc = MPCProblem(A, B, Q, R, Qf, N)
    mpc.S = S
    mpc.H = H
    mpc.P = P
    
    # Setup constraints: u_bar <= U <= u_bar_upper
    # Reformulate as G U <= h
    # u_bar <= U -->  -U <= -u_bar
    # U <= u_bar_upper
    G = np.vstack([-np.eye(m), np.eye(m)])
    h_func = lambda x: np.hstack([-u_bar, u_bar_upper])
    
    penalty = ConstraintPenalty(G, h_func)
    
    print(f"  Constraints: {penalty.p} inequalities")
    
    # Create variational MPC
    lambda_param = 0.01
    Sigma_0 = 1.0 * np.eye(m)
    
    vmpc = VariationalMPC(mpc, penalty, lambda_param=lambda_param, Sigma_0=Sigma_0)
    
    print(f"    Lambda: {lambda_param}")
    print(f"    Sigma_0: {Sigma_0[0,0]} * I")
    
    # Estimate control
    K = 10000
    U_hat = vmpc.U_hat_K(x0, K, random_state=9)
    
    print(f"\nVEMPC solution (K={K}):")
    print(f"U_hat = {U_hat}")
    print(f"Cost = {J0(U_hat, x0):.6f}")
    
    # Check feasibility
    is_feas = penalty.is_feasible(U_hat, x0)
    print(f"    Feasible: {is_feas}")
    
    # Comparison
    error = np.linalg.norm(U_star - U_hat)
    print(f"\nError ||U_star - U_hat||: {error:.6f}")
    
    # ---------
    # Visualization
    # ---------
    
    u1 = np.linspace(-5, 1, 100)
    u2 = np.linspace(-3, 1, 100)
    U1, U2 = np.meshgrid(u1, u2)
    Z = np.zeros_like(U1)
    
    for i in range(len(u1)):
        for j in range(len(u2)):
            U_ij = np.array([U1[j, i], U2[j, i]])
            Z[j, i] = J0(U_ij, x0)
    
    fig = plt.figure(figsize=(14, 5))
    
    # 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(U1, U2, Z, cmap='viridis', alpha=0.7, edgecolor='none')
    
    ax1.scatter([U_star[0]], [U_star[1]], [J0(U_star, x0)], 
                color='red', s=100, marker='*', label='CVXPY U_star', zorder=5)
    ax1.scatter([U_hat[0]], [U_hat[1]], [J0(U_hat, x0)], 
                color='blue', s=100, marker='o', label='VEMPC U_hat', zorder=5)
    
    # Draw feasible region box
    z_box = np.min(Z) - 1
    box_x = [u_bar[0], u_bar_upper[0], u_bar_upper[0], u_bar[0], u_bar[0]]
    box_y = [u_bar[1], u_bar[1], u_bar_upper[1], u_bar_upper[1], u_bar[1]]
    ax1.plot(box_x, box_y, [z_box]*5, 'k-', linewidth=3, label='Feasible region')
    
    ax1.set_xlabel('U[0]')
    ax1.set_ylabel('U[1]')
    ax1.set_zlabel('J(U; x0)')
    ax1.set_title('Cost Landscape')
    ax1.legend()
    
    # 2D contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(U1, U2, Z, levels=25, cmap='viridis')
    
    # Feasible region
    ax2.plot(box_x, box_y, 'k-', linewidth=2, label='Feasible region')
    ax2.fill(box_x, box_y, alpha=0.2, color='green')
    
    # Solutions
    ax2.plot(U_star[0], U_star[1], 'r*', markersize=15, label='CVXPY U_star')
    ax2.plot(U_hat[0], U_hat[1], 'bo', markersize=10, label='VEMPC U_hat')
    
    ax2.set_xlabel('U[0]')
    ax2.set_ylabel('U[1]')
    ax2.set_title('Contour Plot')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    fig.colorbar(contour, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('simple_qp_test.png', dpi=150)
    print(f"\nPlot saved as: simple_qp_test.png")
    plt.show()
    
    print("\n\n\nTest complete!")


if __name__ == '__main__':
    simple_qp_test()