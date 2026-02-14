"""
Visualize the tilting effect in variational MPC.
Shows prior kappa_0 vs tilted kappa_tilde sampling distributions.
Compare U_hat from both distributions against U_star.
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D

from vempc.core.mpc import MPCProblem
from vempc.core.constraints import ConstraintPenalty
from vempc.core.variational import VariationalMPC


def visualize_tilting():
    """
    Compare prior kappa_0 vs tilted kappa_tilde distributions.
    Compare U_hat from kappa_0 vs U_hat from tildekappa vs U_star.
    """
    
    print("Visualizing Tilting Effect: kappa_0 vs tildekappa")
    
    np.random.seed(9)
    
    # QP parameters
    x0 = np.array([-5.5, -3.5]) # should try multiple cases


    P = np.array([[1.0, 0.2],
                  [0.2, 1.5]])
    S = np.array([[0.8, -0.4],
                  [-0.3, 0.6]])
    H = np.array([[2.0, 0.5],
                  [0.5, 3.0]])
    
    # Easy case: unconstrained optimum near feasible region
    u_bar = np.array([-4.0, -2.0])
    u_bar_upper = np.array([-0.5, 0.5])
    
    print(f"x0 = {x0}")
    
    # Cost function
    def J0(U, x0):
        return x0 @ P @ x0 + x0 @ S @ U + 0.5 * U @ H @ U
    
    # CVXPY solution (ground truth)
    U_cvx = cp.Variable(2)
    objective = cp.Minimize(x0 @ P @ x0 + x0 @ S @ U_cvx + 0.5 * cp.quad_form(U_cvx, H))
    constraints = [U_cvx >= u_bar, U_cvx <= u_bar_upper]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    U_star = U_cvx.value
    
    print(f"\nCVXPY U_star = {U_star}")
    print(f"Cost(U_star) = {J0(U_star, x0):.6f}")
    
    # MPC problem
    n, m, N = 2, 2, 1
    A = np.eye(n)
    B = np.eye(m)
    Q = P
    R = np.zeros((m, m))
    Qf = np.zeros((n, n))
    
    mpc = MPCProblem(A, B, Q, R, Qf, N)
    mpc.S = S
    mpc.H = H
    mpc.P = P
    
    # Setup constraints
    G = np.vstack([-np.eye(m), np.eye(m)])
    h_func = lambda x: np.hstack([-u_bar, u_bar_upper])
    penalty = ConstraintPenalty(G, h_func)
    
    # Variational MPC
    lambda_param = 0.1
    Sigma_0 = 1.0 * np.eye(m)
    
    vmpc = VariationalMPC(mpc, penalty, lambda_param=lambda_param, Sigma_0=Sigma_0)
    
    print(f"\nlambda = {lambda_param}")
    print(f"Sigma_0 = {Sigma_0[0,0]} * I")
    
    # tilted distribution parameters
    m_U = vmpc.m_U(x0)
    Sigma_U = vmpc.Sigma_U
    
    print(f"\nkappa_0 = N(0, Sigma_0)")
    print(f"Mean: [0, 0]")
    
    print(f"\ntildekappa = N(m_U(x_0), Sigma_U)")
    print(f"m_U(x_0) = {m_U}")
    
    # Sample from both distributions
    K = 1000

    # U_hat from kappa_0 (initial reference)
    U_hat_prior = vmpc.U_hat_K(x0, K, random_state=9, reference='kappa_0')
    
    # U_hat from tildekappa (tilted)
    U_hat_tilted = vmpc.U_hat_K(x0, K, random_state=9, reference='tilde_kappa')
    
    # Check if estimates failed
    if U_hat_prior is None:
        print("\nWARNING: kappa_0 estimate failed (no feasible samples)")
        print("         Using a placeholder for visualization")
        U_hat_prior = np.array([np.nan, np.nan])

    if U_hat_tilted is None:
        print("\nWARNING: tildekappa estimate failed (no feasible samples)")
        print("         Using a placeholder for visualization")
        U_hat_tilted = np.array([np.nan, np.nan])
        
    # samples for visualization
    samples_prior = vmpc.sample_kappa_0(K, random_state=9)
    samples_tilted = vmpc.sample_kappa_tilde(x0, K, random_state=9)


    feas_rate_prior = penalty.feasibility_rate(samples_prior, x0)
    feas_rate_tilted = penalty.feasibility_rate(samples_tilted, x0)
    
    print(f"\nFeasibility rates:")
    print(f"kappa_0:    {feas_rate_prior:.2%}")
    print(f"tildekappa: {feas_rate_tilted:.2%}")
    if feas_rate_prior > 0:
        print(f"Ratio:        {feas_rate_tilted/feas_rate_prior:.2f}x")
    
    print(f"\nEstimated controls:")
    print(f"U_star (CVXPY): {U_star}")
    print(f"U_hat from kappa_0: {U_hat_prior}")
    print(f"U_hat from tildekappa:  {U_hat_tilted}")

    # Only compute costs and errors if estimates are valid
    if not np.any(np.isnan(U_hat_prior)):
        cost_prior = J0(U_hat_prior, x0)
        error_prior = np.linalg.norm(U_star - U_hat_prior)
        is_feas_prior = penalty.is_feasible(U_hat_prior, x0)
    else:
        cost_prior = np.nan
        error_prior = np.nan
        is_feas_prior = False

    if not np.any(np.isnan(U_hat_tilted)):
        cost_tilted = J0(U_hat_tilted, x0)
        error_tilted = np.linalg.norm(U_star - U_hat_tilted)
        is_feas_tilted = penalty.is_feasible(U_hat_tilted, x0)
    else:
        cost_tilted = np.nan
        error_tilted = np.nan
        is_feas_tilted = False

    print(f"\nCosts:")
    print(f"J(U_star):  {J0(U_star, x0):.6f}")
    print(f"J(U_hat from kappa_0):    {cost_prior if not np.isnan(cost_prior) else 'N/A (no estimate)'}")
    print(f"J(U_hat from tildekappa): {cost_tilted if not np.isnan(cost_tilted) else 'N/A (no estimate)'}")

    print(f"\nErrors:")
    print(f"||U_star - U_hat from kappa_0||:    {error_prior if not np.isnan(error_prior) else 'N/A'}")
    print(f"||U_star - U_hat from tildekappa||: {error_tilted if not np.isnan(error_tilted) else 'N/A'}")
    if not np.isnan(error_prior) and not np.isnan(error_tilted) and error_tilted > 1e-10:
        print(f"    Error reduction:    {error_prior/error_tilted:.2f}x")

    print(f"\nFeasibility:")
    print(f"U_hat from kappa_0 feasible?:    {is_feas_prior if not np.any(np.isnan(U_hat_prior)) else 'N/A'}")
    print(f"U_hat from tildekappa feasible?: {is_feas_tilted if not np.any(np.isnan(U_hat_tilted)) else 'N/A'}")

    
    
    # Cost landscape grid
    u1 = np.linspace(-5, 1, 100)
    u2 = np.linspace(-3, 1, 100)
    U1, U2 = np.meshgrid(u1, u2)
    Z = np.zeros_like(U1)
    
    for i in range(len(u1)):
        for j in range(len(u2)):
            U_ij = np.array([U1[j, i], U2[j, i]])
            Z[j, i] = J0(U_ij, x0)
    
    # Zoomed-in grid for feasible region
    margin = 0.1
    u1_zoom = np.linspace(u_bar[0] - margin, u_bar_upper[0] + margin, 80)
    u2_zoom = np.linspace(u_bar[1] - margin, u_bar_upper[1] + margin, 80)
    U1_zoom, U2_zoom = np.meshgrid(u1_zoom, u2_zoom)
    Z_zoom = np.zeros_like(U1_zoom)
    
    for i in range(len(u1_zoom)):
        for j in range(len(u2_zoom)):
            U_ij = np.array([U1_zoom[j, i], U2_zoom[j, i]])
            Z_zoom[j, i] = J0(U_ij, x0)
    
    # Feasible region box
    box_x = [u_bar[0], u_bar_upper[0], u_bar_upper[0], u_bar[0], u_bar[0]]
    box_y = [u_bar[1], u_bar[1], u_bar_upper[1], u_bar_upper[1], u_bar[1]]
    
    
    # visuals
    fig = plt.figure(figsize=(16, 14))
    
    # kappa_0 with cost landscape
    ax1 = fig.add_subplot(3, 3, 1, projection='3d')
    surf = ax1.plot_surface(U1, U2, Z, cmap='viridis', alpha=0.6, edgecolor='none')
    
    # Scatter prior samples
    z_samples = np.array([J0(samples_prior[i], x0) for i in range(min(500, K))])
    ax1.scatter(samples_prior[:500, 0], samples_prior[:500, 1], z_samples,
               alpha=0.2, s=1, color='gray')
    
    # Solutions
    ax1.scatter([U_star[0]], [U_star[1]], [J0(U_star, x0)],
               color='red', s=150, marker='*', label='U_star', zorder=5)
    ax1.scatter([U_hat_prior[0]], [U_hat_prior[1]], [J0(U_hat_prior, x0)],
               color='orange', s=100, marker='o', label='U_hat (kappa_0)', zorder=5)
    
    # Feasible region
    z_box = np.min(Z) - 1
    ax1.plot(box_x, box_y, [z_box]*5, 'k-', linewidth=3)
    
    ax1.set_xlabel('U[0]')
    ax1.set_ylabel('U[1]')
    ax1.set_zlabel('J(U; x0)')
    ax1.set_title('Prior kappa_0 samples', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8)
    
    # tildekappa with cost landscape
    ax2 = fig.add_subplot(3, 3, 2, projection='3d')
    surf = ax2.plot_surface(U1, U2, Z, cmap='viridis', alpha=0.6, edgecolor='none')
    
    # Scatter tilted samples
    z_samples_tilted = np.array([J0(samples_tilted[i], x0) for i in range(min(500, K))])
    ax2.scatter(samples_tilted[:500, 0], samples_tilted[:500, 1], z_samples_tilted,
               alpha=0.2, s=1, color='blue')
    
    # Solutions
    ax2.scatter([U_star[0]], [U_star[1]], [J0(U_star, x0)],
               color='red', s=150, marker='*', label='U_star', zorder=5)
    ax2.scatter([U_hat_tilted[0]], [U_hat_tilted[1]], [J0(U_hat_tilted, x0)],
               color='magenta', s=100, marker='o', label='U_hat (tildekappa)', zorder=5)
    
    # Feasible region
    ax2.plot(box_x, box_y, [z_box]*5, 'k-', linewidth=3)
    
    ax2.set_xlabel('U[0]')
    ax2.set_ylabel('U[1]')
    ax2.set_zlabel('J(U; x0)')
    ax2.set_title('Tilted tildekappa samples', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)
    
    # Combined 3D view
    ax3 = fig.add_subplot(3, 3, 3, projection='3d')
    surf = ax3.plot_surface(U1, U2, Z, cmap='viridis', alpha=0.5, edgecolor='none')
    
    # Solutions
    ax3.scatter([U_star[0]], [U_star[1]], [J0(U_star, x0)],
               color='red', s=150, marker='*', label='U_star', zorder=5)
    ax3.scatter([U_hat_prior[0]], [U_hat_prior[1]], [J0(U_hat_prior, x0)],
               color='orange', s=100, marker='o', label='U_hat (kappa_0)', zorder=5)
    ax3.scatter([U_hat_tilted[0]], [U_hat_tilted[1]], [J0(U_hat_tilted, x0)],
               color='magenta', s=100, marker='s', label='U_hat (tildekappa)', zorder=5)
    
    # Feasible region
    ax3.plot(box_x, box_y, [z_box]*5, 'k-', linewidth=3, label='Feasible')
    
    ax3.set_xlabel('U[0]')
    ax3.set_ylabel('U[1]')
    ax3.set_zlabel('J(U; x0)')
    ax3.set_title('Cost Landscape Comparison', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8)



    ax4 = fig.add_subplot(3, 3, 4)
    contour = ax4.contour(U1, U2, Z, levels=25, cmap='viridis')
    
    # Prior samples
    ax4.scatter(samples_prior[:500, 0], samples_prior[:500, 1],
               alpha=0.3, s=5, color='gray', label='kappa_0 samples')
    
    # Mean
    ax4.plot(0, 0, 'kx', markersize=15, label='kappa_0 mean', zorder=5)
    
    # Covariance ellipse
    eigenvalues, eigenvectors = np.linalg.eig(Sigma_0)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * 2 * np.sqrt(eigenvalues)
    ellipse = Ellipse((0, 0), width, height, angle=angle,
                     facecolor='none', edgecolor='black', linewidth=2,
                     linestyle='--', label='2sig')
    ax4.add_patch(ellipse)
    
    # Feasible region
    ax4.plot(box_x, box_y, 'k-', linewidth=2)
    ax4.fill(box_x, box_y, alpha=0.1, color='green')
    
    # Solutions
    ax4.plot(U_star[0], U_star[1], 'r*', markersize=15, label='U_star', zorder=5)
    ax4.plot(U_hat_prior[0], U_hat_prior[1], 'o', color='orange', markersize=10, 
             label='U_hat (kappa_0)', zorder=5)
    
    ax4.set_xlabel('U[0]')
    ax4.set_ylabel('U[1]')
    ax4.set_title('Prior kappa_0', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    ax4.set_xlim(-6, 2)
    ax4.set_ylim(-4, 2)
    
    # Plot 5: tildekappa contour
    ax5 = fig.add_subplot(3, 3, 5)
    contour = ax5.contour(U1, U2, Z, levels=25, cmap='viridis')
    
    # Tilted samples
    ax5.scatter(samples_tilted[:500, 0], samples_tilted[:500, 1],
               alpha=0.3, s=5, color='blue', label='tildekappa samples')
    
    # Mean
    ax5.plot(m_U[0], m_U[1], 'rx', markersize=15, label='m_U', zorder=5)
    
    # Covariance ellipse
    eigenvalues, eigenvectors = np.linalg.eig(Sigma_U)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * 2 * np.sqrt(eigenvalues)
    ellipse = Ellipse(m_U, width, height, angle=angle,
                     facecolor='none', edgecolor='red', linewidth=2,
                     linestyle='--', label='2sig')
    ax5.add_patch(ellipse)
    
    # Feasible region
    ax5.plot(box_x, box_y, 'k-', linewidth=2)
    ax5.fill(box_x, box_y, alpha=0.1, color='green')
    
    # Solutions
    ax5.plot(U_star[0], U_star[1], 'r*', markersize=15, label='U_star', zorder=5)
    ax5.plot(U_hat_tilted[0], U_hat_tilted[1], 's', color='magenta', markersize=10, 
             label='U_hat (tildekappa)', zorder=5)
    
    ax5.set_xlabel('U[0]')
    ax5.set_ylabel('U[1]')
    ax5.set_title('Tilted tildekappa', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.axis('equal')
    ax5.set_xlim(-6, 2)
    ax5.set_ylim(-4, 2)
    
    # Plot 6: Combined contour
    ax6 = fig.add_subplot(3, 3, 6)
    contour = ax6.contour(U1, U2, Z, levels=25, cmap='viridis')
    
    # Feasible region
    ax6.plot(box_x, box_y, 'k-', linewidth=2, label='Feasible')
    ax6.fill(box_x, box_y, alpha=0.1, color='green')
    
    # Solutions - ALL THREE
    ax6.plot(U_star[0], U_star[1], 'r*', markersize=15, label='U_star', zorder=5)
    ax6.plot(U_hat_prior[0], U_hat_prior[1], 'o', color='orange', markersize=12, 
             label='U_hat (kappa_0)', zorder=5)
    ax6.plot(U_hat_tilted[0], U_hat_tilted[1], 's', color='magenta', markersize=12, 
             label='U_hat (tildekappa)', zorder=5)
    
    ax6.set_xlabel('U[0]')
    ax6.set_ylabel('U[1]')
    ax6.set_title('Comparison: Full View', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.axis('equal')
    fig.colorbar(contour, ax=ax6)
    
    # ------------------------------
    # Row 3: ZOOMED 2D contours (feasible region focus)
    # ------------------------------
    
    # Plot 7: kappa_0 zoomed
    ax7 = fig.add_subplot(3, 3, 7)
    contour_zoom = ax7.contour(U1_zoom, U2_zoom, Z_zoom, levels=20, cmap='viridis')
    
    # Only show samples near feasible region
    mask_prior = (samples_prior[:, 0] >= u_bar[0] - margin) & \
                 (samples_prior[:, 0] <= u_bar_upper[0] + margin) & \
                 (samples_prior[:, 1] >= u_bar[1] - margin) & \
                 (samples_prior[:, 1] <= u_bar_upper[1] + margin)
    ax7.scatter(samples_prior[mask_prior, 0], samples_prior[mask_prior, 1],
               alpha=0.4, s=10, color='gray', label='kappa_0 samples')
    
    # Mean (if visible)
    if (0 >= u_bar[0] - margin) and (0 <= u_bar_upper[0] + margin):
        ax7.plot(0, 0, 'kx', markersize=12, label='kappa_0 mean', zorder=5)
    
    # Feasible region
    ax7.plot(box_x, box_y, 'k-', linewidth=3)
    ax7.fill(box_x, box_y, alpha=0.15, color='green', label='Feasible')
    
    # Solutions
    ax7.plot(U_star[0], U_star[1], 'r*', markersize=20, label='U_star', zorder=5)
    ax7.plot(U_hat_prior[0], U_hat_prior[1], 'o', color='orange', markersize=14, 
             label='U_hat (kappa_0)', zorder=5)
    
    ax7.set_xlabel('U[0]')
    ax7.set_ylabel('U[1]')
    ax7.set_title('Prior kappa_0 (ZOOMED)', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)
    ax7.axis('equal')
    fig.colorbar(contour_zoom, ax=ax7)
    
    # Plot 8: tildekappa zoomed
    ax8 = fig.add_subplot(3, 3, 8)
    contour_zoom = ax8.contour(U1_zoom, U2_zoom, Z_zoom, levels=20, cmap='viridis')
    
    # Only show samples near feasible region
    mask_tilted = (samples_tilted[:, 0] >= u_bar[0] - margin) & \
                  (samples_tilted[:, 0] <= u_bar_upper[0] + margin) & \
                  (samples_tilted[:, 1] >= u_bar[1] - margin) & \
                  (samples_tilted[:, 1] <= u_bar_upper[1] + margin)
    ax8.scatter(samples_tilted[mask_tilted, 0], samples_tilted[mask_tilted, 1],
               alpha=0.4, s=10, color='blue', label='tildekappa samples')
    
    # Mean
    ax8.plot(m_U[0], m_U[1], 'rx', markersize=12, label='m_U', zorder=5)
    
    # Covariance ellipse (zoomed)
    eigenvalues, eigenvectors = np.linalg.eig(Sigma_U)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * 2 * np.sqrt(eigenvalues)
    ellipse = Ellipse(m_U, width, height, angle=angle,
                     facecolor='none', edgecolor='red', linewidth=2,
                     linestyle='--', label='2sig')
    ax8.add_patch(ellipse)
    
    # Feasible region
    ax8.plot(box_x, box_y, 'k-', linewidth=3)
    ax8.fill(box_x, box_y, alpha=0.15, color='green', label='Feasible')
    
    # Solutions
    ax8.plot(U_star[0], U_star[1], 'r*', markersize=20, label='U_star', zorder=5)
    ax8.plot(U_hat_tilted[0], U_hat_tilted[1], 's', color='magenta', markersize=14, 
             label='U_hat (tildekappa)', zorder=5)
    
    ax8.set_xlabel('U[0]')
    ax8.set_ylabel('U[1]')
    ax8.set_title('Tilted tildekappa (ZOOMED)', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)
    ax8.axis('equal')
    fig.colorbar(contour_zoom, ax=ax8)
    
    # Plot 9: Combined zoomed
    ax9 = fig.add_subplot(3, 3, 9)
    contour_zoom = ax9.contour(U1_zoom, U2_zoom, Z_zoom, levels=20, cmap='viridis')
    
    # Feasible region
    ax9.plot(box_x, box_y, 'k-', linewidth=3, label='Feasible')
    ax9.fill(box_x, box_y, alpha=0.15, color='green')
    
    # m_U
    ax9.plot(m_U[0], m_U[1], 'rx', markersize=12, label='m_U', zorder=4)
    
    ax9.plot(U_star[0], U_star[1], 'r*', markersize=20, label='U_star', zorder=5)
    ax9.plot(U_hat_prior[0], U_hat_prior[1], 'o', color='orange', markersize=16, 
             label='U_hat (kappa_0)', zorder=5)
    ax9.plot(U_hat_tilted[0], U_hat_tilted[1], 's', color='magenta', markersize=16, 
             label='U_hat (tildekappa)', zorder=5)
    
    ax9.set_xlabel('U[0]')
    ax9.set_ylabel('U[1]')
    ax9.set_title('Comparison (ZOOMED)', fontsize=12, fontweight='bold')
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)
    ax9.axis('equal')
    fig.colorbar(contour_zoom, ax=ax9)
    
    plt.tight_layout()
    plt.savefig('qp_var_tester_tilt.png', dpi=150)
    print(f"\nPlot saved as: qp_var_tester_tilt.png")
    plt.show()

if __name__ == '__main__':
    visualize_tilting()