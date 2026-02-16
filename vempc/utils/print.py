import numpy as np

def print_setup(M, m, l, g, dt,
                    Ac, Bc, A, B,
                    N, T_steps,
                    Q, R, Qf,
                    x_max, v_max, theta_max, omega_max, u_max,
                    x_bound, Gx, hx, Gu, hu):
    W = 80
    np.set_printoptions(precision=6, suppress=True, linewidth=140)

    print("\n" + "="*W)
    print("SYSTEM PARAMETERS")
    print("="*W)
    print(f"M   = {M}")
    print(f"m   = {m}")
    print(f"l   = {l}")
    print(f"g   = {g}")
    print(f"dt  = {dt}")

    print("\n" + "="*W)
    print("MATRICES")
    print("="*W)
    print(f"A shape = {A.shape}")
    print(A)
    print(f"\nB shape = {B.shape}")
    print(B)
    print(f"\nn (states) = {A.shape[0]}, m_in (inputs) = {B.shape[1]}")

    print("\n" + "="*W)
    print("MPC SETUP")
    print("="*W)
    print(f"Horizon N        = {N}")
    print(f"Simulation steps = {T_steps}")

    print("\n" + "="*W)
    print("BOX BOUNDS")
    print("="*W)
    print(f"x_max     = {x_max}")
    print(f"v_max     = {v_max}")
    print(f"theta_max = {theta_max}  (rad)")
    print(f"omega_max = {omega_max}")
    print(f"u_max     = {u_max}")
    print("\nx_bound =")
    print(x_bound)

    print("\n" + "="*W)
    print("POLYHEDRAL CONSTRAINTS")
    print("="*W)
    print(f"Gx shape = {Gx.shape}, hx shape = {hx.shape}")
    print("Gx =")
    print(Gx)
    print("hx =")
    print(hx)

    print("\n" + "-"*W)
    print(f"Gu shape = {Gu.shape}, hu shape = {hu.shape}")
    print("Gu =")
    print(Gu)
    print("hu =")
    print(hu)


def print_condensed_mpc(P, S, H, Lambda, Psi, G, h_of_x0=None, x0=None):
    W = 80
    np.set_printoptions(precision=4, suppress=True, linewidth=140)

    print("\n" + "=" * W)
    print("CONDENSED MPC MATRICES")
    print("=" * W)

    print(f"P shape = {P.shape}")
    print(P)
    print(f"\nS shape = {S.shape}")
    print(S)
    print(f"\nH shape = {H.shape}")
    print(H)

    print(f"\nLambda shape = {Lambda.shape}")
    print(Lambda)
    print(f"\nPsi shape = {Psi.shape}")
    print(Psi)

    if G is not None:
        print("\n" + "-" * W)
        print(f"G shape = {G.shape}")
        print(G)

    if h_of_x0 is not None and x0 is not None:
        h = h_of_x0(x0)
        print(f"\nh(x0) shape = {h.shape}")
        print(h)    