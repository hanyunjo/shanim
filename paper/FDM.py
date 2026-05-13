import numpy as np
from scipy.linalg import solve_banded

# 3. FDM
# 1) vanilla
# 1-1) BS
# 1-1-1) FTCS(만기->현재)
def FTCS_BS_vanilla(eta, type='call', S_max=4, dS=0.01, dt=0.0001):
    S0, K, r, sigma, T = eta

    # stability condition check
    var_stability = dS**2 / (sigma**2 * S_max**2)
    if dt > var_stability:
        print(f"경고 : 안정화 조건 위반 dt={dt:.6f} > {var_stability:.6f}")

    # grid
    S_grid = np.arange(0, S_max + dS, dS)  # 주가 그리드
    t_grid = np.arange(0, T + dt, dt)       # 시간 그리드
    N = len(S_grid)
    steps = len(t_grid)

    # init condition (maturity payoff)
    if type == 'call':
        value = np.maximum(S_grid - K, 0)
    elif type == 'put':
        value = np.maximum(K - S_grid, 0)

    # coefficient
    alpha = 0.5 * sigma**2 * S_grid**2 / dS**2 
    beta  = r * S_grid / (2 * dS)

    a = dt * (alpha - beta)         # 하삼각
    b = 1 - dt * (2 * alpha + r)    # 대각
    c = dt * (alpha + beta)         # 상삼각

    # Explicit FDM (maturity → present)
    for t in range(steps - 1):
        tau = t * dt
        value_new = value.copy()
        value_new[1:-1] = (a[1:-1] * value[:-2] + b[1:-1] * value[1:-1] + c[1:-1] * value[2:])

        # boundary condition
        if type == 'call':
            value_new[0]  = 0
            value_new[-1] = S_max - K * np.exp(-r * tau)
        elif type == 'put':
            value_new[0]  = K * np.exp(-r * tau)
            value_new[-1] = 0

        value = value_new

    # S0에 해당하는 인덱스 보간
    idx = S0 / dS
    i   = int(idx)
    w   = idx - i
    price = (1 - w) * value[i] + w * value[i + 1]
    return price


# 1-1-2) Crank-Nicolson
# th는 weight
def _CN_matrix_theta(sigma, r, S_grid, dt, dS, th=0.5):
    alpha = sigma**2 * S_grid**2 / (2 * dS**2)
    beta  = r * S_grid / (2 * dS)

    # left side (implicit)
    a_l = -th * dt * (alpha - beta)  # 하삼각
    b_l =  1 + th * dt * (2*alpha + r) # 대각
    c_l = -th * dt * (alpha + beta)  # 상삼각

    # right side (explicit)
    a_r =  (1-th) * dt * (alpha - beta)
    b_r =  1 - (1-th) * dt * (2*alpha + r)
    c_r =  (1-th) * dt * (alpha + beta)

    return a_l, b_l, c_l, a_r, b_r, c_r

def _solve_tridiagonal(a, b, c, r_in, N):
    ab = np.zeros((3, N))
    ab[0, 1:] = c[:-1]   # 상삼각
    ab[1, :]  = b         # 대각
    ab[2, :-1] = a[1:]   # 하삼각
    return solve_banded((1, 1), ab, r_in)

def CN_BS_vanilla(eta, type='call', S_max=4, dS=0.01, dt=0.01):
    S0, K, r, sigma, T = eta
    
    S_grid = np.arange(0, S_max + dS, dS)
    N = len(S_grid)
    steps = int(T / dt)

    # maturity payoff
    if type == 'call':
        value = np.maximum(S_grid - K, 0).copy()
    elif type == 'put':
        value = np.maximum(K - S_grid, 0).copy()

    a_l, b_l, c_l, a_r, b_r, c_r = _CN_matrix_theta(sigma, r, S_grid, dt, dS)

    for t in range(steps):
        tau = (t + 1) * dt

        # boundary condition
        if type == 'call':
            bc_low  = 0.0
            bc_high = S_max - K * np.exp(-r * tau)
        elif type == 'put':
            bc_low  = K * np.exp(-r * tau)
            bc_high = 0.0

        # right side calculating (except endpoints)
        r_side = np.zeros(N)
        r_side[0]  = bc_low
        r_side[-1] = bc_high
        r_side[1:-1] = (a_r[1:-1] * value[:-2]
                   + b_r[1:-1] * value[1:-1]
                   + c_r[1:-1] * value[2:])

        # tridiagonal matrix (except endpoints)
        a_in = a_l[1:-1].copy()
        b_in = b_l[1:-1].copy()
        c_in = c_l[1:-1].copy()
        r_in = r_side[1:-1].copy()

        # known value
        r_in[0]  -= a_in[0]  * bc_low
        r_in[-1] -= c_in[-1] * bc_high

        value_in = _solve_tridiagonal(a_in, b_in, c_in, r_in, N - 2)

        value[0]    = bc_low
        value[1:-1] = value_in
        value[-1]   = bc_high

    idx = S0 / dS
    i   = int(idx)
    w   = idx - i
    price = (1 - w) * value[i] + w * value[i + 1]
    return price




# 1-2) Heston
# Operator
# S term 
def _F0_heston(U, S, v, r, dS): 
    # F0 = (1/2)*v*S²*V_SS + r*S*V_S - r*V
    F = np.zeros_like(U)
    d2U_dS2 = (U[2:, 1:-1] - 2*U[1:-1, 1:-1] + U[:-2, 1:-1]) / dS**2
    dU_dS  = (U[2:, 1:-1] - U[:-2, 1:-1]) / (2*dS)
    F[1:-1, 1:-1] = (0.5 * v[None, 1:-1] * S[1:-1, None]**2 * d2U_dS2
                     + r * S[1:-1, None] * dU_dS
                     - r * U[1:-1, 1:-1])
    return F
 
 # v term 
def _F1_heston(U, v, kappa, theta, xi, dv):
    # F1 = (1/2)*ξ²*v*V_vv + κ(θ-v)*V_v
    F = np.zeros_like(U)
    d2U_dv2 = (U[1:-1, 2:] - 2*U[1:-1, 1:-1] + U[1:-1, :-2]) / dv**2
    dU_dv  = (U[1:-1, 2:] - U[1:-1, :-2]) / (2*dv)
    F[1:-1, 1:-1] = (0.5 * xi**2 * v[None, 1:-1] * d2U_dv2
                     + kappa * (theta - v[None, 1:-1]) * dU_dv)
    return F
 
 # mix term
def _F2_heston(U, S, v, rho, xi, dS, dv):
    # F2 = ρ*ξ*v*S*V_Sv
    F = np.zeros_like(U)
    d2U_dsdv = (U[2:, 2:] - U[2:, :-2] - U[:-2, 2:] + U[:-2, :-2]) / (4*dS*dv)
    F[1:-1, 1:-1] = rho * xi * v[None, 1:-1] * S[1:-1, None] * d2U_dsdv
    return F
 
 
def _solve_S_sweep(r_side, S_grid, v_grid, r, dS, dt, th, bc_low, bc_high, NS, Nv):
    # S 방향 implicit : (I - th*dt*A0) Y = r_side
    Y = r_side.copy()
    Y[0, :]  = bc_low
    Y[-1, :] = bc_high
 
    for j in range(1, Nv - 1):
        vj = v_grid[j]
        alpha = 0.5 * vj * S_grid**2 / dS**2
        beta  = r * S_grid / (2*dS)
 
        a_l = -th * dt * (alpha - beta)
        b_l =  1 + th * dt * (2*alpha + r)
        c_l = -th * dt * (alpha + beta)
 
        a_in = a_l[1:-1].copy()
        b_in = b_l[1:-1].copy()
        c_in = c_l[1:-1].copy()
        r_in = r_side[1:-1, j].copy()
 
        # known value
        r_in[0]  -= a_in[0]  * bc_low
        r_in[-1] -= c_in[-1] * bc_high
 
        Y[1:-1, j] = _solve_tridiagonal(a_in, b_in, c_in, r_in, NS - 2)
 
    return Y
 
 
def _solve_v_sweep(r_side, v_grid, kappa, theta, xi, dv, dt, th, NS, Nv):
    # v 방향 implicit : (I - th*dt*A1) Y = r_side
    Y = r_side.copy()
 
    for i in range(1, NS - 1):
        alpha = 0.5 * xi**2 * v_grid / dv**2
        beta  = kappa * (theta - v_grid) / (2*dv)
 
        a_l = -th * dt * (alpha - beta)
        b_l =  1 + th * dt * (2*alpha)
        c_l = -th * dt * (alpha + beta)
 
        a_in = a_l[1:-1].copy()
        b_in = b_l[1:-1].copy()
        c_in = c_l[1:-1].copy()
        r_in = r_side[i, 1:-1].copy()

        b_in[0] += a_in[0] # because of U_0 = U_1 unknown variable
        b_in[-1] += c_in[-1]

        Y[i, 1:-1] = _solve_tridiagonal(a_in, b_in, c_in, r_in, Nv - 2)
 
        Y[i, 0]  = Y[i, 1] #  U_v = 0
        Y[i, -1] = Y[i, -2]
 
    return Y
 
 
def CS_ADI_heston_vanilla(eta, type='call', S_max=4.0, v_max=1.5, dS=0.01, dv=0.001, dt=0.01):
    S0, K, r, kappa, theta, xi, rho, v0, T = eta

    S_grid = np.arange(0, S_max + dS, dS)
    v_grid = np.arange(0, v_max + dv, dv)
    dS = S_grid[1] - S_grid[0]
    dv = v_grid[1] - v_grid[0]
    NS = len(S_grid)
    Nv = len(v_grid)
    steps = int(T / dt)
    th = 0.5  # Crank-Nicolson
 
    # initial condition
    S2D = S_grid[:, None] * np.ones((1, Nv)) # None is for converting to a column vector
    if type == 'call':
        U = np.maximum(S2D - K, 0.0)
    elif type == 'put':
        U = np.maximum(K - S2D, 0.0)
 
    for t in range(steps):
        tau = (t + 1) * dt
 
        # boundary condition of U
        if type == 'call':
            bc_low  = 0.0
            bc_high = max(S_max - K * np.exp(-r * tau), 0.0)
        elif type == 'put':
            bc_low  = K * np.exp(-r * tau)
            bc_high = 0.0
 
        # operator
        F0_n = _F0_heston(U, S_grid, v_grid, r, dS)
        F1_n = _F1_heston(U, v_grid, kappa, theta, xi, dv)
        F2_n = _F2_heston(U, S_grid, v_grid, rho, xi, dS, dv)
 
        # Step 0: explicit predictor
        Y0 = U + dt * (F0_n + F1_n + F2_n)
 
        # Step 1: S implicit
        r_side1 = Y0 - th * dt * F0_n
        Y1 = _solve_S_sweep(r_side1, S_grid, v_grid, r, dS, dt, th, bc_low, bc_high, NS, Nv)
 
        # Step 2: v implicit
        r_side2 = Y1 - th * dt * F1_n
        Y2 = _solve_v_sweep(r_side2, v_grid, kappa, theta, xi, dv, dt, th, NS, Nv)
        Y2[0, :]  = bc_low
        Y2[-1, :] = bc_high
 
        # CS correction
        F2_Y2 = _F2_heston(Y2, S_grid, v_grid, rho, xi, dS, dv)
        Y0t = Y0 + 0.5 * dt * (F2_Y2 - F2_n)
 
        # Step 3: S implicit
        r_side3 = Y0t - th * dt * F0_n
        Y1t = _solve_S_sweep(r_side3, S_grid, v_grid, r, dS, dt, th, bc_low, bc_high, NS, Nv)
 
        # Step 4: v implicit
        r_side4 = Y1t - th * dt * F1_n
        U_new = _solve_v_sweep(r_side4, v_grid, kappa, theta, xi, dv, dt, th, NS, Nv)
        U_new[0, :]  = bc_low
        U_new[-1, :] = bc_high
 
        U = U_new
 
    idx = S0 / dS;  i = int(idx);  wi = idx - i
    jdx = v0 / dv;  j = int(jdx);  wj = jdx - j
    price = ((1-wi)*(1-wj) * U[i, j]   + wi*(1-wj) * U[i+1, j]
           + (1-wi)*wj     * U[i, j+1] + wi*wj     * U[i+1, j+1])
    return price














# 2) barrier
# 2-1) BS
# FTCS
def FTCS_BS_barrier(eta, type='call', B=0.8, S_max=4.0, dS=0.01, dt=0.0001):
    S0, K, r, sigma, T = eta

    # S <= B : knock-out
    S_grid = np.arange(B, S_max + dS, dS)
    steps = int(T / dt)
    N = len(S_grid)

    if type == 'call':
        V = np.maximum(S_grid - K, 0)
    elif type == 'put':
        V = np.maximum(K - S_grid, 0)


    alpha = 0.5 * sigma**2 * S_grid**2 / dS**2
    beta  = r * S_grid / (2 * dS)

    a = dt * (alpha - beta)
    b = 1 - dt * (2 * alpha + r)
    c = dt * (alpha + beta)

    # Explicit
    for t in range(steps):
        tau = (t + 1)  * dt
        V_new = V.copy()
        V_new[1:-1] = (a[1:-1] * V[:-2]
                     + b[1:-1] * V[1:-1]
                     + c[1:-1] * V[2:])

        # boundary condition
        V_new[0]  = 0   # S=B : knock-out
        if type == 'call':
            V_new[-1] = S_max - K * np.exp(-r * tau)
        elif type == 'put':
            V_new[-1] = 0

        V = V_new

    idx = (S0 - B) / dS
    i   = int(idx)
    w   = idx - i
    price = (1 - w) * V[i] + w * V[i + 1]
    return price



#  CN
def CN_BS_barrier(eta, type='call', B=0.8, S_max=4.0, dS=0.01, dt=0.01):
    S0, K, r, sigma, T = eta

    S_grid = np.arange(B, S_max + dS, dS)
    N = len(S_grid)
    steps = int(T / dt)

    # check payoff discountinuity, Rannacher: t_rannacher 스텝은 fully implicit
    has_discontinuity = (type == 'call' and K < B) or (type == 'put' and B < K)
    t_rannacher = 30 if has_discontinuity else 0

    # payoff
    if type == 'call':
        value = np.maximum(S_grid - K, 0).copy()
    elif type == 'put':
        value = np.maximum(K - S_grid, 0).copy()

    # CN matrix (θ=0.5)
    al, bl, cl, ar, br, cr = _CN_matrix_theta(sigma, r, S_grid, dt, dS, th=0.5)
    # BTCS matrix (θ=1)
    al1, bl1, cl1, ar1, br1, cr1 = _CN_matrix_theta(sigma, r, S_grid, dt, dS, th=1.0)

    for t in range(steps):
        tau = (t + 1) * dt

        if t < t_rannacher:
            a_l, b_l, c_l, a_r, b_r, c_r = al1, bl1, cl1, ar1, br1, cr1
        else:
            a_l, b_l, c_l, a_r, b_r, c_r = al, bl, cl, ar, br, cr

        r_side = np.zeros(N)
        r_side[1:-1] = (a_r[1:-1] * value[:-2]
                        + b_r[1:-1] * value[1:-1]
                        + c_r[1:-1] * value[2:])

        # boundary condition
        bc_low  = 0.0  # S=B: knock-out
        if type == 'call':
            bc_high = S_max - K * np.exp(-r * tau)
        elif type == 'put':
            bc_high = 0.0

        r_side[0]  = bc_low
        r_side[-1] = bc_high

        a_in = a_l[1:-1].copy()
        b_in = b_l[1:-1].copy()
        c_in = c_l[1:-1].copy()
        r_in = r_side[1:-1].copy()

        r_in[0]  -= a_in[0]  * bc_low
        r_in[-1] -= c_in[-1] * bc_high

        value_in = _solve_tridiagonal(a_in, b_in, c_in, r_in, N - 2)

        value[0]    = bc_low
        value[1:-1] = value_in
        value[-1]   = bc_high

    idx = (S0 - B) / dS
    i   = int(idx)
    w   = idx - i
    price = (1 - w) * value[i] + w * value[i + 1]
    return price




# 2-2) Heston
# CS
def CS_ADI_heston_barrier(eta, type='call', B=0.8, S_max=4.0, v_max=1.5, dS=0.01, dv=0.001, dt=0.01):
    S0, K, r, kappa, theta, xi, rho, v0, T = eta

    S_grid = np.arange(B, S_max + dS, dS)
    v_grid = np.arange(0, v_max + dv, dv)
    dS = S_grid[1] - S_grid[0]
    dv = v_grid[1] - v_grid[0]
    NS = len(S_grid)
    Nv = len(v_grid)
    steps = int(T / dt)
    th = 0.5
    
    has_discontinuity = (type == 'call' and K < B) or (type == 'put' and B < K)
    t_rannacher = 30 if has_discontinuity else 0
 
    S2D = S_grid[:, None] * np.ones((1, Nv))
    if type == 'call':
        U = np.maximum(S2D - K, 0.0)
    elif type == 'put':
        U = np.maximum(K - S2D, 0.0)
    # S_min = B : knock-out
    U[0, :] = 0.0
 
    for t in range(steps):
        tau = (t + 1) * dt
        th = 1.0 if t < t_rannacher else 0.5
 
        bc_low = 0.0  # knocked out
        if type == 'call':
            bc_high = max(S_max - K * np.exp(-r * tau), 0.0)
        elif type == 'put':
            bc_high = 0.0
 
        F0_n = _F0_heston(U, S_grid, v_grid, r, dS)
        F1_n = _F1_heston(U, v_grid, kappa, theta, xi, dv)
        F2_n = _F2_heston(U, S_grid, v_grid, rho, xi, dS, dv)
 
        Y0 = U + dt * (F0_n + F1_n + F2_n)
 
        r_side1 = Y0 - th * dt * F0_n
        Y1 = _solve_S_sweep(r_side1, S_grid, v_grid, r, dS, dt, th, bc_low, bc_high, NS, Nv)
 
        r_side2 = Y1 - th * dt * F1_n
        Y2 = _solve_v_sweep(r_side2, v_grid, kappa, theta, xi, dv, dt, th, NS, Nv)
        Y2[0, :]  = 0.0
        Y2[-1, :] = bc_high
 
        # correction
        F2_Y2 = _F2_heston(Y2, S_grid, v_grid, rho, xi, dS, dv)
        Y0t = Y0 + 0.5 * dt * (F2_Y2 - F2_n)
 
        r_side3 = Y0t - th * dt * F0_n
        Y1t = _solve_S_sweep(r_side3, S_grid, v_grid, r, dS, dt, th, bc_low, bc_high, NS, Nv)
 
        r_side4 = Y1t - th * dt * F1_n
        U_new = _solve_v_sweep(r_side4, v_grid, kappa, theta, xi, dv, dt, th, NS, Nv)
        U_new[0, :]  = 0.0
        U_new[-1, :] = bc_high
 
        U = U_new
 
    idx = (S0 - B) / dS;  i = int(idx);  wi = idx - i
    jdx = v0 / dv;        j = int(jdx);  wj = jdx - j
    price = ((1-wi)*(1-wj) * U[i, j]   + wi*(1-wj) * U[i+1, j]
           + (1-wi)*wj     * U[i, j+1] + wi*wj     * U[i+1, j+1])
    return price