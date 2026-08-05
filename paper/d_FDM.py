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
 
 
def CS_ADI_heston_vanilla(
    eta, type='call', S_max=4.0, v_max=1.5, dS=0.01, dv=0.001,
    dt=0.01, return_surface=False, snapshot_steps=(),
):
    S0, K, r, kappa, theta, xi, rho, v0, T = eta

    S_grid = np.arange(0, S_max + dS, dS)
    v_grid = np.arange(0, v_max + dv, dv)
    dS = S_grid[1] - S_grid[0]
    dv = v_grid[1] - v_grid[0]
    NS = len(S_grid)
    Nv = len(v_grid)
    steps = int(round(T / dt))
    if steps <= 0 or not np.isclose(steps * dt, T, rtol=0.0, atol=1e-10):
        raise ValueError("T must be a positive integer multiple of dt.")
    th = 0.5  # Crank-Nicolson
 
    # initial condition
    S2D = S_grid[:, None] * np.ones((1, Nv)) # None is for converting to a column vector
    if type == 'call':
        U = np.maximum(S2D - K, 0.0)
    elif type == 'put':
        U = np.maximum(K - S2D, 0.0)
 
    snapshot_steps = {int(step) for step in snapshot_steps}
    if any(step < 0 or step > steps for step in snapshot_steps):
        raise ValueError('snapshot_steps must lie between 0 and the final time step.')
    snapshots = {0: U.copy()} if 0 in snapshot_steps else {}

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
        if t + 1 in snapshot_steps:
            snapshots[t + 1] = U.copy() # 만기 T까지의 결과

    if return_surface:
        snapshots[steps] = U
        return snapshots, S_grid, v_grid

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
    t_rannacher = 4 if has_discontinuity else 0

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
def CS_ADI_heston_barrier(
    eta, type='call', B=0.8, S_max=4.0, v_max=1.5, dS=0.01, dv=0.001,
    dt=0.01, return_surface=False, snapshot_steps=(),
):
    S0, K, r, kappa, theta, xi, rho, v0, T = eta

    S_grid = np.arange(B, S_max + dS, dS)
    v_grid = np.arange(0, v_max + dv, dv)
    dS = S_grid[1] - S_grid[0]
    dv = v_grid[1] - v_grid[0]
    NS = len(S_grid)
    Nv = len(v_grid)
    steps = int(round(T / dt))
    if steps <= 0 or not np.isclose(steps * dt, T, rtol=0.0, atol=1e-10):
        raise ValueError("T must be a positive integer multiple of dt.")
    th = 0.5
    
    has_discontinuity = (type == 'call' and K < B) or (type == 'put' and B < K)
    t_rannacher = 4 if has_discontinuity else 0
 
    S2D = S_grid[:, None] * np.ones((1, Nv))
    if type == 'call':
        U = np.maximum(S2D - K, 0.0)
    elif type == 'put':
        U = np.maximum(K - S2D, 0.0)
    # S_min = B : knock-out
    U[0, :] = 0.0
 
    snapshot_steps = {int(step) for step in snapshot_steps}
    if any(step < 0 or step > steps for step in snapshot_steps):
        raise ValueError('snapshot_steps must lie between 0 and the final time step.')
    snapshots = {0: U.copy()} if 0 in snapshot_steps else {}

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
        if t + 1 in snapshot_steps:
            snapshots[t + 1] = U.copy()

    if return_surface:
        snapshots[steps] = U
        return snapshots, S_grid, v_grid

    idx = (S0 - B) / dS;  i = int(idx);  wi = idx - i
    jdx = v0 / dv;        j = int(jdx);  wj = jdx - j
    price = ((1-wi)*(1-wj) * U[i, j]   + wi*(1-wj) * U[i+1, j]
           + (1-wi)*wj     * U[i, j+1] + wi*wj     * U[i+1, j+1])
    return price












# =========================================================
# Greeks: FDM bump-and-revalue finite differences
# =========================================================
def _fdm_check_opt_type(opt_type):
    if opt_type not in ("call", "put"):
        raise ValueError("opt_type must be 'call' or 'put'.")

def _fdm_aligned_step(raw_step, grid_step):
    if raw_step <= 0 or grid_step <= 0:
        raise ValueError("raw_step and grid_step must be positive.")
    return max(1, int(round(raw_step / grid_step))) * grid_step

def _fdm_default_bs_steps(eta, relative_step=0.01, dS=0.01, dt=0.01):
    S0, _, r, sigma, T = map(float, eta)
    return {
        "S0": _fdm_aligned_step(max(abs(S0) * relative_step, dS), dS),
        "r": max(abs(r) * relative_step, 1e-5),
        "sigma": max(abs(sigma) * relative_step, 1e-5),
        "T": _fdm_aligned_step(max(abs(T) * relative_step, dt), dt),
    }

def _fdm_default_heston_steps(eta, relative_step=0.01, dS=0.01, dv=0.001, dt=0.01):
    S0, _, r, _, _, _, _, v0, T = map(float, eta)
    return {
        "S0": _fdm_aligned_step(max(abs(S0) * relative_step, dS), dS),
        "r": max(abs(r) * relative_step, 1e-5),
        "v0": _fdm_aligned_step(max(abs(v0) * relative_step, dv), dv),
        "T": _fdm_aligned_step(max(abs(T) * relative_step, dt), dt),
    }

def _fdm_first_derivative(price, eta_base, index, h, is_valid):
    eta_up = eta_base.copy()
    eta_up[index] += h
    if not is_valid(eta_up):
        raise ValueError(
            "No valid forward bump is available. Reduce h or check parameter bounds."
        )

    base_price = price(eta_base)
    price_up = price(eta_up)
    return (price_up - base_price) / h, {
        "scheme": "forward",
        "base_price": base_price,
        "price_up": price_up,
        "h": h,
    }


def _fdm_bilinear_price(surface, S_grid, v_grid, S0, v0):
    i = int(np.searchsorted(S_grid, S0, side="right") - 1)
    j = int(np.searchsorted(v_grid, v0, side="right") - 1)
    i = int(np.clip(i, 0, len(S_grid) - 2))
    j = int(np.clip(j, 0, len(v_grid) - 2))
    wi = (S0 - S_grid[i]) / (S_grid[i + 1] - S_grid[i])
    wj = (v0 - v_grid[j]) / (v_grid[j + 1] - v_grid[j])
    return float(
        (1 - wi) * (1 - wj) * surface[i, j]
        + wi * (1 - wj) * surface[i + 1, j]
        + (1 - wi) * wj * surface[i, j + 1]
        + wi * wj * surface[i + 1, j + 1]
    )


def FDM_BS_greeks(
    eta,
    B=None,
    opt_type="call",
    greeks=("delta", "vega", "rho", "theta"),
    h=None,
    relative_step=0.01,
    solver_kwargs=None,
):
    _fdm_check_opt_type(opt_type)
    eta_base = np.asarray(eta, dtype=float)
    if eta_base.shape != (5,):
        raise ValueError("BS eta must be (S0, K, r, sigma, T).")

    kwargs = dict(solver_kwargs or {})
    price_fn = CN_BS_vanilla if B is None else CN_BS_barrier
    default_dt = 0.001

    dS = float(kwargs.get("dS", 0.01))
    dt = float(kwargs.get("dt", default_dt))
    S_max = float(kwargs.get("S_max", 4.0))
    steps = _fdm_default_bs_steps(eta_base, relative_step, dS, dt)
    if h is not None:
        steps.update({key: float(value) for key, value in h.items()})
    steps["S0"] = _fdm_aligned_step(steps["S0"], dS)
    steps["T"] = _fdm_aligned_step(steps["T"], dt)

    def is_valid(current_eta):
        S0, _, _, sigma, T = current_eta
        s_low = B if B is not None else 0.0
        return S0 > s_low and S0 < S_max and sigma > 0.0 and T > 0.0

    price_cache = {}

    def price(current_eta):
        key = tuple(np.round(current_eta, 14))
        if key not in price_cache:
            if B is None:
                price_cache[key] = float(price_fn(current_eta, type=opt_type, **kwargs))
            else:
                price_cache[key] = float(price_fn(current_eta, type=opt_type, B=B, **kwargs))
        return price_cache[key]

    mapping = {
        "delta" : (0, "S0", 1.0),
        "rho"   : (2, "r", 1.0),
        "vega"  : (3, "sigma", 1.0),
        "theta" : (4, "T", -1.0),
    }
    unknown = set(greeks) - set(mapping)
    if unknown:
        raise ValueError(f"Unknown BS Greeks: {sorted(unknown)}")

    result = {"price": price(eta_base), "h": steps, "details": {}}
    for greek in greeks:
        index, step_name, sign = mapping[greek]
        value, detail = _fdm_first_derivative(
            price, eta_base, index, steps[step_name], is_valid
        )
        result[greek] = sign * value
        result["details"][greek] = detail
    return result


def FDM_heston_greeks(
    eta,
    B=None,
    opt_type="call",
    greeks=("delta", "rho", "v0", "theta"),
    h=None,
    relative_step=0.01,
    solver_kwargs=None,
):
    _fdm_check_opt_type(opt_type)
    eta_base = np.asarray(eta, dtype=float)
    if eta_base.shape != (9,):
        raise ValueError("Heston eta must be (S0, K, r, kappa, theta, xi, rho, v0, T).")

    kwargs = dict(solver_kwargs or {})
    kwargs.pop("return_surface", None)
    kwargs.pop("snapshot_steps", None)
    dS = float(kwargs.get("dS", 0.01))
    dv = float(kwargs.get("dv", 0.001))
    dt = float(kwargs.get("dt", 0.01))
    S_max = float(kwargs.get("S_max", 4.0))
    v_max = float(kwargs.get("v_max", 1.5))
    steps = _fdm_default_heston_steps(eta_base, relative_step, dS, dv, dt)
    if h is not None:
        unknown_h = set(h) - set(steps)
        if unknown_h:
            raise ValueError(f"Unknown Heston bump variables: {sorted(unknown_h)}")
        steps.update({key: float(value) for key, value in h.items()})
    steps["S0"] = _fdm_aligned_step(steps["S0"], dS)
    steps["v0"] = _fdm_aligned_step(steps["v0"], dv)
    steps["T"] = _fdm_aligned_step(steps["T"], dt)

    price_fn = CS_ADI_heston_vanilla if B is None else CS_ADI_heston_barrier

    def is_valid(current_eta):
        S0, _, _, kappa, theta, xi, rho, v0, T = current_eta
        s_low = B if B is not None else 0.0
        return (
            S0 > s_low
            and S0 < S_max
            and kappa > 0.0
            and theta > 0.0
            and xi > 0.0
            and -1.0 < rho < 1.0
            and 0.0 <= v0 < v_max
            and T > 0.0
        )

    if not is_valid(eta_base):
        raise ValueError("Base Heston parameters are outside the FDM domain.")

    S0, _, _, _, _, _, _, v0, T = eta_base
    eta_extended = eta_base.copy()
    eta_extended[8] += steps["T"]
    base_steps = int(round(T / dt))
    final_steps = int(round(eta_extended[8] / dt))

    print(f"Calculating base")
    if B is None:
        snapshots, S_grid, v_grid = price_fn(
            eta_extended,
            type=opt_type,
            return_surface=True,
            snapshot_steps=(base_steps,),
            **kwargs,
        )
    else:
        snapshots, S_grid, v_grid = price_fn(
            eta_extended,
            type=opt_type,
            B=B,
            return_surface=True,
            snapshot_steps=(base_steps,),
            **kwargs,
        )
    print(f"Calculating base finish")

    surface_T = snapshots[base_steps]
    surface_T_plus = snapshots[final_steps]
    base_price = _fdm_bilinear_price(surface_T, S_grid, v_grid, S0, v0)

    mapping = {
        "delta": (0, "S0", 1.0),
        "rho": (2, "r", 1.0),
        "v0": (7, "v0", 1.0),
        "theta": (8, "T", -1.0),
    }
    requested = tuple(greeks)
    unknown = set(requested) - set(mapping)
    if unknown:
        raise ValueError(f"Unknown Heston sensitivities: {sorted(unknown)}")

    result = {"price": base_price, "h": steps, "details": {}}
    for greek in requested:
        index, step_name, sign = mapping[greek]
        step = steps[step_name]

        if greek == "delta":
            price_up = _fdm_bilinear_price(
                surface_T, S_grid, v_grid, S0 + step, v0
            )
        elif greek == "v0":
            price_up = _fdm_bilinear_price(
                surface_T, S_grid, v_grid, S0, v0 + step
            )
        elif greek == "theta":
            price_up = _fdm_bilinear_price(
                surface_T_plus, S_grid, v_grid, S0, v0
            )
        else:
            print(f"Calculating {greek}")
            eta_up = eta_base.copy()
            eta_up[index] += step
            if not is_valid(eta_up):
                raise ValueError(
                    "No valid forward bump is available. Reduce h or check parameter bounds."
                )
            if B is None:
                price_up = float(price_fn(eta_up, type=opt_type, **kwargs))
            else:
                price_up = float(price_fn(eta_up, type=opt_type, B=B, **kwargs))

        result[greek] = float(sign * (price_up - base_price) / step)
        result["details"][greek] = {
            "scheme": "forward",
            "base_price": base_price,
            "price_up": price_up,
            "h": step,
        }
    return result