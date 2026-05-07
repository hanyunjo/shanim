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
    M = len(t_grid)

    # init condition (maturity payoff)
    if type == 'call':
        V = np.maximum(S_grid - K, 0)
    elif type == 'put':
        V = np.maximum(K - S_grid, 0)

    # coefficient
    alpha = 0.5 * sigma**2 * S_grid**2 / dS**2 
    beta  = r * S_grid / (2 * dS)

    a = dt * (alpha - beta)         # 하삼각
    b = 1 - dt * (2 * alpha + r)    # 대각
    c = dt * (alpha + beta)         # 상삼각

    # Explicit FDM (maturity → present)
    for t in range(M - 1):
        tau = t * dt
        V_new = V.copy()
        V_new[1:-1] = (a[1:-1] * V[:-2] + b[1:-1] * V[1:-1] + c[1:-1] * V[2:])

        # boundary condition
        if type == 'call':
            V_new[0]  = 0
            V_new[-1] = S_max - K * np.exp(-r * tau)
        elif type == 'put':
            V_new[0]  = K * np.exp(-r * tau)
            V_new[-1] = 0

        V = V_new

    # S0에 해당하는 인덱스 보간
    idx = S0 / dS
    i   = int(idx)
    w   = idx - i
    price = (1 - w) * V[i] + w * V[i + 1]
    return price


# 1-1-2) Crank-Nicolson
def _CN_matrix(sigma, r, S_grid, dt, dS):
    alpha = sigma**2 * S_grid**2 / (2 * dS**2)
    beta  = r * S_grid / (2 * dS)

    # left side (implicit)
    a_l = -dt/2 * (alpha - beta)
    b_l =  1 + dt/2 * (2*alpha + r)
    c_l = -dt/2 * (alpha + beta)

    # right side (explicit)
    a_r =  dt/2 * (alpha - beta)
    b_r =  1 - dt/2 * (2*alpha + r)
    c_r =  dt/2 * (alpha + beta)

    return a_l, b_l, c_l, a_r, b_r, c_r

def _solve_tridiagonal(a, b, c, r_in, N):
    # scipy solve_banded 형식: (2, 1, 0) 행
    ab = np.zeros((3, N))
    ab[0, 1:] = c[:-1]   # 상삼각
    ab[1, :]  = b         # 대각
    ab[2, :-1] = a[1:]   # 하삼각
    return solve_banded((1, 1), ab, r_in)

def CN_BS_vanilla(eta, type='call', S_max=4, dS=0.01, dt=0.01):
    S0, K, r, sigma, T = eta
    
    S_grid = np.arange(0, S_max + dS, dS)
    N = len(S_grid)
    M = int(T / dt)

    # maturity payoff
    if type == 'call':
        V = np.maximum(S_grid - K, 0).copy()
    elif type == 'put':
        V = np.maximum(K - S_grid, 0).copy()

    a_l, b_l, c_l, a_r, b_r, c_r = _CN_matrix(sigma, r, S_grid, dt, dS)

    for t in range(M):
        tau = (t + 1) * dt

        # right side calculating (except endpoints)
        r_side = np.zeros(N)
        r_side[0]  = bc_low
        r_side[-1] = bc_high

        r_side[1:-1] = (a_r[1:-1] * V[:-2]
                   + b_r[1:-1] * V[1:-1]
                   + c_r[1:-1] * V[2:])

        # boundary condition
        if type == 'call':
            bc_low  = 0.0
            bc_high = S_max - K * np.exp(-r * tau)
        elif type == 'put':
            bc_low  = K * np.exp(-r * tau)
            bc_high = 0.0

        # tridiagonal matrix (except endpoints)
        a_in = a_l[1:-1].copy()
        b_in = b_l[1:-1].copy()
        c_in = c_l[1:-1].copy()
        r_in = r_side[1:-1].copy()

        # known value
        r_in[0]  -= a_in[0]  * bc_low
        r_in[-1] -= c_in[-1] * bc_high

        V_in = _solve_tridiagonal(a_in, b_in, c_in, r_in, N - 2)

        V[0]    = bc_low
        V[1:-1] = V_in
        V[-1]   = bc_high

    idx = S0 / dS
    i   = int(idx)
    w   = idx - i
    return (1 - w) * V[i] + w * V[i + 1]




# 1-2) Heston
























# 2) barrier
# 2-1) BS
# FTCS
def FTCS_BS_barrier(eta, type='call', B=0.8, S_max=4, dS=0.01, dt=0.0001, ):
    S0, K, r, sigma, T = eta

    # (S가 B 이하는 knock-out이므로 B부터 시작)
    S_grid = np.arange(B, S_max + dS, dS)
    M = int(T / dt)
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
    for _ in range(M):
        V_new = V.copy()
        V_new[1:-1] = (a[1:-1] * V[:-2]
                     + b[1:-1] * V[1:-1]
                     + c[1:-1] * V[2:])

        # boundary condition
        V_new[0]  = 0   # S=B : knock-out → 가치 0
        if type == 'call':
            V_new[-1] = S_max - K * np.exp(-r * T)
        elif type == 'put':
            V_new[-1] = 0

        V = V_new

    # S0 보간
    idx = (S0 - B) / dS
    i   = int(idx)
    w   = idx - i
    price = (1 - w) * V[i] + w * V[i + 1]
    return price



#  CN
def CN_BS_barrier(eta, type='call', B=0.8, S_max=4, dS=0.01, dt=0.01, ):
    S0, K, r, sigma, T = eta

    S_grid = np.arange(B, S_max + dS, dS)
    N = len(S_grid)
    M = int(T / dt)

    # payoff
    if type == 'call':
        V = np.maximum(S_grid - K, 0).copy()
    elif type == 'put':
        V = np.maximum(K - S_grid, 0).copy()

    a_l, b_l, c_l, a_r, b_r, c_r = _CN_matrix(sigma, r, S_grid, dt, dS)

    for t in range(M):
        tau = (t + 1) * dt

        r_side = np.zeros(N)
        r_side[1:-1] = (a_r[1:-1] * V[:-2]
                        + b_r[1:-1] * V[1:-1]
                        + c_r[1:-1] * V[2:])

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

        V_in = _solve_tridiagonal(a_in, b_in, c_in, r_in, N - 2)

        V[0]    = bc_low
        V[1:-1] = V_in
        V[-1]   = bc_high

    # S0 보간
    idx = (S0 - B) / dS
    i   = int(idx)
    w   = idx - i
    return (1 - w) * V[i] + w * V[i + 1]




# 2-2) Heston