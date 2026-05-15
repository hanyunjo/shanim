import numpy as np
import cupy as cp
import time

# 2. Monte Carlo
# 1) vanilla
# 1-1) BS
def generate_BS_paths_gpu(eta, n_paths=1000, dt=0.001, B=0):
    S0, K, r, sigma, T = eta
    n_steps = int(T / dt)

    W = cp.random.randn(n_paths, n_steps)
    S = cp.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0

    knocked = cp.zeros(n_paths, dtype=bool) # 1 : barrier를 건든 것

    for i in range(n_steps):
        S[:, i+1] = S[:, i] * cp.exp((r - 0.5 * sigma**2) * dt + sigma * cp.sqrt(dt) * W[:, i])

        # Brownian Bridge 보정
        if B != 0:
            a = cp.log(S[:, i] / B)
            b = cp.log(S[:, i+1] / B)

            valid   = (a > 0) & (b > 0)  # 두 점 모두 barrier 위
            p_cross = cp.where(valid, cp.exp(-2 * a * b / (sigma**2 * dt + 1e-10)), 0.0) # 베리어 위에 두 점이 있을 때 베리어를 건드릴 확률
            u = cp.random.uniform(size=n_paths) # 0~1사이의 랜덤 변수, 베르누이 시행
            knocked = knocked | (u < p_cross)

    if B != 0:
        knocked = knocked | (S.min(axis=1) <= B)

    ST = S[:, -1]
    return ST, knocked

def MC_BS_vanilla_gpu(eta, n_paths=1000, dt=0.001, type='call'):
    S0, K, r, sigma, T = eta
    
    # ST, knocked = generate_BS_paths_gpu(eta, n_paths, dt)
    Z = cp.random.randn(n_paths)
    ST = S0 * cp.exp((r - 0.5 * sigma**2) * T + sigma * cp.sqrt(T) * Z)

    if type == 'call':
        payoff = cp.maximum(ST - K, 0)
    elif type == 'put':
        payoff = cp.maximum(K - ST, 0)

    return float(cp.exp(-r * T) * payoff.mean())






# 1-2) Heston
def generate_heston_paths(eta, n_paths=2**10, dt=0.001, B=0):
    S0, K, r, kappa, theta, xi, rho, Y0, T = eta
    n_steps = int(T / dt)

    W1 = np.random.randn(n_paths, n_steps)
    W2 = np.random.randn(n_paths, n_steps)
    dWx = np.sqrt(dt) * W1
    dWy = np.sqrt(dt) * (rho * W1 + np.sqrt(1 - rho**2) * W2)

    X = np.zeros((n_paths, n_steps + 1))
    Y = np.zeros((n_paths, n_steps + 1))
    X[:, 0] = 0.0
    Y[:, 0] = Y0

    knocked = np.zeros(n_paths, dtype=bool) # 1 : barrier를 건든 것
    if B != 0:
        log_S0B = np.log(S0 / B) 

    for i in range(n_steps):
        Y_t = np.maximum(Y[:, i], 0)
        # X = ln(S_t/S0)
        X[:, i+1] = X[:, i] + (r - 0.5 * Y_t) * dt + np.sqrt(Y_t) * dWx[:, i]
        Y[:, i+1] = (Y_t
                     + kappa * (theta - Y_t) * dt
                     + xi * np.sqrt(Y_t) * dWy[:, i]
                     + 0.25 * xi**2 * (dWy[:, i]**2 - dt))
        
        # Brownian Bridge 보정
        if B != 0:
            a = log_S0B + X[:, i] 
            b = log_S0B + X[:, i+1]    
            v_local = 0.5 * (Y_t + np.maximum(Y[:, i+1], 0))

            valid   = (a > 0) & (b > 0)
            p_cross = np.where(valid, np.exp(-2 * a * b / (v_local * dt + 1e-10)), 0.0)
            u = np.random.uniform(size=n_paths)
            knocked = knocked | (u < p_cross)

    XT = X[:, -1]
    YT = Y[:, -1]
    if B != 0:
        knocked = knocked | (X.min(axis=1) <= np.log(B/S0))
    mask = filter_paths(XT, T)
    return XT, YT, knocked, mask

# 연율화 수익률 평균/분산 필터링 (배치 전체 기준)
def filter_paths(XT, T):
    annual_return = XT / T
    check_mean = np.abs(annual_return.mean()) <= 0.3
    check_var = annual_return.var() <= 1.0
    return bool(check_mean and check_var)

def generate_heston_paths_gpu(eta, n_paths=1000, dt=0.001, B=0):
    S0, K, r, kappa, theta, xi, rho, Y0, T = eta
    n_steps = int(T / dt)

    W1 = cp.random.randn(n_paths, n_steps)
    W2 = cp.random.randn(n_paths, n_steps)
    dWx = cp.sqrt(dt) * W1
    dWy = cp.sqrt(dt) * (rho * W1 + cp.sqrt(1 - rho**2) * W2)

    X = cp.zeros((n_paths, n_steps + 1))
    Y = cp.zeros((n_paths, n_steps + 1))
    X[:, 0] = 0.0
    Y[:, 0] = Y0

    knocked = cp.zeros(n_paths, dtype=bool) # 1 : barrier를 건든 것
    if B != 0:
        log_S0B = cp.log(S0 / B) 

    for i in range(n_steps):
        Y_t = cp.maximum(Y[:, i], 0) # milstein 식을 사용했고, 음수면 0을 사용하기 때문에 조건을 확인하지 않음.
        X[:, i+1] = X[:, i] + (r - 0.5 * Y_t) * dt + cp.sqrt(Y_t) * dWx[:, i]
        Y[:, i+1] = (Y_t
                     + kappa * (theta - Y_t) * dt
                     + xi * cp.sqrt(Y_t) * dWy[:, i]
                     + 0.25 * xi**2 * (dWy[:, i]**2 - dt))
        
        # Brownian Bridge 보정
        if B != 0:
            a = log_S0B + X[:, i]      # X = ln(S_t/S0), a = ln(S_t/B)
            b = log_S0B + X[:, i+1]
            v_local = 0.5 * (Y_t + cp.maximum(Y[:, i+1], 0))

            valid   = (a > 0) & (b > 0)  # 두 점 모두 barrier 위
            p_cross = cp.where(valid, cp.exp(-2 * a * b / (v_local * dt + 1e-10)), 0.0) # 베리어 위에 두 점이 있을 때 베리어를 건드릴 확률
            u = cp.random.uniform(size=n_paths) # 0~1사이의 랜덤 변수, 베르누이 시행
            knocked = knocked | (u < p_cross)

    XT = X[:, -1]
    YT = Y[:, -1]
    if B != 0:
        knocked = knocked | (X.min(axis=1) <= cp.log(B/S0))
    return XT, YT, knocked

def MC_heston_vanilla_cpu(eta, n_paths=1000, dt=0.001, type='call'):
    S0, K, r, kappa, theta, xi, rho, Y0, T = eta
    XT, YT, MT, masks = generate_heston_paths(eta, n_paths, dt)

    ST = S0 * np.exp(XT)

    if type == 'call':
        payoff = np.maximum(ST - K, 0)
    elif type == 'put':
        payoff = np.maximum(K - ST, 0)

    return float(np.exp(-r * T) * payoff.mean())

def MC_heston_vanilla_gpu(eta, n_paths=1000, dt=0.001, type='call'):
    S0, K, r, kappa, theta, xi, rho, Y0, T = eta
    XT, YT, knocked = generate_heston_paths_gpu(eta, n_paths, dt)

    ST = S0 * cp.exp(XT)

    if type == 'call':
        payoff = cp.maximum(ST - K, 0)
    elif type == 'put':
        payoff = cp.maximum(K - ST, 0)

    return float(cp.exp(-r * T) * payoff.mean())












# 2) barrier 
# 2-1) BS
def MC_BS_barrier_gpu(eta, B, n_paths=1000, dt=0.001, type='call'):
    S0, K, r, sigma, T = eta
    ST, knocked = generate_BS_paths_gpu(eta, n_paths, dt, B)

    if type == 'call':
        payoff = cp.where(knocked, 0, cp.maximum(ST - K, 0))
    elif type == 'put':
        payoff = cp.where(knocked, 0, cp.maximum(K - ST, 0))

    price = float(cp.exp(-r * T) * payoff.mean())
    return price

# 2-2) Heston
def MC_heston_barrier_gpu(eta, B, n_paths=1000, dt=0.001, type='call'):
    S0, K, r, kappa, theta, xi, rho, Y0, T = eta
    XT, YT, knocked = generate_heston_paths_gpu(eta, n_paths, dt, B)

    ST   = S0 * cp.exp(XT)
    
    if type == 'call':
        payoff = cp.where(knocked, 0, cp.maximum(ST - K, 0))
    elif type == 'put':
        payoff = cp.where(knocked, 0, cp.maximum(K - ST, 0))

    return float(cp.exp(-r * T) * payoff.mean())