import numpy as np
import cupy as cp
import time

# 2. Monte Carlo
# 1) vanilla
# 1-1) BS
def generate_BS_paths_gpu(eta, n_paths=1000, dt=0.001):
    S0, K, r, sigma, T = eta
    n_steps = int(T / dt)

    W = cp.random.randn(n_paths, n_steps)
    S = cp.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0

    for i in range(n_steps):
        S[:, i+1] = S[:, i] * cp.exp((r - 0.5 * sigma**2) * dt + sigma * cp.sqrt(dt) * W[:, i])

    ST = S[:, -1]
    MT = S.min(axis=1)
    return ST, MT

def MC_BS_vanilla_gpu(eta, n_paths=1000, dt=0.001, type='call'):
    S0, K, r, sigma, T = eta
    ST, MT = generate_BS_paths_gpu(eta, n_paths, dt)

    if type == 'call':
        payoff = cp.maximum(ST - K, 0)
    elif type == 'put':
        payoff = cp.maximum(K - ST, 0)

    return float(cp.exp(-r * T) * payoff.mean())



# 1-2) Heston
def generate_heston_paths(eta, n_paths=2**10, dt=0.001):
    r, lamb, v_bar, epsilon, rho, Y0, T = eta
    n_steps = int(T / dt)

    W1 = np.random.randn(n_paths, n_steps)
    W2 = np.random.randn(n_paths, n_steps)
    dWx = np.sqrt(dt) * W1
    dWy = np.sqrt(dt) * (rho * W1 + np.sqrt(1 - rho**2) * W2)

    X = np.zeros((n_paths, n_steps + 1))
    Y = np.zeros((n_paths, n_steps + 1))
    X[:, 0] = 0.0
    Y[:, 0] = Y0

    for i in range(n_steps):
        Y_t = np.maximum(Y[:, i], 0)
        X[:, i+1] = X[:, i] + (r - 0.5 * Y_t) * dt + np.sqrt(Y_t) * dWx[:, i]
        Y[:, i+1] = (Y_t
                     + lamb * (v_bar - Y_t) * dt
                     + epsilon * np.sqrt(Y_t) * dWy[:, i]
                     + 0.25 * epsilon**2 * (dWy[:, i]**2 - dt))

    XT = X[:, -1]
    YT = Y[:, -1]
    MT = X.min(axis=1)
    mask = filter_paths(XT, T)
    return XT, YT, MT, mask

# 연율화 수익률 평균/분산 필터링 (배치 전체 기준)
def filter_paths(XT, T):
    annual_return = XT / T
    check_mean = np.abs(annual_return.mean()) <= 0.3
    check_var = annual_return.var() <= 1.0
    return bool(check_mean and check_var)

def generate_heston_paths_gpu(eta, n_paths=1000, dt=0.001):
    r, lamb, v_bar, epsilon, rho, Y0, T = eta
    n_steps = int(T / dt)

    W1 = cp.random.randn(n_paths, n_steps)
    W2 = cp.random.randn(n_paths, n_steps)
    dWx = cp.sqrt(dt) * W1
    dWy = cp.sqrt(dt) * (rho * W1 + cp.sqrt(1 - rho**2) * W2)

    X = cp.zeros((n_paths, n_steps + 1))
    Y = cp.zeros((n_paths, n_steps + 1))
    X[:, 0] = 0.0
    Y[:, 0] = Y0

    for i in range(n_steps):
        Y_t = cp.maximum(Y[:, i], 0)
        X[:, i+1] = X[:, i] + (r - 0.5 * Y_t) * dt + cp.sqrt(Y_t) * dWx[:, i]
        Y[:, i+1] = (Y_t
                     + lamb * (v_bar - Y_t) * dt
                     + epsilon * cp.sqrt(Y_t) * dWy[:, i]
                     + 0.25 * epsilon**2 * (dWy[:, i]**2 - dt))

    XT = X[:, -1]
    YT = Y[:, -1]
    MT = X.min(axis=1)
    return XT, YT, MT

def MC_heston_vanilla_gpu(eta, n_paths=1000, dt=0.001, type='call'):
    r, lamb, v_bar, epsilon, rho, Y0, T = eta
    S0, K = 1.0, 1.0
    XT, YT, MT = generate_heston_paths_gpu(eta, n_paths, dt)

    ST = S0 * cp.exp(XT)

    if type == 'call':
        payoff = cp.maximum(ST - K, 0)
    elif type == 'put':
        payoff = cp.maximum(K - ST, 0)

    return float(cp.exp(-r * T) * payoff.mean())


def MC_heston_vanilla_cpu(eta, n_paths=1000, dt=0.001, type='call'):
    r, lamb, v_bar, epsilon, rho, Y0, T = eta
    S0, K = 1.0, 1.0
    XT, YT, MT, masks = generate_heston_paths(eta, n_paths, dt)

    ST = S0 * np.exp(XT)

    if type == 'call':
        payoff = np.maximum(ST - K, 0)
    elif type == 'put':
        payoff = np.maximum(K - ST, 0)

    return float(np.exp(-r * T) * payoff.mean())









# 2) barrier 
# 2-1) BS
def MC_BS_barrier_gpu(eta, B, n_paths=2**14, dt=0.001, type='call'):
    S0, K, r, sigma, T = eta
    ST, MT = generate_BS_paths_gpu(eta, n_paths, dt)

    if type == 'call':
        payoff = cp.where(MT > B, cp.maximum(ST - K, 0), 0)
    elif type == 'put':
        payoff = cp.where(MT > B, cp.maximum(K - ST, 0), 0)

    price = float(cp.exp(-r * T) * payoff.mean())
    return price

# 2-2) Heston
def MC_heston_barrier_gpu(eta, B, n_paths=2**14, dt=0.001, type='call'):
    r, lamb, v_bar, epsilon, rho, Y0, T = eta
    S0, K = 1.0, 1.0
    XT, YT, MT = generate_heston_paths_gpu(eta, n_paths, dt)

    ST   = S0 * cp.exp(XT)
    MT_S = S0 * cp.exp(MT)

    if type == 'call':
        payoff = cp.where(MT_S > B, cp.maximum(ST - K, 0), 0)
    elif type == 'put':
        payoff = cp.where(MT_S > B, cp.maximum(K - ST, 0), 0)

    return float(cp.exp(-r * T) * payoff.mean())