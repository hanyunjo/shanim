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

        # reflection principel correction
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
def generate_heston_paths(eta, B=0, n_paths=2**10, dt=0.001):
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
        
        # reflection principel correction
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
    return XT, knocked, mask

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
        
        # reflection principel correction
        if B != 0:
            a = log_S0B + X[:, i]      # X = ln(S_t/S0), a = ln(S_t/B)
            b = log_S0B + X[:, i+1]
            v_local = 0.5 * (Y_t + cp.maximum(Y[:, i+1], 0))

            valid   = (a > 0) & (b > 0)  
            p_cross = cp.where(valid, cp.exp(-2 * a * b / (v_local * dt + 1e-10)), 0.0)
            u = cp.random.uniform(size=n_paths)
            knocked = knocked | (u < p_cross)

    XT = X[:, -1]
    YT = Y[:, -1]
    if B != 0:
        knocked = knocked | (X.min(axis=1) <= cp.log(B/S0))
    return XT, knocked

def MC_heston_vanilla_cpu(eta, n_paths=1000, dt=0.001, type='call'):
    S0, K, r, kappa, theta, xi, rho, Y0, T = eta
    XT, MT, masks = generate_heston_paths(eta, n_paths, dt)

    ST = S0 * np.exp(XT)

    if type == 'call':
        payoff = np.maximum(ST - K, 0)
    elif type == 'put':
        payoff = np.maximum(K - ST, 0)

    return float(np.exp(-r * T) * payoff.mean())

def MC_heston_vanilla_gpu(eta, n_paths=1000, dt=0.001, type='call'):
    S0, K, r, kappa, theta, xi, rho, Y0, T = eta
    XT, knocked = generate_heston_paths_gpu(eta, n_paths, dt)

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
    XT, knocked = generate_heston_paths_gpu(eta, n_paths, dt, B)

    ST   = S0 * cp.exp(XT)
    
    if type == 'call':
        payoff = cp.where(knocked, 0, cp.maximum(ST - K, 0))
    elif type == 'put':
        payoff = cp.where(knocked, 0, cp.maximum(K - ST, 0))

    return float(cp.exp(-r * T) * payoff.mean())










# ======================
# Greeks
# ======================
def _mc_validate_option_type(opt_type):
    if opt_type not in ('call', 'put'):
        raise ValueError("opt_type must be 'call' or 'put'.")

def _mc_n_steps(T, dt):
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be a positive finite number.')
    n_steps = int(round(float(T) / float(dt)))
    if n_steps <= 0 or not np.isclose(n_steps * dt, T, rtol=0.0, atol=1e-10):
        raise ValueError(f'T={T} must be a positive integer multiple of dt={dt}.')
    return n_steps

def _mc_normal(rng, n_paths, antithetic):
    if not antithetic:
        return rng.standard_normal(n_paths)
    half = n_paths // 2
    z_half = rng.standard_normal(half)
    return cp.concatenate((z_half, -z_half))[:n_paths]

def _mc_payoff(S_T, K, opt_type):
    if opt_type == 'call':
        return cp.maximum(S_T - K, 0.0)
    return cp.maximum(K - S_T, 0.0)

def _mc_validate_bs_eta(eta):
    eta_array = np.asarray(eta, dtype=float)
    if eta_array.shape != (5,) or not np.all(np.isfinite(eta_array)):
        raise ValueError('BS eta must be finite (S0, K, r, sigma, T).')
    S0, K, _, sigma, T = eta_array
    if min(S0, K, sigma, T) <= 0:
        raise ValueError('S0, K, sigma, and T must be positive.')
    return eta_array

def _mc_validate_heston_eta(eta):
    eta_array = np.asarray(eta, dtype=float)
    if eta_array.shape != (9,) or not np.all(np.isfinite(eta_array)):
        raise ValueError('Heston eta must be finite (S0, K, r, kappa, theta, xi, rho, v0, T).')
    S0, K, _, kappa, long_var, xi, rho, v0, T = eta_array
    if min(S0, K, kappa, long_var, xi, T) <= 0:
        raise ValueError('S0, K, kappa, theta, xi, and T must be positive.')
    if v0 < 0:
        raise ValueError('v0 must be nonnegative.')
    if not -1.0 <= rho <= 1.0:
        raise ValueError('rho must lie in [-1, 1].')
    return eta_array

def _mc_aligned_step(raw_step, dt):
    if not np.isfinite(raw_step) or raw_step <= 0:
        raise ValueError('finite-difference steps must be positive and finite.')
    return max(1, int(round(raw_step / dt))) * dt

def _mc_check_steps(default_steps, h, dt):
    steps = dict(default_steps)
    if h is not None:
        if not isinstance(h, dict):
            raise TypeError("h must be a dictionary such as {'delta': 0.01, 'vega': 0.001}.")
        unknown = set(h) - set(steps)
        if unknown:
            raise ValueError(f'Unknown bump variables: {sorted(unknown)}')
        for name, value in h.items():
            value = float(value)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"h['{name}'] must be a positive finite number.")
            steps[name] = value
    steps['theta'] = _mc_aligned_step(steps['theta'], dt)
    return steps

def _mc_bs_batch_prices_gpu(
    etas,
    B=None,
    n_paths=1_000,
    dt=0.001,
    opt_type='call',
    seed=None,
    antithetic=True,
):
    _mc_validate_option_type(opt_type)
    eta_array = np.asarray(etas, dtype=float)
    if eta_array.ndim != 2 or eta_array.shape[1] != 5:
        raise ValueError('etas must have shape (n_scenarios, 5).')
    for scenario in eta_array:
        _mc_validate_bs_eta(scenario)
    if int(n_paths) != n_paths or n_paths <= 0:
        raise ValueError('n_paths must be a positive integer.')
    n_paths = int(n_paths)

    S0, K, r, sigma, T = eta_array.T
    if B is not None:
        B = float(B)
        if not np.isfinite(B) or B <= 0 or B >= np.min(S0):
            raise ValueError('Down barrier must satisfy 0 < B < every spot value.')

    rng = cp.random.RandomState(seed)
    S0_gpu = cp.asarray(S0)[:, None]
    K_gpu = cp.asarray(K)[:, None]
    r_gpu = cp.asarray(r)[:, None]
    sigma_gpu = cp.asarray(sigma)[:, None]
    T_gpu = cp.asarray(T)[:, None]

    if B is None:
        z = _mc_normal(rng, n_paths, antithetic)
        ST = S0_gpu * cp.exp(
            (r_gpu - 0.5 * sigma_gpu ** 2) * T_gpu
            + sigma_gpu * cp.sqrt(T_gpu) * z
        )
        prices = cp.exp(-cp.asarray(r) * cp.asarray(T)) * _mc_payoff(
            ST, K_gpu, opt_type
        ).mean(axis=1)
        return cp.asnumpy(prices).astype(float)

    n_steps = np.asarray([_mc_n_steps(maturity, dt) for maturity in T])
    n_scenarios = len(eta_array)
    sqrt_dt = cp.sqrt(dt)
    S = cp.repeat(S0_gpu, n_paths, axis=1)
    survival = cp.ones((n_scenarios, n_paths), dtype=cp.float64)

    for i in range(int(n_steps.max())):
        z = _mc_normal(rng, n_paths, antithetic)
        old = S
        new = old * cp.exp(
            (r_gpu - 0.5 * sigma_gpu ** 2) * dt + sigma_gpu * sqrt_dt * z
        )
        a, b = cp.log(old / B), cp.log(new / B)
        valid = (a > 0.0) & (b > 0.0)
        exponent = cp.where(
            valid, -2.0 * a * b / (sigma_gpu ** 2 * dt), 0.0
        )
        active = cp.asarray(i < n_steps)[:, None]
        survival = cp.where(
            active, survival * (-cp.expm1(exponent)), survival
        )
        S = cp.where(active, new, S)

    prices = cp.exp(-cp.asarray(r) * cp.asarray(T)) * (
        survival * _mc_payoff(S, K_gpu, opt_type)
    ).mean(axis=1)
    return cp.asnumpy(prices).astype(float)


def _mc_heston_batch_prices_gpu(
    etas,
    B=None,
    n_paths=1_000,
    dt=0.001,
    opt_type='call',
    seed=None,
    antithetic=True,
):
    _mc_validate_option_type(opt_type)
    eta_array = np.asarray(etas, dtype=float)
    if eta_array.ndim != 2 or eta_array.shape[1] != 9:
        raise ValueError('etas must have shape (n_scenarios, 9).')
    for scenario in eta_array:
        _mc_validate_heston_eta(scenario)
    if int(n_paths) != n_paths or n_paths <= 0:
        raise ValueError('n_paths must be a positive integer.')
    n_paths = int(n_paths)

    S0, K, r, kappa, long_var, xi, corr, v0, T = eta_array.T
    if B is not None:
        B = float(B)
        if not np.isfinite(B) or B <= 0 or B >= np.min(S0):
            raise ValueError('Down barrier must satisfy 0 < B < every spot value.')

    n_steps = np.asarray([_mc_n_steps(maturity, dt) for maturity in T])
    n_scenarios = len(eta_array)
    sqrt_dt = cp.sqrt(dt)
    rng = cp.random.RandomState(seed)
    X = cp.zeros((n_scenarios, n_paths), dtype=cp.float64)
    V = cp.repeat(cp.asarray(v0)[:, None], n_paths, axis=1)
    survival = cp.ones((n_scenarios, n_paths), dtype=cp.float64)
    r_gpu = cp.asarray(r)[:, None]
    kappa_gpu = cp.asarray(kappa)[:, None]
    long_var_gpu = cp.asarray(long_var)[:, None]
    xi_gpu = cp.asarray(xi)[:, None]
    corr_gpu = cp.asarray(corr)[:, None]
    S0_gpu = cp.asarray(S0)[:, None]
    K_gpu = cp.asarray(K)[:, None]
    if B is not None:
        log_S0B = cp.asarray(np.log(S0 / B))[:, None]

    for i in range(int(n_steps.max())):
        w1 = _mc_normal(rng, n_paths, antithetic)
        w2 = _mc_normal(rng, n_paths, antithetic)
        V_pos = cp.maximum(V, 0.0)
        dWx = sqrt_dt * w1
        dWv = sqrt_dt * (
            corr_gpu * w1
            + cp.sqrt(cp.maximum(1.0 - corr_gpu ** 2, 0.0)) * w2
        )
        X_new = X + (r_gpu - 0.5 * V_pos) * dt + cp.sqrt(V_pos) * dWx
        V_new = (
            V_pos
            + kappa_gpu * (long_var_gpu - V_pos) * dt
            + xi_gpu * cp.sqrt(V_pos) * dWv
            + 0.25 * xi_gpu ** 2 * (dWv ** 2 - dt)
        )
        active = cp.asarray(i < n_steps)[:, None]
        if B is not None:
            a, b = log_S0B + X, log_S0B + X_new
            v_local = 0.5 * (V_pos + cp.maximum(V_new, 0.0))
            valid = (a > 0.0) & (b > 0.0)
            exponent = cp.where(
                valid, -2.0 * a * b / (v_local * dt + 1e-14), 0.0
            )
            survival = cp.where(
                active, survival * (-cp.expm1(exponent)), survival
            )
        X = cp.where(active, X_new, X)
        V = cp.where(active, V_new, V)

    payoff = _mc_payoff(S0_gpu * cp.exp(X), K_gpu, opt_type)
    if B is not None:
        payoff *= survival
    prices = cp.exp(-cp.asarray(r) * cp.asarray(T)) * payoff.mean(axis=1)
    return cp.asnumpy(prices).astype(float)


def MC_BS_greeks_gpu(
    eta,
    B=None,
    opt_type='call',
    n_paths=1_000,
    dt=0.001,
    greeks=('delta', 'vega', 'rho', 'theta'),
    h=None,
    relative_step=0.01,
    seed=1234,
    antithetic=True,
):
    _mc_validate_option_type(opt_type)
    eta_base = _mc_validate_bs_eta(eta)
    _mc_n_steps(eta_base[4], dt)
    if not np.isfinite(relative_step) or relative_step <= 0:
        raise ValueError('relative_step must be positive and finite.')
    if B is not None and (not np.isfinite(B) or B <= 0 or B >= eta_base[0]):
        raise ValueError('Down barrier must satisfy 0 < B < S0.')

    S0, _, r, sigma, T = eta_base
    defaults = {
        'delta': max(abs(S0) * relative_step, 1e-6),
        'rho': max(abs(r) * relative_step, 1e-6),
        'vega': max(abs(sigma) * relative_step, 1e-6),
        'theta': max(abs(T) * relative_step, dt),
    }
    steps = _mc_check_steps(defaults, h, dt)
    mapping = {
        'delta': (0, 'delta', 1.0),
        'rho': (2, 'rho', 1.0),
        'vega': (3, 'vega', 1.0),
        'theta': (4, 'theta', -1.0),
    }
    requested = tuple(greeks)
    unknown = set(requested) - (set(mapping))
    if unknown:
        raise ValueError(f'Unknown BS Greeks: {sorted(unknown)}')

    scenarios = [eta_base.copy()]
    bumped = []
    for greek in requested:
        index, step_name, sign = mapping[greek]
        eta_up = eta_base.copy()
        eta_up[index] += steps[step_name]
        scenarios.append(eta_up)
        bumped.append((greek, step_name, sign))

    prices = _mc_bs_batch_prices_gpu(
        scenarios, B=B, n_paths=n_paths, dt=dt, opt_type=opt_type,
        seed=seed, antithetic=antithetic,
    )
    price_base = float(prices[0])
    result = {'h': dict(steps), 'antithetic': bool(antithetic), 'details': {}}
    for scenario_index, (greek, step_name, sign) in enumerate(bumped, start=1):
        step = steps[step_name]
        price_up = float(prices[scenario_index])
        result[greek] = float(sign * (price_up - price_base) / step)
        result['details'][greek] = {
            'scheme': 'forward',
            'price_base': price_base,
            'price_up': price_up,
            'h': step,
        }
    return result

def MC_heston_greeks_gpu(
    eta,
    B=None,
    opt_type='call',
    n_paths=100_000,
    dt=0.001,
    greeks=('delta', 'rho', 'v0', 'theta'),
    h=None,
    relative_step=0.01,
    seed=1234,
    antithetic=True,
):
    _mc_validate_option_type(opt_type)
    eta_base = _mc_validate_heston_eta(eta)
    _mc_n_steps(eta_base[8], dt)
    if not np.isfinite(relative_step) or relative_step <= 0:
        raise ValueError('relative_step must be positive and finite.')
    if B is not None and (not np.isfinite(B) or B <= 0 or B >= eta_base[0]):
        raise ValueError('Down barrier must satisfy 0 < B < S0.')

    S0, _, r, kappa, long_var, xi, rho, v0, T = eta_base
    defaults = {
        'delta': max(abs(S0) * relative_step, 1e-6),
        'rho': max(abs(r) * relative_step, 1e-6),
        'v0': max(abs(v0) * relative_step, 1e-6),
        'theta': max(abs(T) * relative_step, dt),
    }
    steps = _mc_check_steps(defaults, h, dt)
    mapping = {
        'delta': (0, 'delta', 1.0),
        'rho': (2, 'rho', 1.0),
        'v0': (7, 'v0', 1.0),
        'theta': (8, 'theta', -1.0),
    }
    requested = tuple(greeks)
    unknown = set(requested) - (set(mapping))
    if unknown:
        raise ValueError(f'Unknown Heston sensitivities: {sorted(unknown)}')

    scenarios = [eta_base.copy()]
    bumped = []
    for greek in requested:
        index, step_name, sign = mapping[greek]
        eta_up = eta_base.copy()
        eta_up[index] += steps[step_name]
        scenarios.append(eta_up)
        bumped.append((greek, step_name, sign))

    prices = _mc_heston_batch_prices_gpu(
        scenarios, B=B, n_paths=n_paths, dt=dt, opt_type=opt_type,
        seed=seed, antithetic=antithetic,
    )
    price_base = float(prices[0])
    result = {'h': dict(steps), 'antithetic': bool(antithetic), 'details': {}}
    for scenario_index, (greek, step_name, sign) in enumerate(bumped, start=1):
        step = steps[step_name]
        price_up = float(prices[scenario_index])
        result[greek] = float(sign * (price_up - price_base) / step)
        result['details'][greek] = {
            'scheme': 'forward',
            'price_base': price_base,
            'price_up': price_up,
            'h': step,
        }
    return result
