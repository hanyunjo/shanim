import numpy as np
from scipy.stats import norm

# 1. closed-form
# 1)vanilla
def BS_vanilla(eta, type='call'):
    # type : 'call' or 'put'
    S0, K, r, sigma, T = eta

    sqT  = sigma * np.sqrt(T)
    disc = np.exp(-r * T)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / sqT
    d2 = d1 - sqT

    if type == 'call':
        price = S0 * norm.cdf(d1) - K * disc * norm.cdf(d2)
    elif type == 'put':
        price = K * disc * norm.cdf(-d2) - S0 * norm.cdf(-d1)

    return price



#2) down-and-out
def BS_barrier(eta, B, type='call'):
    S0, K, r, sigma, T  = eta
    
    lam = r / sigma**2 + 0.5          
    sqT = sigma * np.sqrt(T)          
    disc = np.exp(-r * T)                     

    # vanilla
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / sqT
    d2 = d1 - sqT

    def vanilla_call():
        return S0 * norm.cdf(d1) - K * disc * norm.cdf(d2)

    def vanilla_put():
        return K * disc * norm.cdf(-d2) - S0 * norm.cdf(-d1)


    coef1 = S0 * (B / S0)**(2 * lam)           
    coef2 = K * disc * (B / S0)**(2 * lam - 2) 
    x1 = np.log(B / S0) / sqT + lam * sqT
    x2 = x1 - sqT
    y1 = np.log(B**2 / (S0 * K)) / sqT + lam * sqT
    y2 = y1 - sqT

    if type == 'call':
        C_do = (vanilla_call() 
                - coef1 * norm.cdf(y1) 
                + coef2 * norm.cdf(y2))           
        
        return C_do
    elif type == 'put':
        x_h1 = np.log(S0 / B) / sqT + lam * sqT
        x_h2 = x_h1 - sqT

        P_do = (vanilla_put()
                + S0 * norm.cdf(-x_h1) - K * disc * norm.cdf(-x_h2)
                - coef1 * (norm.cdf(y1) - norm.cdf(x1))
                + coef2 * (norm.cdf(y2) - norm.cdf(x2)))
        
    return P_do




# ======================
# Greeks
# ======================
_GREEK_PARAMETER_INFO = {
    'delta': 0,
    'rho': 2,
    'vega': 3,
    'theta': 4,
}


def _validate_option_type(option_type):
    if option_type not in ('call', 'put'):
        raise ValueError("type must be 'call' or 'put'.")


def _validate_bs_eta(eta):
    eta_array = np.asarray(eta, dtype=float)

    if eta_array.shape != (5,):
        raise ValueError('eta must be (S0, K, r, sigma, T).')
    if not np.all(np.isfinite(eta_array)):
        raise ValueError('eta must contain only finite values.')

    S0, K, _, sigma, T = eta_array
    if S0 <= 0:
        raise ValueError('S0 must be positive.')
    if K <= 0:
        raise ValueError('K must be positive.')
    if sigma <= 0:
        raise ValueError('sigma must be positive.')
    if T <= 0:
        raise ValueError('T must be positive.')

    return eta_array


def _default_greek_steps(eta, relative_step=1e-2):
    if relative_step <= 0:
        raise ValueError('relative_step must be positive.')

    S0, _, r, sigma, T = eta
    return {
        'delta': max(abs(S0) * relative_step, 1e-6),
        'rho': max(abs(r) * relative_step, 1e-6),
        'vega': max(abs(sigma) * relative_step, 1e-6),
        'theta': max(abs(T) * relative_step, 1e-6),
    }


def _check_greek_steps(eta, h, relative_step):
    steps = _default_greek_steps(eta, relative_step=relative_step)

    if h is None:
        return steps
    if not isinstance(h, dict):
        raise TypeError(
            "h must be a dictionary, for example "
            "{'delta': 0.01, 'vega': 0.001, 'rho': 0.0001, 'theta': 0.01}."
        )

    invalid_names = set(h) - set(steps)
    if invalid_names:
        raise ValueError(f'Unknown bump variables: {sorted(invalid_names)}')

    for name, value in h.items():
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"h['{name}'] must be a positive finite number.")
        steps[name] = value

    return steps


def BS_vanilla_greeks(eta, type='call'):
    _validate_option_type(type)
    S0, K, r, sigma, T = _validate_bs_eta(eta)

    sqrt_T = np.sqrt(T)
    sigma_sqrt_T = sigma * sqrt_T
    discount = np.exp(-r * T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T
    pdf_d1 = norm.pdf(d1)

    vega = S0 * pdf_d1 * sqrt_T

    if type == 'call':
        delta = norm.cdf(d1)
        rho = K * T * discount * norm.cdf(d2)
        theta = -S0 * sigma * pdf_d1 / (2.0 * sqrt_T) - r * K * discount * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1.0
        rho = -K * T * discount * norm.cdf(-d2)
        theta = -S0 * sigma * pdf_d1 / (2.0 * sqrt_T) + r * K * discount * norm.cdf(-d2)

    return {
        'price': float(BS_vanilla((S0, K, r, sigma, T), type=type)),
        'delta': float(delta),
        'vega': float(vega),
        'rho': float(rho),
        'theta': float(theta),
    }


def BS_barrier_greeks(eta, B, type='call', h=None, relative_step=1e-2):
    _validate_option_type(type)

    eta_base = _validate_bs_eta(eta)
    S0, _, _, sigma, T = eta_base
    B = float(B)
    if not np.isfinite(B) or B <= 0:
        raise ValueError('B must be a positive finite number.')
    if B >= S0:
        raise ValueError('Down-and-out Greeks require B < S0.')

    steps = _check_greek_steps(eta_base, h, relative_step)
    base_price = float(BS_barrier(eta_base, B=B, type=type))
    greeks = {}

    for greek_name, index in _GREEK_PARAMETER_INFO.items():
        step = steps[greek_name]
        eta_up = eta_base.copy()
        eta_up[index] += step
        up_price = float(BS_barrier(eta_up, B=B, type=type))

        sensitivity = (up_price - base_price) / step

        greeks[greek_name] = -sensitivity if greek_name == 'theta' else sensitivity


    return {
        'price': base_price,
        'delta': float(greeks['delta']),
        'vega': float(greeks['vega']),
        'rho': float(greeks['rho']),
        'theta': float(greeks['theta']),
        'h': steps,
    }


def BS_greeks(eta, type='call', B=None, h=None, relative_step=1e-2):
    if B is None:
        return BS_vanilla_greeks(eta, type=type)

    return BS_barrier_greeks(
        eta, B=B, type=type, h=h, relative_step=relative_step,
    )
