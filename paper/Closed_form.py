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



#2) down-and-out call
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
    x1 = np.log(B**2 / (S0 * K)) / sqT + lam * sqT
    x2 = x1 - sqT

    if type == 'call':
        C_do = (vanilla_call() 
                - coef1 * norm.cdf(x1) 
                + coef2 * norm.cdf(x2))            
        
        return C_do
    elif type == 'put':
        x_h1 = np.log(S0 / B) / sqT + lam * sqT
        x_h2 = x_h1 - sqT

        y1 = np.log(B**2 / (S0 * K)) / sqT + lam * sqT
        y2 = y1 - sqT

        P_do = (vanilla_put()
                + S0 * norm.cdf(-x_h1) - K * disc * norm.cdf(-x_h2)
                - coef1 * (norm.cdf(y1) - norm.cdf(x1))
                + coef2 * (norm.cdf(y2) - norm.cdf(x2)))


        return P_do