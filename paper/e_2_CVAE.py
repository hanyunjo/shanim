"""
CVAE Architecture (Sohn et al. 2015):
"""

import torch
import torch.nn as nn
import numpy as np

if not torch.cuda.is_available():
    raise ValueError("Cannot use GPU cuda")
device = torch.device("cuda")


# ────────────
# Sub-networks
# ────────────
def _mlp(in_dim: int, hidden_dims: list, out_dim: int, activation=nn.Tanh) -> nn.Sequential:
    # Tanh : Greeks 계산 시 미분가능성 보장
    layers = []
    dims = [in_dim] + hidden_dims + [out_dim]
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i + 1]), activation()]
    return nn.Sequential(*layers)


class RecognitionNet(nn.Module):
    """
    q_φ(z | x, η)
    Input  : [x, η]  dim = dim_x + dim_eta
    Output : μ_z, log σ_z  dim = dim_z each
    """
    def __init__(self, dim_x, dim_eta, dim_z, hidden_dims):
        super().__init__()
        self.net    = _mlp(dim_x + dim_eta, hidden_dims[:-1], hidden_dims[-1])
        self.mu     = nn.Linear(hidden_dims[-1], dim_z)
        self.logvar = nn.Linear(hidden_dims[-1], dim_z)

    def forward(self, x, eta):
        h       = self.net(torch.cat([x, eta], dim=-1))
        mu      = self.mu(h)
        log_var = self.logvar(h)   # log_var 사용 : 음수/연속성 문제 방지
        log_var = torch.clamp(log_var, min=-10.0, max=5.0)  # log_var의 범위를 제한해 loss에서 exp 계산 안정성 향상, 의미상 log(var)이기 때문
        return mu, log_var


class PriorNet(nn.Module):
    """
    p_θ(z | η)
    Input  : η  dim = dim_eta
    Output : μ_p, log σ_p  dim = dim_z each
    """
    def __init__(self, dim_eta, dim_z, hidden_dims):
        super().__init__()
        self.net    = _mlp(dim_eta, hidden_dims[:-1], hidden_dims[-1])
        self.mu     = nn.Linear(hidden_dims[-1], dim_z)
        self.logvar = nn.Linear(hidden_dims[-1], dim_z)

    def forward(self, eta):
        h       = self.net(eta)
        mu      = self.mu(h)
        log_var = self.logvar(h)
        log_var = torch.clamp(log_var, min=-10.0, max=5.0)
        return mu, log_var


class DecoderNet(nn.Module):
    """
    p_θ(x | z, η)
    Input  : [z, η]  dim = dim_z + dim_eta
    Output : μ_x, log σ_x  dim = dim_x each
    """
    def __init__(self, dim_z, dim_eta, dim_x, hidden_dims):
        super().__init__()
        self.net    = _mlp(dim_z + dim_eta, hidden_dims[:-1], hidden_dims[-1])
        self.mu     = nn.Linear(hidden_dims[-1], dim_x)
        self.logvar = nn.Linear(hidden_dims[-1], dim_x)

    def forward(self, z, eta):
        h       = self.net(torch.cat([z, eta], dim=-1))
        mu      = self.mu(h)
        log_var = self.logvar(h)
        log_var = torch.clamp(log_var, min=-10.0, max=5.0)
        return mu, log_var


# ─────────
# CVAE
# ─────────
class CVAE(nn.Module):
    def __init__(self,
                 dim_x: int = 2,       # (X_T, M_T)=2, (X_T)=1
                 dim_eta: int = 7,     # BS=3, Heston=7
                 dim_z: int = 8,       # latent dim
                 hidden_dims: list = None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128, 64]

        self.recognition = RecognitionNet(dim_x, dim_eta, dim_z, hidden_dims)
        self.prior       = PriorNet(             dim_eta, dim_z, hidden_dims)
        self.decoder     = DecoderNet(dim_z,     dim_eta, dim_x, hidden_dims)

    @staticmethod
    def reparameterize(mu, log_var, eps=None):
        # z = μ + ε·σ,  ε ~ N(0, I)
        std = torch.exp(0.5 * log_var)
        if eps is None:
            eps = torch.randn_like(std)
        return mu + eps * std

    # forward + ELBO 
    def forward(self, x, eta):
        mu_q, lv_q = self.recognition(x, eta)   # reg : q_φ(z|x,η)
        mu_p, lv_p = self.prior(eta)            # prior : p_θ(z|η)
        z = self.reparameterize(mu_q, lv_q)
        mu_x, lv_x = self.decoder(z, eta)       # decoder : p_θ(x|z,η)

        # Reconstruction loss : Gaussian NLL
        recon_loss = 0.5 * (
            lv_x + (x - mu_x).pow(2) / lv_x.exp() + np.log(2 * np.pi)
        ).sum(dim=-1).mean()

        # KL between two Gaussians : KL( q_φ(z|x,η) || p_θ(z|η) ) = KL(recog || prior)
        kl_loss = 0.5 * (
            lv_p - lv_q
            + (lv_q.exp() + (mu_q - mu_p).pow(2)) / lv_p.exp()
            - 1
        ).sum(dim=-1).mean()

        return recon_loss, kl_loss

    @torch.no_grad()
    def sample(self, eta: torch.Tensor, n_samples: int = 10000):
        self.eval() 
        if eta.dim() == 1:
            eta = eta.unsqueeze(0) # (1, dim_eta), expand할려면 2차로
        
        eta = eta.expand(n_samples, -1).to(device)
        mu_p, lv_p = self.prior(eta)
        eps_z = torch.randn_like(mu_p)
        z = self.reparameterize(mu_p, lv_p, eps_z)

        mu_x, lv_x = self.decoder(z, eta)
        eps_x = torch.randn_like(mu_x)
        samples = self.reparameterize(mu_x, lv_x, eps_x)
        return samples

    @torch.no_grad()
    def price_barrier(self, eta: torch.Tensor, B: float, K: float,
                      r: float, T: float, opt_type: str = 'call', 
                      n_samples: int = 10000):

        samples = self.sample(eta, n_samples) # (N, dim_x) : (X_T, M_T)
        X_T = samples[:, 0]
        S_T = torch.exp(X_T) #  X_T : log_return
        M_T = samples[:, 1]
        
        alive = (M_T > np.log(B)).float() # Knock-out mask (S0=1이므로 ln(B/S0)=ln(B))

        if opt_type == 'call':
            payoff = torch.clamp(S_T - K, min=0.0) * alive
        elif opt_type == 'put':
            payoff = torch.clamp(K - S_T, min=0.0) * alive
        else:
            raise ValueError("opt_type must be 'call' or 'put'")

        return np.exp(-r * T) * payoff.mean().item()

    @torch.no_grad()
    def price_vanilla(self, eta: torch.Tensor, K: float,
                      r: float, T: float, opt_type: str = 'call',
                      n_samples: int = 10000):

        samples = self.sample(eta, n_samples)  # (N, 1): (X_T)
        X_T = samples[:, 0]
        S_T = torch.exp(X_T)

        if opt_type == 'call':
            payoff = torch.clamp(S_T - K, min=0.0)
        elif opt_type == 'put':
            payoff = torch.clamp(K - S_T, min=0.0)
        else:
            raise ValueError("opt_type must be 'call' or 'put'")

        return np.exp(-r * T) * payoff.mean().item()
