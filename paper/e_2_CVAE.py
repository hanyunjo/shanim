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
# BatchNorm utility
# ────────────
def freeze_batchnorm(model: nn.Module, freeze_affine: bool = True):
    # BN의 gamma/beta도 고정.
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()
            if freeze_affine:
                if m.weight is not None:
                    m.weight.requires_grad_(False)
                if m.bias is not None:
                    m.bias.requires_grad_(False)


def set_batchnorm_eval(model: nn.Module):
    """
    cvae.train() 호출 후에도 BatchNorm만 eval 모드로 유지하기 위한 함수.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()


# ────────────
# Sub-networks
# ────────────
def _hidden_mlp(in_dim: int, hidden_dims: list, activation=nn.Tanh, use_bn: bool = False) -> nn.Sequential:
    layers = []
    prev_dim = in_dim

    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(h_dim))
        layers.append(activation())
        prev_dim = h_dim

    return nn.Sequential(*layers)


class RecognitionNet(nn.Module):
    """
    q_φ(z | x, η)
    Input  : [x, η]  dim = dim_x + dim_eta
    Output : μ_z, log σ_z  dim = dim_z each
    """
    def __init__(self, dim_x, dim_eta, dim_z, hidden_dims, use_bn=False):
        super().__init__()
        self.net = _hidden_mlp(dim_x + dim_eta, hidden_dims, use_bn=use_bn)
        self.mu = nn.Linear(hidden_dims[-1], dim_z)
        self.logvar = nn.Linear(hidden_dims[-1], dim_z)

    def forward(self, x, eta):
        h = self.net(torch.cat([x, eta], dim=-1))
        mu = self.mu(h)
        log_var = self.logvar(h)
        log_var = torch.clamp(log_var, min=-10.0, max=5.0)
        return mu, log_var


class PriorNet(nn.Module):
    """
    p_θ(z | η)
    Input  : η  dim = dim_eta
    Output : μ_p, log σ_p  dim = dim_z each
    """
    def __init__(self, dim_eta, dim_z, hidden_dims, use_bn=False):
        super().__init__()
        self.net = _hidden_mlp(dim_eta, hidden_dims, use_bn=use_bn)
        self.mu = nn.Linear(hidden_dims[-1], dim_z)
        self.logvar = nn.Linear(hidden_dims[-1], dim_z)

    def forward(self, eta):
        h = self.net(eta)
        mu = self.mu(h)
        log_var = self.logvar(h)
        log_var = torch.clamp(log_var, min=-10.0, max=5.0)
        return mu, log_var


class DecoderNet(nn.Module):
    """
    p_θ(x | z, η)
    Input  : [z, η]  dim = dim_z + dim_eta
    Output : μ_x, log σ_x  dim = dim_x each
    """
    def __init__(self, dim_z, dim_eta, dim_x, hidden_dims, use_bn=False):
        super().__init__()
        self.net = _hidden_mlp(dim_z + dim_eta, hidden_dims, use_bn=use_bn)
        self.mu = nn.Linear(hidden_dims[-1], dim_x)
        self.logvar = nn.Linear(hidden_dims[-1], dim_x)

    def forward(self, z, eta):
        h = self.net(torch.cat([z, eta], dim=-1))
        mu = self.mu(h)
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
                 hidden_dims: list = None,
                 use_bn: bool = False):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128, 64]

        self.dim_x = dim_x
        self.dim_eta = dim_eta
        self.dim_z = dim_z
        self.hidden_dims = hidden_dims
        self.use_bn = use_bn

        self.recognition = RecognitionNet(dim_x, dim_eta, dim_z, hidden_dims, use_bn=use_bn)
        self.prior = PriorNet(dim_eta, dim_z, hidden_dims, use_bn=use_bn)
        self.decoder = DecoderNet(dim_z, dim_eta, dim_x, hidden_dims, use_bn=use_bn)

    @staticmethod
    def reparameterize(mu, log_var, eps=None):
        # z = μ + ε·σ,  ε ~ N(0, I)
        std = torch.exp(0.5 * log_var)
        if eps is None:
            eps = torch.randn_like(std)
        return mu + eps * std

    # forward + ELBO
    def forward(self, x, eta):
        mu_q, lv_q = self.recognition(x, eta)
        mu_p, lv_p = self.prior(eta)
        z = self.reparameterize(mu_q, lv_q)
        mu_x, lv_x = self.decoder(z, eta)

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
            eta = eta.unsqueeze(0)

        model_device = next(self.parameters()).device
        eta = eta.expand(n_samples, -1).to(model_device)
        mu_p, lv_p = self.prior(eta)
        eps_z = torch.randn_like(mu_p)
        z = self.reparameterize(mu_p, lv_p, eps_z)

        mu_x, lv_x = self.decoder(z, eta)
        eps_x = torch.randn_like(mu_x)
        samples = self.reparameterize(mu_x, lv_x, eps_x)
        return samples

    @torch.no_grad()
    def price_vanilla(self, eta: torch.Tensor, K: float,
                      r: float, T: float, opt_type: str = 'call',
                      n_samples: int = 10000):

        samples = self.sample(eta, n_samples)
        X_T = samples[:, 0]
        S_T = torch.exp(X_T)

        if opt_type == 'call':
            payoff = torch.clamp(S_T - K, min=0.0)
        elif opt_type == 'put':
            payoff = torch.clamp(K - S_T, min=0.0)
        else:
            raise ValueError("opt_type must be 'call' or 'put'")

        return np.exp(-r * T) * payoff.mean().item()
    
    @torch.no_grad()
    def price_barrier(self, eta: torch.Tensor, B: float, K: float,
                      r: float, T: float, opt_type: str = 'call',
                      n_samples: int = 10000):

        samples = self.sample(eta, n_samples)
        X_T = samples[:, 0]
        S_T = torch.exp(X_T)
        M_T = samples[:, 1]

        alive = (M_T > np.log(B)).float()

        if opt_type == 'call':
            payoff = torch.clamp(S_T - K, min=0.0) * alive
        elif opt_type == 'put':
            payoff = torch.clamp(K - S_T, min=0.0) * alive
        else:
            raise ValueError("opt_type must be 'call' or 'put'")

        return np.exp(-r * T) * payoff.mean().item()