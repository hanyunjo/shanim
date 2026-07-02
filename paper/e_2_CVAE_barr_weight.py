import torch
import torch.nn as nn
import numpy as np

if not torch.cuda.is_available():
    raise ValueError("Cannot use GPU cuda")
device = torch.device("cuda")


def freeze_batchnorm(model: nn.Module, freeze_affine: bool = True):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()
            if freeze_affine:
                if m.weight is not None:
                    m.weight.requires_grad_(False)
                if m.bias is not None:
                    m.bias.requires_grad_(False)


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


def gaussian_nll_per_sample(x, mu, logvar):
    logvar = torch.clamp(logvar, min=-10.0, max=5.0)
    nll = 0.5 * (
        logvar + (x - mu).pow(2) / logvar.exp() + np.log(2 * np.pi)
    )
    return nll.sum(dim=-1)


def _normalize_weight(weight, eps=1e-8):
    return weight / (weight.mean().detach() + eps)


def barrier_put_region_weight(
    x_raw,
    S0=1.0,
    K=1.0,
    B=0.8,
    alpha=3.0,
    h=0.05,
    normalize=True,
):
    """
    Weight samples that are ITM for put and near the down barrier.

    x_raw[:, 0] = X_T, x_raw[:, 1] = m_T on raw scale.
    """
    if x_raw.shape[-1] < 2:
        raise ValueError("barrier_put_region_weight requires x_raw with [X_T, m_T].")
    if h <= 0:
        raise ValueError("h must be positive.")

    XT = x_raw[:, 0]
    MT = x_raw[:, 1]
    k_log = float(np.log(K / S0))
    b_log = float(np.log(B / S0))

    put_side = (XT < k_log).to(dtype=x_raw.dtype)
    near_barrier = torch.exp(-torch.abs(MT - b_log) / h)
    weight = 1.0 + alpha * put_side * near_barrier
    weight = weight.detach()
    if normalize:
        weight = _normalize_weight(weight)
    return weight


def barrier_near_weight(
    x_raw,
    S0=1.0,
    B=0.8,
    alpha=3.0,
    h=0.05,
    normalize=True,
):
    """
    Weight samples near the down barrier regardless of call/put payoff side.

    x_raw[:, 1] = m_T on raw scale.
    """
    if x_raw.shape[-1] < 2:
        raise ValueError("barrier_near_weight requires x_raw with [X_T, m_T].")
    if h <= 0:
        raise ValueError("h must be positive.")

    MT = x_raw[:, 1]
    b_log = float(np.log(B / S0))
    near_barrier = torch.exp(-torch.abs(MT - b_log) / h)
    weight = 1.0 + alpha * near_barrier
    weight = weight.detach()
    if normalize:
        weight = _normalize_weight(weight)
    return weight


class RecognitionNet(nn.Module):
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


class CVAEBarrWeight(nn.Module):
    def __init__(self,
                 dim_x: int = 2,
                 dim_eta: int = 7,
                 dim_z: int = 8,
                 hidden_dims: list = None,
                 use_bn: bool = False,
                 weight_mode: str = "barrier_put",
                 weight_alpha: float = 3.0,
                 weight_h: float = 0.05,
                 weight_normalize: bool = True,
                 S0: float = 1.0,
                 K: float = 1.0,
                 B: float = 0.8):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128, 64]

        self.dim_x = dim_x
        self.dim_eta = dim_eta
        self.dim_z = dim_z
        self.hidden_dims = hidden_dims
        self.use_bn = use_bn

        self.weight_mode = weight_mode
        self.weight_alpha = float(weight_alpha)
        self.weight_h = float(weight_h)
        self.weight_normalize = bool(weight_normalize)
        self.S0 = float(S0)
        self.K = float(K)
        self.B = float(B)

        self.recognition = RecognitionNet(dim_x, dim_eta, dim_z, hidden_dims, use_bn=use_bn)
        self.prior = PriorNet(dim_eta, dim_z, hidden_dims, use_bn=use_bn)
        self.decoder = DecoderNet(dim_z, dim_eta, dim_x, hidden_dims, use_bn=use_bn)

    @staticmethod
    def reparameterize(mu, log_var, eps=None):
        std = torch.exp(0.5 * log_var)
        if eps is None:
            eps = torch.randn_like(std)
        return mu + eps * std

    def reconstruction_weight(self, x_raw, weight_mode=None):
        mode = self.weight_mode if weight_mode is None else weight_mode
        if mode is None or mode == "none":
            return torch.ones(x_raw.shape[0], dtype=x_raw.dtype, device=x_raw.device)
        if mode in ("barrier_put", "put_region", "barrier_put_region"):
            return barrier_put_region_weight(
                x_raw,
                S0=self.S0,
                K=self.K,
                B=self.B,
                alpha=self.weight_alpha,
                h=self.weight_h,
                normalize=self.weight_normalize,
            )
        if mode in ("barrier_near", "near"):
            return barrier_near_weight(
                x_raw,
                S0=self.S0,
                B=self.B,
                alpha=self.weight_alpha,
                h=self.weight_h,
                normalize=self.weight_normalize,
            )
        raise ValueError("weight_mode must be one of None, 'none', 'barrier_put', or 'barrier_near'.")

    def forward(self, x, eta, return_kl_dim=False, x_raw=None, weight_mode=None, return_weight_stats=False):
        """
        x is the reconstruction target used by the decoder.
        x_raw must be raw-scale [X_T, m_T] when x is normalized.
        """
        mu_q, lv_q = self.recognition(x, eta)
        mu_p, lv_p = self.prior(eta)
        z = self.reparameterize(mu_q, lv_q)
        mu_x, lv_x = self.decoder(z, eta)

        nll_i = gaussian_nll_per_sample(x, mu_x, lv_x)
        x_for_weight = x if x_raw is None else x_raw
        weight = self.reconstruction_weight(x_for_weight, weight_mode=weight_mode)
        recon_loss = (weight * nll_i).mean()

        kl_dim_batch = 0.5 * (
            lv_p - lv_q
            + (lv_q.exp() + (mu_q - mu_p).pow(2)) / lv_p.exp()
            - 1
        )
        kl_dim_mean = kl_dim_batch.mean(dim=0)
        kl_loss = kl_dim_mean.sum()

        if return_weight_stats:
            weight_stats = {
                "mean": weight.mean().detach(),
                "min": weight.min().detach(),
                "max": weight.max().detach(),
            }
            if return_kl_dim:
                return recon_loss, kl_loss, kl_dim_mean, weight_stats
            return recon_loss, kl_loss, weight_stats

        if return_kl_dim:
            return recon_loss, kl_loss, kl_dim_mean
        return recon_loss, kl_loss

    @torch.no_grad()
    def latent_kl_by_dim(self, x, eta):
        was_training = self.training
        self.eval()
        _, _, kl_dim_mean = self.forward(x, eta, return_kl_dim=True)
        if was_training:
            self.train()
        return kl_dim_mean

    @torch.no_grad()
    def sample(self, eta: torch.Tensor, n_samples: int = 10000):
        self.eval()
        if eta.dim() == 1:
            eta = eta.unsqueeze(0)

        model_device = next(self.parameters()).device
        eta = eta.expand(n_samples, -1).to(model_device)
        mu_p, lv_p = self.prior(eta)
        z = self.reparameterize(mu_p, lv_p, torch.randn_like(mu_p))

        mu_x, lv_x = self.decoder(z, eta)
        samples = self.reparameterize(mu_x, lv_x, torch.randn_like(mu_x))
        return samples

    @torch.no_grad()
    def price_vanilla(self, eta: torch.Tensor, K: float, r: float, T: float, opt_type: str = 'call',
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
    def price_barrier(self, eta: torch.Tensor, B: float, K: float, r: float, T: float, opt_type: str = 'call',
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

    @torch.no_grad()
    def total_pricing(self, eta: torch.Tensor, B: float, K: float,
                      r: float, T: float, n_samples: int = 10000):
        samples = self.sample(eta, n_samples)
        if samples.shape[1] < 2:
            raise ValueError("total_pricing requires samples with [X_T, M_T].")

        X_T = samples[:, 0]
        M_T = samples[:, 1]
        S_T = torch.exp(X_T)
        discount = float(np.exp(-r * T))

        call_payoff = torch.clamp(S_T - K, min=0.0)
        put_payoff = torch.clamp(K - S_T, min=0.0)
        alive = (M_T > np.log(B)).float()

        return {
            "van_call": discount * call_payoff.mean().item(),
            "van_put": discount * put_payoff.mean().item(),
            "barr_call": discount * (call_payoff * alive).mean().item(),
            "barr_put": discount * (put_payoff * alive).mean().item(),
        }


# Drop-in import compatibility:
# from e_2_CVAE_barr_weight import CVAE
CVAE = CVAEBarrWeight
