import torch
import torch.nn as nn
import numpy as np

if not torch.cuda.is_available():
    raise ValueError("Cannot use GPU cuda")
device = torch.device("cuda")

# ────────────
# Sub-networks
# ────────────
def freeze_batchnorm(model, freeze_affine=True):
    # BN layer는 계속 forward에 사용하고, gamma/beta만 업데이트를 멈춘다.
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()
            if freeze_affine:
                if m.weight is not None:
                    m.weight.requires_grad_(False)
                if m.bias is not None:
                    m.bias.requires_grad_(False)


class ResidualBlock(nn.Module):
    def __init__(self, dim, activation=nn.Tanh, use_bn=False):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim) if use_bn else nn.Identity()
        self.activation = activation()
        self.linear2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim) if use_bn else nn.Identity()

        # Start as an exact identity mapping and learn the residual gradually.
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        residual = self.linear1(x)
        residual = self.bn1(residual)
        residual = self.activation(residual)
        residual = self.linear2(residual)
        residual = self.bn2(residual)
        return x + residual


def _hidden_mlp(in_dim, hidden_dims, activation=nn.Tanh,
                use_bn=False, residual_blocks=0):
    if not isinstance(residual_blocks, (int, np.integer)) or isinstance(residual_blocks, bool):
        raise TypeError("residual_blocks must be a non-negative integer.")
    residual_blocks = int(residual_blocks)
    if residual_blocks < 0:
        raise ValueError("residual_blocks must be >= 0.")
    if not hidden_dims:
        raise ValueError("hidden_dims must contain at least one hidden width.")

    eligible_blocks = 0
    layer_idx = 1
    prev_dim = hidden_dims[0]
    while layer_idx < len(hidden_dims):
        if (
            layer_idx + 1 < len(hidden_dims)
            and hidden_dims[layer_idx] == prev_dim
            and hidden_dims[layer_idx + 1] == prev_dim
        ):
            eligible_blocks += 1
            layer_idx += 2
        else:
            prev_dim = hidden_dims[layer_idx]
            layer_idx += 1

    if residual_blocks > eligible_blocks:
        raise ValueError(
            f"residual_blocks={residual_blocks}, but hidden_dims={hidden_dims} "
            f"has only {eligible_blocks} two-layer equal-width block(s)."
        )

    layers = [nn.Linear(in_dim, hidden_dims[0])]
    if use_bn:
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
    layers.append(activation())

    prev_dim = hidden_dims[0]
    layer_idx = 1
    used_residual_blocks = 0

    while layer_idx < len(hidden_dims):
        use_residual = (
            used_residual_blocks < residual_blocks
            and layer_idx + 1 < len(hidden_dims)
            and hidden_dims[layer_idx] == prev_dim
            and hidden_dims[layer_idx + 1] == prev_dim
        )
        if use_residual:
            layers.append(ResidualBlock(prev_dim, activation=activation, use_bn=use_bn))
            used_residual_blocks += 1
            layer_idx += 2
        else:
            h_dim = hidden_dims[layer_idx]
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(activation())
            prev_dim = h_dim
            layer_idx += 1

    return nn.Sequential(*layers)


class RecognitionNet(nn.Module):
    def __init__(self, dim_x, dim_eta, dim_z, hidden_dims, use_bn=False,
                 residual_blocks=0):
        super().__init__()
        self.net = _hidden_mlp(
            dim_x + dim_eta, hidden_dims, use_bn=use_bn,
            residual_blocks=residual_blocks,
        )
        self.mu = nn.Linear(hidden_dims[-1], dim_z)
        self.logvar = nn.Linear(hidden_dims[-1], dim_z)

    def forward(self, x, eta):
        h = self.net(torch.cat([x, eta], dim=-1))
        mu = self.mu(h)
        log_var = self.logvar(h)
        log_var = torch.clamp(log_var, min=-10.0, max=5.0)
        return mu, log_var


class PriorNet(nn.Module):
    def __init__(self, dim_eta, dim_z, hidden_dims, use_bn=False,
                 residual_blocks=0):
        super().__init__()
        self.net = _hidden_mlp(
            dim_eta, hidden_dims, use_bn=use_bn,
            residual_blocks=residual_blocks,
        )
        self.mu = nn.Linear(hidden_dims[-1], dim_z)
        self.logvar = nn.Linear(hidden_dims[-1], dim_z)

    def forward(self, eta):
        h = self.net(eta)
        mu = self.mu(h)
        log_var = self.logvar(h)
        log_var = torch.clamp(log_var, min=-10.0, max=5.0)
        return mu, log_var


class DecoderNet(nn.Module):
    def __init__(self, dim_z, dim_eta, dim_x, hidden_dims, use_bn=False,
                 residual_blocks=0):
        super().__init__()
        self.net = _hidden_mlp(
            dim_z + dim_eta, hidden_dims, use_bn=use_bn,
            residual_blocks=residual_blocks,
        )
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
    RESIDUAL_LAYOUT = "paired_equal_width_v1"

    def __init__(self,
                 dim_x=2,       # (X_T, M_T)=2, (X_T)=1
                 dim_eta=7,     # BS=3, Heston=7
                 dim_z=8,       # latent dim
                 hidden_dims=None,
                 use_bn=False,
                 residual_blocks=0, # block size가 아니라 block 개수
                 ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128, 64]

        self.dim_x = dim_x
        self.dim_eta = dim_eta
        self.dim_z = dim_z
        if not isinstance(residual_blocks, (int, np.integer)) or isinstance(residual_blocks, bool):
            raise TypeError("residual_blocks must be a non-negative integer.")
        residual_blocks = int(residual_blocks)
        if residual_blocks < 0:
            raise ValueError("residual_blocks must be >= 0.")

        self.hidden_dims = list(hidden_dims)
        self.use_bn = bool(use_bn)
        self.residual_blocks = residual_blocks
        self.residual_layout = self.RESIDUAL_LAYOUT

        common_kwargs = {
            "hidden_dims": self.hidden_dims,
            "use_bn": self.use_bn,
            "residual_blocks": self.residual_blocks,
        }
        self.recognition = RecognitionNet(dim_x, dim_eta, dim_z, **common_kwargs)
        self.prior = PriorNet(dim_eta, dim_z, **common_kwargs)
        self.decoder = DecoderNet(dim_z, dim_eta, dim_x, **common_kwargs)

    @staticmethod
    def reparameterize(mu, log_var, eps=None):
        # z = μ + ε·σ,  ε ~ N(0, I)
        std = torch.exp(0.5 * log_var)
        if eps is None:
            eps = torch.randn_like(std)
        return mu + eps * std

    # forward + ELBO
    def forward(self, x, eta, return_kl_dim=False):
        mu_q, lv_q = self.recognition(x, eta)
        mu_p, lv_p = self.prior(eta)
        z = self.reparameterize(mu_q, lv_q)
        mu_x, lv_x = self.decoder(z, eta)

        # Reconstruction loss : Gaussian NLL
        recon_loss = 0.5 * (
            lv_x + (x - mu_x).pow(2) / lv_x.exp() + np.log(2 * np.pi)
        ).sum(dim=-1).mean()

        # KL per latent dimension: shape (batch, dim_z)
        kl_dim_batch = 0.5 * (
            lv_p - lv_q
            + (lv_q.exp() + (mu_q - mu_p).pow(2)) / lv_p.exp()
            - 1
        )
        kl_dim_mean = kl_dim_batch.mean(dim=0)
        kl_loss = kl_dim_mean.sum()

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
    def sample(self, eta, n_samples = 10000):
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
    def price_vanilla(self, eta, K, r, T,
                      opt_type = 'call',
                      n_samples = 10000
                      ):
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
    def price_barrier(self, eta, B, K, r, T,
                      opt_type = 'call',
                      n_samples = 10000
                      ):
        if self.dim_x < 2:
            raise ValueError(
                "Barrier pricing is unavailable because this CVAE was trained without M_T."
            )

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
    def total_pricing(self, eta, B, K, r, T, n_samples = 10000):
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