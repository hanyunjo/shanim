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
    def sample_valid(self, eta, n_samples=10000, max_resample_rounds=100,
                     valid_resample=True, initial_samples=None):
        n_samples = int(n_samples)
        max_resample_rounds = int(max_resample_rounds)
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if max_resample_rounds < 1:
            raise ValueError("max_resample_rounds must be >= 1")
        if not isinstance(valid_resample, (bool, np.bool_)):
            raise TypeError("valid_resample must be True or False")
        valid_resample = bool(valid_resample)

        valid_batches = []
        accepted_samples = 0
        attempted_samples = 0
        rejected_samples = 0
        sampling_rounds = max_resample_rounds if valid_resample else 1

        for round_idx in range(sampling_rounds):
            remaining = n_samples - accepted_samples
            if remaining <= 0:
                break

            draw_count = remaining
            if round_idx == 0 and initial_samples is not None:
                candidates = initial_samples
                if candidates.shape[0] != n_samples:
                    raise ValueError(
                        "initial_samples must contain exactly n_samples rows"
                    )
                draw_count = n_samples
            else:
                candidates = self.sample(eta, draw_count)
            if candidates.ndim != 2 or candidates.shape[1] < 1:
                raise ValueError(
                    "CVAE samples must have shape (N, 1) or (N, 2+)"
                )

            finite = torch.isfinite(candidates).all(dim=1)
            if candidates.shape[1] >= 2:
                X_T = candidates[:, 0]
                M_T = candidates[:, 1]
                valid = finite & (
                    M_T <= torch.minimum(torch.zeros_like(X_T), X_T) + 1e-6
                )
            else: # for X_T
                valid = finite

            attempted_samples += draw_count
            rejected_samples += int((~valid).sum().item())
            accepted = candidates[valid]
            if accepted.numel() > 0:
                accepted = accepted[:remaining]
                valid_batches.append(accepted)
                accepted_samples += int(accepted.shape[0])
            del candidates, finite, valid, accepted

        if accepted_samples == 0:
            raise RuntimeError(
                f"No valid CVAE paths were found in {attempted_samples:,} draws."
            )
        if valid_resample and accepted_samples < n_samples:
            raise RuntimeError(
                f"Could obtain only {accepted_samples:,} valid paths after "
                f"{attempted_samples:,} draws. The CVAE is producing too many "
                "invalid running-minimum samples."
            )

        samples = torch.cat(valid_batches, dim=0)
        regenerated_samples = max(attempted_samples - n_samples, 0)
        discarded_fraction = rejected_samples / attempted_samples
        regenerated_fraction = regenerated_samples / n_samples
        diagnostics = {
            "validation_applied": True,
            "valid_resample": valid_resample,
            "invalid_path_fraction": discarded_fraction,
            "rejected_path_count": rejected_samples,
            "total_drawn_samples": attempted_samples,
            "accepted_path_count": accepted_samples,
            "requested_sample_count": n_samples,
            "used_sample_count": accepted_samples,
            "discarded_sample_count": rejected_samples,
            "discarded_sample_fraction": discarded_fraction,
            "discarded_sample_pct": 100.0 * discarded_fraction,
            "regenerated_sample_count": regenerated_samples,
            "regenerated_sample_fraction": regenerated_fraction,
            "regenerated_sample_pct": 100.0 * regenerated_fraction,
        }
        self.last_sampling_diagnostics = diagnostics
        return samples, diagnostics

    @torch.no_grad()
    def _samples_for_pricing(self, eta, n_samples, validate_samples=True,
                             valid_resample=True, initial_samples=None):
        if not isinstance(validate_samples, (bool, np.bool_)):
            raise TypeError("validate_samples must be True or False")
        if not isinstance(valid_resample, (bool, np.bool_)):
            raise TypeError("valid_resample must be True or False")

        n_samples = int(n_samples)
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")

        if validate_samples:
            return self.sample_valid(
                eta, n_samples, valid_resample=valid_resample,
                initial_samples=initial_samples
            )

        samples = (
            initial_samples if initial_samples is not None
            else self.sample(eta, n_samples)
        )
        diagnostics = {
            "validation_applied": False,
            "valid_resample": False,
            "invalid_path_fraction": None,
            "rejected_path_count": None,
            "total_drawn_samples": n_samples,
            "accepted_path_count": None,
            "requested_sample_count": n_samples,
            "used_sample_count": n_samples,
            "discarded_sample_count": None,
            "discarded_sample_fraction": None,
            "discarded_sample_pct": None,
            "regenerated_sample_count": 0,
            "regenerated_sample_fraction": 0.0,
            "regenerated_sample_pct": 0.0,
        }
        self.last_sampling_diagnostics = diagnostics
        return samples, diagnostics

    @torch.no_grad()
    def _apply_mt_correction(self, X_T, M_T, mt_corr=False):
        if not isinstance(mt_corr, (bool, np.bool_)):
            raise TypeError("mt_corr must be True or False")

        if mt_corr:
            mt_upper = torch.minimum(torch.zeros_like(X_T), X_T)
            corrected_mask = M_T > mt_upper
            corrected_M_T = torch.minimum(M_T, mt_upper)
        else:
            corrected_mask = torch.zeros_like(M_T, dtype=torch.bool)
            corrected_M_T = M_T

        corrected_count = int(corrected_mask.sum().item())
        corrected_fraction = corrected_count / int(M_T.shape[0])
        correction_diagnostics = {
            "mt_corr_applied": bool(mt_corr),
            "mt_corrected_sample_count": corrected_count,
            "mt_corrected_sample_fraction": corrected_fraction,
            "mt_corrected_sample_pct": 100.0 * corrected_fraction,
        }
        return corrected_M_T, correction_diagnostics

    @torch.no_grad()
    def price_vanilla(self, eta, K, r, T,
                      opt_type = 'call',
                      n_samples = 10000,
                      validate_samples = True,
                      valid_resample = True
                      ):
        # Vanilla payoff depends only on X_T, so M_T validity is intentionally ignored.
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
                      n_samples = 10000,
                      validate_samples = True,
                      valid_resample = True,
                      mt_corr = False
                      ):
        if self.dim_x < 2:
            raise ValueError(
                "Barrier pricing is unavailable because this CVAE was trained without M_T."
            )

        samples, diagnostics = self._samples_for_pricing(
            eta, n_samples, validate_samples=validate_samples,
            valid_resample=valid_resample
        )
        X_T = samples[:, 0]
        S_T = torch.exp(X_T)
        if validate_samples and not torch.isfinite(S_T).all():
            raise FloatingPointError(
                "A valid CVAE path overflowed during exp(X_T)."
            )
        M_T = samples[:, 1]
        M_T, correction_diagnostics = self._apply_mt_correction(
            X_T, M_T, mt_corr=mt_corr
        )
        diagnostics.update(correction_diagnostics)
        self.last_sampling_diagnostics = diagnostics

        alive = (M_T > np.log(B)).float()

        if opt_type == 'call':
            payoff = torch.clamp(S_T - K, min=0.0) * alive
        elif opt_type == 'put':
            payoff = torch.clamp(K - S_T, min=0.0) * alive
        else:
            raise ValueError("opt_type must be 'call' or 'put'")

        return np.exp(-r * T) * payoff.mean().item()

    @torch.no_grad()
    def total_pricing(self, eta, B, K, r, T, n_samples = 10000,
                      validate_samples = True, valid_resample = True,
                      mt_corr = False):
        if self.dim_x < 2:
            raise ValueError("total_pricing requires samples with [X_T, M_T].")

        # Use every initially generated X_T for vanilla pricing.
        raw_samples = self.sample(eta, n_samples)
        vanilla_X_T = raw_samples[:, 0]
        vanilla_S_T = torch.exp(vanilla_X_T)
        vanilla_call = torch.clamp(vanilla_S_T - K, min=0.0)
        vanilla_put = torch.clamp(K - vanilla_S_T, min=0.0)

        # Apply the running-minimum validity rule only to barrier pricing.
        barrier_samples, diagnostics = self._samples_for_pricing(
            eta, n_samples, validate_samples=validate_samples,
            valid_resample=valid_resample, initial_samples=raw_samples
        )
        barrier_X_T = barrier_samples[:, 0]
        M_T = barrier_samples[:, 1]
        M_T, correction_diagnostics = self._apply_mt_correction(
            barrier_X_T, M_T, mt_corr=mt_corr
        )
        diagnostics.update(correction_diagnostics)
        self.last_sampling_diagnostics = diagnostics
        barrier_S_T = torch.exp(barrier_X_T)
        if validate_samples and not torch.isfinite(barrier_S_T).all():
            raise FloatingPointError(
                "A valid CVAE path overflowed during exp(X_T)."
            )

        barrier_call = torch.clamp(barrier_S_T - K, min=0.0)
        barrier_put = torch.clamp(K - barrier_S_T, min=0.0)
        alive = (M_T > np.log(B)).float()
        discount = float(np.exp(-r * T))

        return {
            "van_call": discount * vanilla_call.mean().item(),
            "van_put": discount * vanilla_put.mean().item(),
            "barr_call": discount * (barrier_call * alive).mean().item(),
            "barr_put": discount * (barrier_put * alive).mean().item(),
            "vanilla_sample_count": int(raw_samples.shape[0]),
            **diagnostics,
        }

