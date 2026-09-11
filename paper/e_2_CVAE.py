import torch
import torch.nn as nn
import numpy as np

if not torch.cuda.is_available():
    raise ValueError("Cannot use GPU cuda")
device = torch.device("cuda")


def _normalize_tail_multiple(multiple):
    if isinstance(multiple, (bool, np.bool_)) or not isinstance(
        multiple, (int, float, np.integer, np.floating)
    ):
        raise TypeError("multiple must be a real number")
    multiple = float(multiple)
    if not np.isfinite(multiple) or multiple < 0 or (0 < multiple <= 1):
        raise ValueError("multiple must be 0 (disabled) or greater than 1")
    return multiple

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
                     valid_resample=True, initial_samples=None,
                     validate_support=True, multiple=0):
        n_samples = int(n_samples)
        max_resample_rounds = int(max_resample_rounds)
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if max_resample_rounds < 1:
            raise ValueError("max_resample_rounds must be >= 1")
        if not isinstance(valid_resample, (bool, np.bool_)):
            raise TypeError("valid_resample must be True or False")
        valid_resample = bool(valid_resample)
        if not isinstance(validate_support, (bool, np.bool_)):
            raise TypeError("validate_support must be True or False")
        validate_support = bool(validate_support)
        multiple = _normalize_tail_multiple(multiple)
        tail_filter_applied = multiple > 0
        xt_upper = float(np.log(multiple)) if tail_filter_applied else None

        valid_batches = []
        accepted_samples = 0
        attempted_samples = 0
        rejected_samples = 0
        support_rejected_samples = 0
        tail_rejected_samples = 0
        must_refill = valid_resample or tail_filter_applied
        sampling_rounds = max_resample_rounds if must_refill else 1

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
            if validate_support and candidates.shape[1] >= 2:
                X_T = candidates[:, 0]
                M_T = candidates[:, 1]
                support_invalid = (~finite) | ~(
                    M_T <= torch.minimum(torch.zeros_like(X_T), X_T) + 1e-6
                )
            elif validate_support:
                support_invalid = ~finite
            else:
                support_invalid = torch.zeros(
                    candidates.shape[0], dtype=torch.bool,
                    device=candidates.device,
                )
            tail_invalid = (
                (~finite) | (candidates[:, 0] >= xt_upper)
                if tail_filter_applied
                else torch.zeros_like(support_invalid)
            )
            invalid = support_invalid | tail_invalid
            valid = ~invalid

            attempted_samples += draw_count
            rejected_samples += int(invalid.sum().item())
            support_rejected_samples += int(support_invalid.sum().item())
            tail_rejected_samples += int(tail_invalid.sum().item())
            accepted = candidates[valid]
            if accepted.numel() > 0:
                accepted = accepted[:remaining]
                valid_batches.append(accepted)
                accepted_samples += int(accepted.shape[0])
            del candidates, finite, support_invalid, tail_invalid, invalid, valid, accepted

        if accepted_samples == 0:
            raise RuntimeError(
                f"No valid CVAE paths were found in {attempted_samples:,} draws."
            )
        if must_refill and accepted_samples < n_samples:
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
            "validation_applied": validate_support,
            "valid_resample": must_refill,
            "valid_resample_requested": valid_resample,
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
            "support_invalid_count": support_rejected_samples,
            "tail_filter_applied": tail_filter_applied,
            "multiple": multiple,
            "X_T_upper": xt_upper,
            "tail_rejected_sample_count": tail_rejected_samples,
            "tail_rejected_sample_fraction": tail_rejected_samples / attempted_samples,
            "tail_rejected_sample_pct": 100.0 * tail_rejected_samples / attempted_samples,
        }
        self.last_sampling_diagnostics = diagnostics
        return samples, diagnostics

    @torch.no_grad()
    def _samples_for_pricing(self, eta, n_samples, validate_samples=True,
                             valid_resample=True, initial_samples=None,
                             multiple=0):
        if not isinstance(validate_samples, (bool, np.bool_)):
            raise TypeError("validate_samples must be True or False")
        if not isinstance(valid_resample, (bool, np.bool_)):
            raise TypeError("valid_resample must be True or False")
        multiple = _normalize_tail_multiple(multiple)
        tail_filter_applied = multiple > 0

        n_samples = int(n_samples)
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")

        if validate_samples or tail_filter_applied:
            return self.sample_valid(
                eta, n_samples, valid_resample=valid_resample,
                initial_samples=initial_samples,
                validate_support=validate_samples,
                multiple=multiple,
            )

        samples = (
            initial_samples if initial_samples is not None
            else self.sample(eta, n_samples)
        )
        diagnostics = {
            "validation_applied": False,
            "valid_resample": False,
            "valid_resample_requested": bool(valid_resample),
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
            "support_invalid_count": None,
            "tail_filter_applied": False,
            "multiple": multiple,
            "X_T_upper": None,
            "tail_rejected_sample_count": 0,
            "tail_rejected_sample_fraction": 0.0,
            "tail_rejected_sample_pct": 0.0,
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
    def _barrier_survival_weight(self, samples, M_T, B, sigma=None, dt=0.001,
                                 bbc_correction=True):
        """Return the down-barrier survival weight for generated path features.

        dim_x=2 uses the discrete running-minimum indicator.  dim_x=3 uses the
        BS Brownian-bridge variance ``sigma**2 * dt`` and dim_x=4 uses the
        generated local average variance ``v_avg * dt``.  The bridge correction
        covers the stored minimum-to-neighbour interval represented by M_T and
        M_next.
        """
        if not isinstance(bbc_correction, (bool, np.bool_)):
            raise TypeError("bbc_correction must be True or False")
        bbc_correction = bool(bbc_correction)
        B = float(B)
        if not np.isfinite(B) or B <= 0.0:
            raise ValueError("B must be a positive finite number")

        barrier_log = float(np.log(B))
        discrete_alive = M_T > barrier_log
        diagnostics = {
            "bbc_correction_requested": bbc_correction,
            "bbc_correction_applied": False,
            "bbc_model": None,
            "bbc_dt": None,
            "bbc_cross_probability_mean": 0.0,
            "bbc_cross_probability_max": 0.0,
            "bbc_expected_additional_knockout_count": 0.0,
            "bbc_survival_weight_mean": float(
                discrete_alive.to(dtype=M_T.dtype).mean().item()
            ),
            "bbc_m_next_corrected_count": 0,
            "bbc_v_avg_corrected_count": 0,
        }

        if self.dim_x < 3 or not bbc_correction:
            return discrete_alive.to(dtype=M_T.dtype), diagnostics

        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be a positive finite number for BBC pricing")

        M_next_raw = samples[:, 2]
        # M_T is the discrete path minimum, so its neighbouring point cannot be
        # below it. Correct occasional support violations from Gaussian decoding.
        finite_pair = torch.isfinite(M_T) & torch.isfinite(M_next_raw)
        M_next = torch.maximum(M_next_raw, M_T)
        m_next_corrected = finite_pair & (M_next_raw < M_T)

        if self.dim_x == 3:
            if sigma is None:
                raise ValueError(
                    "dim_x=3 BBC pricing requires the raw (unscaled) BS sigma. "
                    "Pass sigma=... to price_barrier() or total_pricing()."
                )
            sigma = float(sigma)
            if not np.isfinite(sigma) or sigma <= 0.0:
                raise ValueError("sigma must be a positive finite number")
            local_variance = torch.full_like(M_T, sigma ** 2)
            variance_corrected = torch.zeros_like(M_T, dtype=torch.bool)
            bbc_model = "bs"
        else:
            v_avg_raw = samples[:, 3]
            finite_variance = torch.isfinite(v_avg_raw)
            variance_corrected = (~finite_variance) | (v_avg_raw < 0.0)
            local_variance = torch.nan_to_num(
                v_avg_raw, nan=0.0, posinf=0.0, neginf=0.0
            ).clamp_min(0.0)
            finite_pair = finite_pair & finite_variance
            bbc_model = "heston"

        endpoint_alive = finite_pair & discrete_alive & (M_next > barrier_log)
        distance_min = (M_T - barrier_log).clamp_min(0.0)
        distance_next = (M_next - barrier_log).clamp_min(0.0)
        denominator = local_variance * dt + 1e-10
        exponent = -2.0 * distance_min * distance_next / denominator
        p_cross = torch.where(
            endpoint_alive,
            torch.exp(torch.clamp(exponent, max=0.0)),
            torch.zeros_like(M_T),
        ).clamp_(0.0, 1.0)

        # Conditional expectation of the MC Bernoulli bridge test.  This has the
        # same expected payoff as u < p_cross but avoids extra pricing variance.
        survival = torch.where(
            endpoint_alive,
            1.0 - p_cross,
            torch.zeros_like(M_T),
        )
        diagnostics.update({
            "bbc_correction_applied": True,
            "bbc_model": bbc_model,
            "bbc_dt": dt,
            "bbc_cross_probability_mean": float(p_cross.mean().item()),
            "bbc_cross_probability_max": float(p_cross.max().item()),
            "bbc_expected_additional_knockout_count": float(p_cross.sum().item()),
            "bbc_survival_weight_mean": float(survival.mean().item()),
            "bbc_m_next_corrected_count": int(m_next_corrected.sum().item()),
            "bbc_v_avg_corrected_count": int(variance_corrected.sum().item()),
        })
        return survival, diagnostics

    @torch.no_grad()
    def price_vanilla(self, eta, K, r, T,
                      opt_type = 'call',
                      n_samples = 10000,
                      validate_samples = True,
                      valid_resample = True,
                      multiple = 0,
                      ):
        # Vanilla payoff depends only on X_T, so M_T validity is intentionally ignored.
        del validate_samples
        raw_samples = self.sample(eta, n_samples)
        samples, diagnostics = self._samples_for_pricing(
            eta, n_samples, validate_samples=False,
            valid_resample=valid_resample, initial_samples=raw_samples,
            multiple=multiple,
        )
        self.last_sampling_diagnostics = diagnostics
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
                      mt_corr = False,
                      sigma = None,
                      dt = 0.001,
                      bbc_correction = True,
                      multiple = 0,
                      ):
        if self.dim_x < 2:
            raise ValueError(
                "Barrier pricing is unavailable because this CVAE was trained without M_T."
            )

        samples, diagnostics = self._samples_for_pricing(
            eta, n_samples, validate_samples=validate_samples,
            valid_resample=valid_resample, multiple=multiple,
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

        survival, bbc_diagnostics = self._barrier_survival_weight(
            samples, M_T, B, sigma=sigma, dt=dt,
            bbc_correction=bbc_correction,
        )
        diagnostics.update(bbc_diagnostics)
        self.last_sampling_diagnostics = diagnostics

        if opt_type == 'call':
            payoff = torch.clamp(S_T - K, min=0.0) * survival
        elif opt_type == 'put':
            payoff = torch.clamp(K - S_T, min=0.0) * survival
        else:
            raise ValueError("opt_type must be 'call' or 'put'")

        return np.exp(-r * T) * payoff.mean().item()

    @torch.no_grad()
    def total_pricing(self, eta, B, K, r, T, n_samples = 10000,
                      validate_samples = True, valid_resample = True,
                      mt_corr = False, sigma = None, dt = 0.001,
                      bbc_correction = True, multiple = 0):
        multiple = _normalize_tail_multiple(multiple)
        tail_filter_applied = multiple > 0

        # Use every initially generated X_T for vanilla pricing.
        raw_samples = self.sample(eta, n_samples)
        prefilter_diagnostics = None
        if tail_filter_applied:
            raw_samples, prefilter_diagnostics = self._samples_for_pricing(
                eta, n_samples,
                validate_samples=validate_samples and self.dim_x >= 2,
                valid_resample=valid_resample,
                initial_samples=raw_samples,
                multiple=multiple,
            )
        vanilla_X_T = raw_samples[:, 0]
        vanilla_S_T = torch.exp(vanilla_X_T)
        if not torch.isfinite(vanilla_S_T).all():
            raise FloatingPointError("A CVAE path overflowed during exp(X_T).")
        vanilla_call = torch.clamp(vanilla_S_T - K, min=0.0)
        vanilla_put = torch.clamp(K - vanilla_S_T, min=0.0)
        discount = float(np.exp(-r * T))

        vanilla_results = {
            "van_call": discount * vanilla_call.mean().item(),
            "van_put": discount * vanilla_put.mean().item(),
            "vanilla_sample_count": int(raw_samples.shape[0]),
            "model_output_dim": int(self.dim_x),
            "has_running_minimum": bool(self.dim_x >= 2),
        }

        # An X_T-only model contains no path-minimum information. It can price
        # vanilla options, but barrier payoffs must not be fabricated from it.
        if self.dim_x < 2:
            if prefilter_diagnostics is None:
                _, diagnostics = self._samples_for_pricing(
                    eta, n_samples, validate_samples=False,
                    valid_resample=valid_resample,
                    initial_samples=raw_samples, multiple=multiple,
                )
            else:
                diagnostics = prefilter_diagnostics
            diagnostics.update({
                "mt_corr_applied": False,
                "mt_corrected_sample_count": 0,
                "mt_corrected_sample_fraction": 0.0,
                "mt_corrected_sample_pct": 0.0,
                "bbc_correction_requested": bool(bbc_correction),
                "bbc_correction_applied": False,
                "bbc_model": None,
                "bbc_dt": None,
                "bbc_cross_probability_mean": 0.0,
                "bbc_cross_probability_max": 0.0,
                "bbc_expected_additional_knockout_count": 0.0,
                "bbc_survival_weight_mean": None,
                "bbc_m_next_corrected_count": 0,
                "bbc_v_avg_corrected_count": 0,
            })
            self.last_sampling_diagnostics = diagnostics
            return {
                **vanilla_results,
                "priced_option_keys": ["van_call", "van_put"],
                **diagnostics,
            }

        # Apply the running-minimum validity rule only to barrier pricing.
        if prefilter_diagnostics is None:
            barrier_samples, diagnostics = self._samples_for_pricing(
                eta, n_samples, validate_samples=validate_samples,
                valid_resample=valid_resample, initial_samples=raw_samples,
                multiple=multiple,
            )
        else:
            barrier_samples = raw_samples
            diagnostics = prefilter_diagnostics
        barrier_X_T = barrier_samples[:, 0]
        M_T = barrier_samples[:, 1]
        M_T, correction_diagnostics = self._apply_mt_correction(
            barrier_X_T, M_T, mt_corr=mt_corr
        )
        diagnostics.update(correction_diagnostics)
        barrier_S_T = torch.exp(barrier_X_T)
        if validate_samples and not torch.isfinite(barrier_S_T).all():
            raise FloatingPointError(
                "A valid CVAE path overflowed during exp(X_T)."
            )

        barrier_call = torch.clamp(barrier_S_T - K, min=0.0)
        barrier_put = torch.clamp(K - barrier_S_T, min=0.0)
        survival, bbc_diagnostics = self._barrier_survival_weight(
            barrier_samples, M_T, B, sigma=sigma, dt=dt,
            bbc_correction=bbc_correction,
        )
        diagnostics.update(bbc_diagnostics)
        self.last_sampling_diagnostics = diagnostics

        return {
            **vanilla_results,
            "barr_call": discount * (barrier_call * survival).mean().item(),
            "barr_put": discount * (barrier_put * survival).mean().item(),
            "priced_option_keys": [
                "van_call", "van_put", "barr_call", "barr_put"
            ],
            **diagnostics,
        }
