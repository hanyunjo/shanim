import torch
import numpy as np

from e_2_CVAE import CVAE as BaseCVAE


def gaussian_nll_per_sample(x, mu, lv):
    lv = torch.clamp(lv, min=-10.0, max=5.0)
    nll = 0.5 * (
        lv + (x - mu).pow(2) / lv.exp() + np.log(2 * np.pi)
    )
    return nll.sum(dim=-1)


def _normalize_weight(weight, eps=1e-8):
    return weight / (weight.mean().detach() + eps)


class CVAEBarrWeight(BaseCVAE):
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
                B: float = 0.8,
                cvae_type: str = "barr_weight",
                ):
        super().__init__(
            dim_x=dim_x,
            dim_eta=dim_eta,
            dim_z=dim_z,
            hidden_dims=hidden_dims,
            use_bn=use_bn,
        )
        self.weight_mode = weight_mode
        self.weight_alpha = float(weight_alpha)
        self.weight_h = float(weight_h)
        self.weight_normalize = bool(weight_normalize)
        self.S0 = float(S0)
        self.K = float(K)
        self.B = float(B)
        self.cvae_type = cvae_type

    def reconstruction_weight(self, x_raw):
        mode = self.weight_mode
        cvae_type = self.cvae_type
        if cvae_type not in ("barr_weight", "barrweight", "normal_weight", "normalweight"):
            raise ValueError("reconstruction_weight supports cvae_type 'barr_weight' or 'normal_weight'.")
        if mode not in ("barrier_put", "barrierput", "barrier_near", "barriernear", "all_put"):
            raise ValueError("weight_mode must be one of 'barrier_put', 'barrier_near', or 'all_put'.")
        if self.weight_h <= 0:
            raise ValueError("weight_h must be positive.")
        if x_raw.shape[-1] < 1:
            raise ValueError("require x_raw with at least X_T.")

        XT = x_raw[:, 0]
        k_log = float(np.log(self.K / self.S0))
        put_side = (XT < k_log).to(dtype=x_raw.dtype)

        if cvae_type in ("barr_weight", "barrweight"):
            if x_raw.shape[-1] < 2:
                raise ValueError("require x_raw with [X_T, M_T].")
            MT = x_raw[:, 1]
            b_log = float(np.log(self.B / self.S0))
            near_barrier = torch.exp(-torch.abs(MT - b_log) / self.weight_h)

            if mode in ("barrier_put", "barrierput"):
                weight = 1.0 + self.weight_alpha * put_side * near_barrier
            elif mode in ("barrier_near", "barriernear"):
                weight = 1.0 + self.weight_alpha * near_barrier
            else:
                raise ValueError("weight_mode must be one of 'barrier_put' or 'barrier_near' for cvae_type='barr_weight'.")

        elif cvae_type in ("normal_weight", "normalweight"):
            if mode in ("barrier_put", "barrierput"):
                if x_raw.shape[-1] < 2:
                    raise ValueError("require x_raw with [X_T, M_T].")
                MT = x_raw[:, 1]
                b_log = float(np.log(self.B / self.S0))
                in_barrier = (MT > b_log).to(dtype=x_raw.dtype)
                weight = 1.0 + self.weight_alpha * put_side * in_barrier
            elif mode == "all_put":
                weight = 1.0 + self.weight_alpha * put_side
            else:
                raise ValueError("weight_mode must be one of 'barrier_put' or 'all_put' for cvae_type='normal_weight'.")

        weight = weight.detach()
        if self.weight_normalize:
            weight = _normalize_weight(weight)
        return weight

    def additive_reconstruction_loss(self, nll_i, x_raw):
        mode = self.weight_mode
        if mode not in ("barrier_put", "barrierput", "all_put"):
            raise ValueError("weight_mode must be one of 'barrier_put' or 'all_put' for cvae_type='add_put_loss'.")
        if x_raw.shape[-1] < 1:
            raise ValueError("require x_raw with at least X_T.")

        XT = x_raw[:, 0]
        k_log = float(np.log(self.K / self.S0))
        put_mask = XT < k_log

        if mode in ("barrier_put", "barrierput"):
            if x_raw.shape[-1] < 2:
                raise ValueError("require x_raw with [X_T, M_T].")
            MT = x_raw[:, 1]
            b_log = float(np.log(self.B / self.S0))
            region_mask = put_mask & (MT > b_log)
        else:
            region_mask = put_mask

        recon_all = nll_i.mean()
        mask = region_mask.to(dtype=nll_i.dtype)
        recon_region = (nll_i * mask).sum() / (mask.sum() + 1e-8)
        recon_loss = recon_all + self.weight_alpha * recon_region
        stats = {
            "recon_all": recon_all.detach(),
            "recon_region": recon_region.detach(),
            "region_count": region_mask.sum().detach(),
            "region_ratio": region_mask.to(dtype=nll_i.dtype).mean().detach(),
        }
        return recon_loss, stats

    def forward(self,
                x,
                eta,
                return_kl_dim=False,
                return_weight_stats=False,
                ):
        mu_q, lv_q = self.recognition(x, eta)
        mu_p, lv_p = self.prior(eta)
        z = self.reparameterize(mu_q, lv_q)
        mu_x, lv_x = self.decoder(z, eta)

        nll_i = gaussian_nll_per_sample(x, mu_x, lv_x)
        if self.cvae_type in ("add_put_loss", "additive_put_loss"):
            recon_loss, weight_stats = self.additive_reconstruction_loss(nll_i, x)
        else:
            weight = self.reconstruction_weight(x_raw=x)
            recon_loss = (weight * nll_i).mean()
            weight_stats = {
                "mean": weight.mean().detach(),
                "min": weight.min().detach(),
                "max": weight.max().detach(),
            }

        kl_dim_batch = 0.5 * (
            lv_p - lv_q
            + (lv_q.exp() + (mu_q - mu_p).pow(2)) / lv_p.exp()
            - 1
        )
        kl_dim_mean = kl_dim_batch.mean(dim=0)
        kl_loss = kl_dim_mean.sum()

        if return_weight_stats:
            if return_kl_dim:
                return recon_loss, kl_loss, kl_dim_mean, weight_stats
            return recon_loss, kl_loss, weight_stats

        if return_kl_dim:
            return recon_loss, kl_loss, kl_dim_mean
        return recon_loss, kl_loss
CVAE = CVAEBarrWeight