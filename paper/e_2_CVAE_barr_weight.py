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

    def reconstruction_weight(self, 
                              x_raw,
                              S0=1.0,
                              K=1.0,
                              B=0.8,
                              alpha=3.0,
                              h=0.05,
                              normalize=True,
                              weight_mode=None, # validation 할 때 비교하기 위해 존재.
                              ):
        mode = self.weight_mode if weight_mode is None else weight_mode
        if mode not in ("barrier_put", "barrier_near"):
            raise ValueError("weight_mode must be one of None, 'none', 'barrier_put', or 'barrier_near'.")
        if h <= 0:
            raise ValueError("h must be positive.")
        if x_raw.shape[-1] < 2:
                raise ValueError("require x_raw with [X_T, M_T].")
            
        MT = x_raw[:, 1]
        b_log = float(np.log(B / S0))
        near_barrier = torch.exp(-torch.abs(MT - b_log) / h)

        # ITM for put .
        if mode  in ("barrier_put", "barrierput"):
            XT = x_raw[:, 0]
            k_log = float(np.log(K / S0))
            put_side = (XT < k_log).to(dtype=x_raw.dtype)
            
            weight = 1.0 + alpha * put_side * near_barrier

        elif mode in ("barrier_near", "barriernear"):
            weight = 1.0 + alpha * near_barrier

        #
        weight = weight.detach()
        if normalize:
            weight = _normalize_weight(weight)
        return weight

    def forward(self,
                x,
                eta,
                return_kl_dim=False,
                weight_mode=None,
                return_weight_stats=False,
                ):
        mu_q, lv_q = self.recognition(x, eta)
        mu_p, lv_p = self.prior(eta)
        z = self.reparameterize(mu_q, lv_q)
        mu_x, lv_x = self.decoder(z, eta)

        nll_i = gaussian_nll_per_sample(x, mu_x, lv_x)
        weight = self.reconstruction_weight(x_raw=x, weight_mode=weight_mode)
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
CVAE = CVAEBarrWeight