import unittest
import warnings

import torch
from torch import nn
from torch.nn import functional as F

from e_3_CVAE_greek_AD import CVAE_greeks_AD


class _Prior(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, eta):
        mu = self.scale * eta[:, :1]
        return mu, torch.full_like(mu, -4.0)


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_bias = nn.Parameter(torch.tensor(0.05))
        self.m_bias = nn.Parameter(torch.tensor(-0.25))

    def forward(self, z, eta):
        x_t = self.x_bias + 0.1 * z + 0.2 * eta[:, 1:2]
        m_t = self.m_bias + 0.05 * z + 0.1 * eta[:, :1]
        mu = torch.cat((x_t, m_t), dim=1)
        return mu, torch.full_like(mu, -5.0)


class _TinyCVAE(nn.Module):
    dim_z = 1
    dim_x = 2
    dim_eta = 3

    def __init__(self, nonfinite=False):
        super().__init__()
        self.prior = _Prior()
        self.decoder = _Decoder()
        self.bn = nn.BatchNorm1d(1)
        self.nonfinite = nonfinite

    def reparameterize(self, mu, log_var, eps=None):
        sample = mu + torch.exp(0.5 * log_var) * eps
        if self.nonfinite and sample.shape[1] == self.dim_x:
            sample = sample.clone()
            sample[0, 0] = float("inf")
        return sample


class _TransformedTinyCVAE(_TinyCVAE):
    def _decode_physical_samples(self, model_samples):
        x_t = model_samples[:, 0]
        u_t = model_samples[:, 1]
        m_t = torch.minimum(torch.zeros_like(x_t), x_t) - F.softplus(u_t)
        return torch.stack((x_t, m_t), dim=1)


class CVAEGreekADTests(unittest.TestCase):
    def setUp(self):
        self.model = _TinyCVAE()
        self.checkpoint = {
            "eta_min": torch.tensor([0.0, 0.1, 0.5]),
            "eta_max": torch.tensor([0.1, 0.5, 2.0]),
        }
        self.eta = [1.0, 0.9, 0.02, 0.2, 1.0]
        self.kwargs = {
            "model_type": "bs",
            "B": 0.8,
            "n_samples": 16,
            "batch_size": 8,
            "seed": 1234,
        }

    def test_hard_indicator_restores_mixed_module_modes(self):
        self.model.train()
        self.model.bn.eval()
        original_modes = [module.training for module in self.model.modules()]
        original_requires_grad = [
            parameter.requires_grad for parameter in self.model.parameters()
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = CVAE_greeks_AD(
                self.model,
                self.checkpoint,
                self.eta,
                **self.kwargs,
            )

        self.assertEqual(result["barrier_mode"], "hard")
        self.assertEqual(result["barrier_estimator"], "frozen_indicator_pathwise")
        self.assertEqual(
            [module.training for module in self.model.modules()], original_modes
        )
        self.assertEqual(
            [parameter.requires_grad for parameter in self.model.parameters()],
            original_requires_grad,
        )
        self.assertEqual(result["seed"], 1234)

    def test_hard_mode_is_explicitly_labeled_and_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = CVAE_greeks_AD(
                self.model,
                self.checkpoint,
                self.eta,
                **self.kwargs,
            )

        self.assertEqual(result["barrier_estimator"], "frozen_indicator_pathwise")
        self.assertIn("not the full barrier Greek", result["warning"])
        self.assertTrue(any(item.category is RuntimeWarning for item in caught))

    def test_arbitrary_seed_is_supported_and_reproducible(self):
        kwargs = {
            "model_type": "bs",
            "B": None,
            "n_samples": 8,
            "batch_size": 8,
        }
        first = CVAE_greeks_AD(
            self.model, self.checkpoint, self.eta, seed=12, **kwargs
        )
        repeated = CVAE_greeks_AD(
            self.model, self.checkpoint, self.eta, seed=12, **kwargs
        )
        different = CVAE_greeks_AD(
            self.model, self.checkpoint, self.eta, seed=13, **kwargs
        )

        self.assertEqual(first["seed"], 12)
        self.assertEqual(
            first["base_prices"]["van_call"],
            repeated["base_prices"]["van_call"],
        )
        self.assertEqual(
            first["base_prices"]["van_put"],
            repeated["base_prices"]["van_put"],
        )
        self.assertNotEqual(
            first["base_prices"]["van_call"],
            different["base_prices"]["van_call"],
        )

    def test_nonfinite_samples_raise_and_still_restore_state(self):
        model = _TinyCVAE(nonfinite=True)
        model.train()
        model.bn.eval()
        original_modes = [module.training for module in model.modules()]

        with self.assertRaisesRegex(RuntimeError, "Non-finite CVAE sample"):
            CVAE_greeks_AD(
                model,
                self.checkpoint,
                self.eta,
                **self.kwargs,
            )

        self.assertEqual([module.training for module in model.modules()], original_modes)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_support_aware_coordinates_are_decoded_inside_ad_graph(self):
        model = _TransformedTinyCVAE()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = CVAE_greeks_AD(
                model,
                self.checkpoint,
                self.eta,
                **self.kwargs,
            )

        maximum = result["sample_diagnostics"]["maximum_by_output"]
        self.assertLess(float(maximum[1]), 0.0)
        self.assertTrue(
            torch.isfinite(torch.tensor(result["greeks"]["delta"]["barr_call"]))
        )


if __name__ == "__main__":
    unittest.main()
