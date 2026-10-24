import unittest

import torch
from torch import nn

from e_3_CVAE_greek import CVAE_greeks


class _ValidationCVAE(nn.Module):
    dim_z = 1
    dim_x = 2
    dim_eta = 3

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def prior(self, eta):
        zeros = torch.zeros((eta.shape[0], self.dim_z), device=eta.device)
        return zeros + self.anchor, zeros

    def decoder(self, z, eta):
        zeros = torch.zeros((eta.shape[0], self.dim_x), device=eta.device)
        return zeros + self.anchor, zeros

    @staticmethod
    def reparameterize(mu, log_var, eps=None):
        if mu.shape[1] == 1:
            return eps

        x_t = 0.01 * eps[:, 0]
        m_t = torch.where(
            eps[:, 1] > 0,
            torch.full_like(x_t, 0.1),
            torch.full_like(x_t, -0.2),
        )
        return torch.stack((x_t, m_t), dim=1)


class _NonfiniteOtherColumnCVAE(_ValidationCVAE):
    @staticmethod
    def reparameterize(mu, log_var, eps=None):
        if mu.shape[1] == 1:
            return eps

        x_t = 0.01 * eps[:, 0]
        m_t = torch.full_like(x_t, float("nan"))
        return torch.stack((x_t, m_t), dim=1)


class CVAEGreekValidationTests(unittest.TestCase):
    def setUp(self):
        self.model = _ValidationCVAE()
        self.checkpoint = {
            "eta_min": torch.tensor([0.0, 0.1, 0.5]),
            "eta_max": torch.tensor([0.1, 0.5, 2.0]),
        }
        self.eta = (1.0, 0.9, 0.02, 0.2, 1.0)

    def run_greeks(
        self,
        greeks=("delta",),
        model=None,
        B=0.8,
        seed=1234,
        **kwargs,
    ):
        return CVAE_greeks(
            self.model if model is None else model,
            self.checkpoint,
            self.eta,
            model_type="bs",
            B=B,
            greeks=greeks,
            n_samples=16,
            antithetic=True,
            seed=seed,
            batch_size=16,
            **kwargs,
        )

    def test_valid_resample_rejects_invalid_support_and_refills(self):
        result = self.run_greeks(
            validate_samples=True,
            valid_resample=True,
        )

        diagnostics = result["sampling_diagnostics"]
        self.assertTrue(diagnostics["validation_applied"])
        self.assertTrue(diagnostics["valid_resample"])
        self.assertGreater(diagnostics["support_invalid_count"], 0)
        self.assertGreater(diagnostics["total_drawn_sample_count"], 16)
        draw_counts = diagnostics.get("draw_count_by_round")
        self.assertIsNotNone(
            draw_counts,
            "refill diagnostics must expose the common-noise draw schedule",
        )
        self.assertTrue(all(count == 16 for count in draw_counts))
        self.assertEqual(diagnostics["used_sample_count"], 16)
        self.assertIs(
            diagnostics.get("antithetic_retained"),
            False,
            "row filtering must report that antithetic pairing was not retained",
        )

    def test_valid_resample_false_keeps_original_samples_without_validation(self):
        result = self.run_greeks(
            validate_samples=True,
            valid_resample=False,
        )

        diagnostics = result["sampling_diagnostics"]
        self.assertFalse(diagnostics["validation_applied"])
        self.assertFalse(diagnostics["valid_resample"])
        self.assertIsNone(diagnostics["support_invalid_count"])
        self.assertEqual(diagnostics["total_drawn_sample_count"], 16)
        self.assertEqual(diagnostics["used_sample_count"], 16)

    def test_validate_false_disables_validation_even_if_resample_is_requested(self):
        result = self.run_greeks(
            validate_samples=False,
            valid_resample=True,
        )

        diagnostics = result["sampling_diagnostics"]
        self.assertFalse(diagnostics["validation_applied"])
        self.assertFalse(diagnostics["valid_resample"])
        self.assertIsNone(diagnostics["support_invalid_count"])
        self.assertEqual(diagnostics["total_drawn_sample_count"], 16)

    def test_tail_filter_ignores_nonfinite_non_tail_columns(self):
        try:
            result = self.run_greeks(
                model=_NonfiniteOtherColumnCVAE(),
                B=None,
                validate_samples=False,
                valid_resample=False,
                multiple=10,
            )
        except RuntimeError as exc:
            self.fail(f"tail-only filtering must inspect X_T only: {exc}")

        diagnostics = result["sampling_diagnostics"]
        self.assertFalse(diagnostics["validation_applied"])
        self.assertEqual(diagnostics["tail_rejected_sample_count"], 0)
        self.assertEqual(diagnostics["total_drawn_sample_count"], 16)

    def test_validation_is_applied_to_bumped_option_price_samples(self):
        result = self.run_greeks(
            greeks=("rho",),
            validate_samples=True,
            valid_resample=True,
        )

        diagnostics = result["details"]["rho"].get("sampling_diagnostics")
        self.assertIsNotNone(
            diagnostics,
            "bumped option-price validation diagnostics must be returned",
        )
        self.assertTrue(diagnostics["validation_applied"])
        self.assertGreater(diagnostics["support_invalid_count"], 0)
        self.assertEqual(diagnostics["used_sample_count"], 16)

    def test_seed_none_records_one_effective_seed_for_base_and_bumps(self):
        result = self.run_greeks(
            greeks=("rho",),
            seed=None,
            validate_samples=True,
            valid_resample=True,
        )

        self.assertIsInstance(result["seed"], int)
        base_seed = result["sampling_diagnostics"]["seed"]
        bump_seed = result["details"]["rho"]["sampling_diagnostics"]["seed"]
        self.assertEqual(base_seed, result["seed"])
        self.assertEqual(bump_seed, result["seed"])

    def test_seed_none_does_not_change_global_torch_rng(self):
        torch.manual_seed(77)
        expected_next_draw = torch.rand(4)

        torch.manual_seed(77)
        self.run_greeks(
            seed=None,
            validate_samples=True,
            valid_resample=True,
        )
        actual_next_draw = torch.rand(4)

        torch.testing.assert_close(actual_next_draw, expected_next_draw)


if __name__ == "__main__":
    unittest.main()
