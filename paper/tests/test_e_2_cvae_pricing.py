import unittest
import math

import torch

if not torch.cuda.is_available():
    raise unittest.SkipTest("e_2_CVAE pricing tests require CUDA")

from e_2_CVAE import CVAE


class CVAEPricingOutputTests(unittest.TestCase):
    def make_model(self, dim_x):
        return CVAE(
            dim_x=dim_x,
            dim_eta=3,
            dim_z=1,
            hidden_dims=[8, 8, 8],
            residual_blocks=1,
        ).cuda()

    def test_x_only_total_pricing_returns_only_vanilla(self):
        model = self.make_model(dim_x=1)
        eta = torch.zeros(3, device="cuda")

        prices = model.total_pricing(
            eta, B=0.8, K=1.0, r=0.02, T=1.0, n_samples=64,
            validate_samples=True, valid_resample=True, mt_corr=True,
        )

        self.assertEqual(prices["priced_option_keys"], ["van_call", "van_put"])
        self.assertFalse(prices["has_running_minimum"])
        self.assertEqual(prices["model_output_dim"], 1)
        self.assertNotIn("barr_call", prices)
        self.assertNotIn("barr_put", prices)
        self.assertFalse(prices["validation_applied"])
        self.assertFalse(prices["mt_corr_applied"])
        self.assertEqual(prices["vanilla_sample_count"], 64)

    def test_xt_mt_total_pricing_returns_all_options(self):
        model = self.make_model(dim_x=2)
        eta = torch.zeros(3, device="cuda")

        prices = model.total_pricing(
            eta, B=0.8, K=1.0, r=0.02, T=1.0, n_samples=64,
            validate_samples=False, valid_resample=False, mt_corr=True,
        )

        self.assertEqual(
            prices["priced_option_keys"],
            ["van_call", "van_put", "barr_call", "barr_put"],
        )
        self.assertTrue(prices["has_running_minimum"])
        self.assertEqual(prices["model_output_dim"], 2)
        self.assertIn("barr_call", prices)
        self.assertIn("barr_put", prices)
        self.assertTrue(prices["mt_corr_applied"])

    def test_multiple_rejects_and_refills_independently_of_valid_resample(self):
        model = self.make_model(dim_x=1)
        eta = torch.zeros(3, device="cuda")
        sample_calls = []

        def fake_sample(_eta, n_samples):
            sample_calls.append(n_samples)
            samples = torch.zeros((n_samples, 1), device="cuda")
            samples[0, 0] = math.log(20.0)
            return samples

        model.sample = fake_sample
        prices = model.total_pricing(
            eta, B=0.8, K=1.0, r=0.02, T=1.0, n_samples=64,
            validate_samples=False, valid_resample=False, multiple=10,
        )

        self.assertEqual(sample_calls, [64, 1])
        self.assertTrue(prices["tail_filter_applied"])
        self.assertEqual(prices["tail_rejected_sample_count"], 1)
        self.assertEqual(prices["total_drawn_samples"], 65)

    def test_multiple_rejects_and_refills_when_valid_resample_is_true(self):
        model = self.make_model(dim_x=1)
        eta = torch.zeros(3, device="cuda")
        sample_calls = []

        def fake_sample(_eta, n_samples):
            sample_calls.append(n_samples)
            samples = torch.zeros((n_samples, 1), device="cuda")
            if len(sample_calls) == 1:
                samples[:5, 0] = math.log(20.0)
            return samples

        model.sample = fake_sample
        prices = model.total_pricing(
            eta, B=0.8, K=1.0, r=0.02, T=1.0, n_samples=64,
            validate_samples=False, valid_resample=True, multiple=10,
        )

        self.assertEqual(sample_calls, [64, 5])
        self.assertTrue(prices["tail_filter_applied"])
        self.assertEqual(prices["tail_rejected_sample_count"], 5)
        self.assertEqual(prices["total_drawn_samples"], 69)
        self.assertEqual(prices["vanilla_sample_count"], 64)


if __name__ == "__main__":
    unittest.main()
