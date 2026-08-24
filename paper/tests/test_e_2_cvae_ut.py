import unittest

import torch

if not torch.cuda.is_available():
    raise unittest.SkipTest("e_2_CVAE_UT requires CUDA")

from e_2_CVAE_UT import (
    CVAE,
    CVAEBarrWeight,
    model_to_physical_tensor,
    physical_to_model_tensor,
    running_min_to_u_tensor,
    xu_to_running_min_tensor,
)


class RunningMinimumTransformTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda")

    def test_round_trip_positive_gap(self):
        X_T = torch.tensor([0.2, -0.1, 0.0], device=self.device)
        M_T = torch.tensor([-0.05, -0.3, -0.1], device=self.device)
        physical = torch.stack((X_T, M_T), dim=-1)

        transformed = physical_to_model_tensor(physical)
        reconstructed = model_to_physical_tensor(transformed)

        torch.testing.assert_close(reconstructed, physical, rtol=1e-5, atol=1e-6)

    def test_zero_gap_remains_inside_support(self):
        X_T = torch.tensor([0.2, -0.1, 0.0], device=self.device)
        upper = torch.minimum(torch.zeros_like(X_T), X_T)
        U_T = running_min_to_u_tensor(X_T, upper)
        M_reconstructed = xu_to_running_min_tensor(X_T, U_T)

        self.assertTrue(torch.all(M_reconstructed <= upper))
        torch.testing.assert_close(
            upper - M_reconstructed,
            torch.full_like(upper, 1e-6),
            rtol=1e-4,
            atol=1e-8,
        )

    def test_random_x_u_always_satisfies_support(self):
        X_T = torch.randn(10000, device=self.device)
        U_T = torch.randn(10000, device=self.device)
        M_T = xu_to_running_min_tensor(X_T, U_T)
        upper = torch.minimum(torch.zeros_like(X_T), X_T)
        self.assertTrue(torch.all(M_T <= upper))

    def test_invalid_raw_sample_raises_without_correction(self):
        X_T = torch.tensor([0.2, -0.1], device=self.device)
        M_T = torch.tensor([0.01, -0.05], device=self.device)
        with self.assertRaisesRegex(ValueError, "Raw training data violates"):
            running_min_to_u_tensor(X_T, M_T)


class CVAEUTTests(unittest.TestCase):
    def make_model(self, model_class=CVAE, **kwargs):
        return model_class(
            dim_x=2,
            dim_eta=3,
            dim_z=2,
            hidden_dims=[8, 8, 8],
            residual_blocks=1,
            **kwargs,
        ).cuda()

    def physical_batch(self):
        X_T = torch.tensor([0.2, -0.1, 0.0, -0.3], device="cuda")
        M_T = torch.tensor([-0.05, -0.3, -0.1, -0.4], device="cuda")
        return torch.stack((X_T, M_T), dim=-1)

    def test_forward_accepts_raw_physical_batch(self):
        model = self.make_model()
        eta = torch.zeros(4, 3, device="cuda")
        recon_loss, kl_loss = model(self.physical_batch(), eta)
        self.assertTrue(torch.isfinite(recon_loss))
        self.assertTrue(torch.isfinite(kl_loss))

    def test_sampling_returns_physical_coordinates_by_default(self):
        model = self.make_model()
        eta = torch.zeros(3, device="cuda")
        physical = model.sample(eta, n_samples=256)
        transformed = model.sample(eta, n_samples=256, return_transformed=True)

        self.assertEqual(tuple(physical.shape), (256, 2))
        self.assertEqual(tuple(transformed.shape), (256, 2))
        upper = torch.minimum(torch.zeros_like(physical[:, 0]), physical[:, 0])
        self.assertTrue(torch.all(physical[:, 1] <= upper))

    def test_fixed_noise_sampling_keeps_eta_gradient(self):
        model = self.make_model()
        eta = torch.zeros(3, device="cuda", requires_grad=True)
        eps_z = torch.randn(64, 2, device="cuda")
        eps_x = torch.randn(64, 2, device="cuda")
        samples = model.sample_differentiable(
            eta, n_samples=64, eps_z=eps_z, eps_x=eps_x,
        )
        samples.square().mean().backward()

        self.assertIsNotNone(eta.grad)
        self.assertTrue(torch.isfinite(eta.grad).all())

    def test_new_state_dict_contains_coordinate_metadata(self):
        model = self.make_model()
        metadata = model.state_dict()["_extra_state"]
        self.assertEqual(metadata["model_coordinates"], ["X_T", "U_T"])
        self.assertEqual(metadata["physical_coordinates"], ["X_T", "M_T"])
        self.assertEqual(
            metadata["support_parameterization"],
            "M_T = min(0, X_T) - softplus(U_T)",
        )

    def test_legacy_checkpoint_requires_coordinate_aware_loader(self):
        direct = self.make_model(coordinate_system="direct")
        legacy_state = dict(direct.state_dict())
        legacy_state.pop("_extra_state")

        transformed = self.make_model()
        with self.assertRaisesRegex(ValueError, "probably an old direct"):
            transformed.load_state_dict(legacy_state)

        checkpoint = {
            "model_state": legacy_state,
            "dim_x": 2,
            "dim_eta": 3,
            "dim_z": 2,
            "hidden_dims": [8, 8, 8],
            "residual_blocks": 1,
            "use_bn": False,
        }
        loaded = CVAE.from_checkpoint(checkpoint)
        self.assertEqual(loaded.coordinate_system, "direct")
        self.assertEqual(loaded.model_coordinates, ["X_T", "M_T"])

    def test_weighted_loss_uses_raw_mask_and_transformed_nll(self):
        model = self.make_model(
            CVAEBarrWeight,
            weight_mode="barrier_put",
            weight_alpha=2.0,
            B=0.8,
            K=1.0,
        )
        raw = self.physical_batch()
        transformed = model.prepare_training_target(raw)
        self.assertFalse(torch.allclose(raw[:, 1], transformed[:, 1]))

        eta = torch.zeros(4, 3, device="cuda")
        recon_loss, kl_loss, stats = model(
            raw, eta, return_weight_stats=True,
        )
        self.assertTrue(torch.isfinite(recon_loss))
        self.assertTrue(torch.isfinite(kl_loss))
        self.assertIn("mean", stats)


if __name__ == "__main__":
    unittest.main()
