import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from f_NVP import (
    CRealNVP2D,
    _canonical_target_parameterization,
    _checkpoint_target_parameterization,
    diagnose_crealnvp_samples,
    load_crealnvp_checkpoint,
    mt_to_ut_tensor,
    sample_crealnvp,
    sample_valid_crealnvp,
    train_crealnvp_paper2022,
    ut_to_mt_tensor,
)


class NVPTransformTests(unittest.TestCase):
    def test_round_trip_positive_gap(self):
        X_T = torch.tensor([0.2, -0.1, 0.0])
        M_T = torch.tensor([-0.05, -0.3, -0.1])
        physical = torch.stack((X_T, M_T), dim=1)
        reconstructed = ut_to_mt_tensor(mt_to_ut_tensor(physical))
        torch.testing.assert_close(reconstructed, physical, rtol=1e-5, atol=1e-6)

    def test_zero_gap_stays_inside_support(self):
        X_T = torch.tensor([0.2, -0.1, 0.0])
        upper = torch.minimum(torch.zeros_like(X_T), X_T)
        physical = torch.stack((X_T, upper), dim=1)
        reconstructed = ut_to_mt_tensor(mt_to_ut_tensor(physical))
        self.assertTrue(torch.all(reconstructed[:, 1] <= upper))
        torch.testing.assert_close(
            upper - reconstructed[:, 1],
            torch.full_like(upper, 1e-6),
            rtol=1e-4,
            atol=1e-8,
        )

    def test_invalid_raw_training_sample_raises(self):
        physical = torch.tensor([[0.2, 0.01], [-0.1, -0.05]])
        with self.assertRaisesRegex(ValueError, "Raw NVP training data violates"):
            mt_to_ut_tensor(physical)

    def test_random_ut_is_always_physical(self):
        transformed = torch.randn(10000, 2)
        physical = ut_to_mt_tensor(transformed)
        upper = torch.minimum(
            torch.zeros_like(physical[:, 0]), physical[:, 0],
        )
        self.assertTrue(torch.all(physical[:, 1] <= upper))


class NVPParameterizationTests(unittest.TestCase):
    def make_model(self, target_parameterization="ut"):
        return CRealNVP2D(
            dim_eta=3,
            n_coupling=2,
            hidden_dim=8,
            n_hidden=1,
            use_bn=False,
            scale_clip=2.0,
            target_parameterization=target_parameterization,
        )

    def raw_batch(self):
        return torch.tensor([
            [0.2, -0.05],
            [-0.1, -0.3],
            [0.0, -0.1],
            [-0.3, -0.4],
        ])

    def checkpoint(self, model):
        return {
            "model_state": model.state_dict(),
            "dim_x": 2,
            "dim_eta": 3,
            "n_coupling": model.n_coupling,
            "hidden_dim": model.hidden_dim,
            "n_hidden": model.n_hidden,
            "t_negative_slope": model.t_negative_slope,
            "use_bn": model.use_bn,
            "scale_clip": model.scale_clip,
            "eta_min": np.zeros(3, dtype=np.float32),
            "eta_max": np.ones(3, dtype=np.float32),
            **model.checkpoint_metadata(),
        }

    def test_nll_uses_raw_mt_batch_and_backpropagates(self):
        model = self.make_model("ut")
        eta = torch.zeros(4, 3)
        loss = model.nll(self.raw_batch(), eta)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_public_sample_returns_physical_and_optional_ut(self):
        model = self.make_model("ut")
        eta = torch.zeros(3)
        physical = model.sample(eta, 256, antithetic=True)
        transformed = model.sample(eta, 256, return_transformed=True)
        upper = torch.minimum(torch.zeros_like(physical[:, 0]), physical[:, 0])
        self.assertEqual(tuple(physical.shape), (256, 2))
        self.assertEqual(tuple(transformed.shape), (256, 2))
        self.assertTrue(torch.all(physical[:, 1] <= upper))

    def test_decode_physical_preserves_gradient(self):
        model = self.make_model("ut")
        z = torch.randn(16, 2)
        eta = torch.zeros(16, 3, requires_grad=True)
        physical = model.decode_physical(z, eta)
        physical.square().mean().backward()
        self.assertIsNotNone(eta.grad)
        self.assertTrue(torch.isfinite(eta.grad).all())

    def test_ut_validation_does_not_reject_or_resample(self):
        model = self.make_model("ut")
        ckpt = self.checkpoint(model)
        eta_raw = np.array([0.1, 0.2, 0.5], dtype=np.float32)
        initial = sample_crealnvp(model, ckpt, eta_raw, n_samples=128)
        samples, diagnostics = sample_valid_crealnvp(
            model,
            ckpt,
            eta_raw,
            n_samples=128,
            valid_resample=True,
            initial_samples=initial,
        )
        torch.testing.assert_close(samples, initial)
        self.assertEqual(diagnostics["discarded_sample_count"], 0)
        self.assertEqual(diagnostics["regenerated_sample_count"], 0)
        self.assertFalse(diagnostics["valid_resample"])
        self.assertTrue(diagnostics["valid_resample_requested"])

    def test_diagnostics_reports_zero_ut_support_violations(self):
        model = self.make_model("ut")
        ckpt = self.checkpoint(model)
        diagnostics = diagnose_crealnvp_samples(
            model,
            ckpt,
            np.array([0.1, 0.2, 0.5], dtype=np.float32),
            n_samples=128,
            print_output=False,
        )
        self.assertEqual(diagnostics["support_invalid_count"], 0)
        self.assertEqual(diagnostics["target_parameterization"], "ut")

    def test_checkpoint_loader_selects_ut(self):
        model = self.make_model("ut")
        ckpt = self.checkpoint(model)
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "nvp_ut.pt"
            torch.save(ckpt, checkpoint_path)
            loaded, loaded_ckpt = load_crealnvp_checkpoint(
                checkpoint_path, device="cpu",
            )
        self.assertEqual(loaded.target_parameterization, "ut")
        self.assertEqual(
            _checkpoint_target_parameterization(loaded_ckpt), "ut",
        )

    def test_mt_is_default_and_direct_is_legacy_alias(self):
        self.assertEqual(_canonical_target_parameterization(None), "mt")
        self.assertEqual(_canonical_target_parameterization("mt"), "mt")
        self.assertEqual(_canonical_target_parameterization("direct"), "mt")
        signature = inspect.signature(train_crealnvp_paper2022)
        self.assertEqual(
            signature.parameters["target_parameterization"].default, "mt"
        )
        self.assertEqual(signature.parameters["hidden_dim"].default, 100)
        self.assertEqual(signature.parameters["n_hidden"].default, 4)

    def test_custom_hidden_width_and_depth(self):
        model = CRealNVP2D(
            dim_eta=3,
            n_coupling=2,
            hidden_dim=256,
            n_hidden=5,
            use_bn=False,
            target_parameterization="ut",
        )
        self.assertEqual(model.hidden_dim, 256)
        self.assertEqual(model.n_hidden, 5)
        for coupling in model.layers:
            self.assertEqual(len(coupling.s_net.linears), 5)
            self.assertEqual(coupling.s_net.linears[0].out_features, 256)
            self.assertEqual(len(coupling.t_net.linears), 5)
            self.assertEqual(coupling.t_net.linears[0].out_features, 256)


if __name__ == "__main__":
    unittest.main()
