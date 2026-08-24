import unittest

import torch

if not torch.cuda.is_available():
    raise unittest.SkipTest("CVAE training requires CUDA")

from e_1_run_cvae import (
    BaseCVAE,
    CVAEBarrWeight,
    UTCVAE,
    UTCVAEBarrWeight,
    _checkpoint_target_parameterization,
    _make_cvae,
    _normalize_target_parameterization,
)


class CVAEModelSelectionTests(unittest.TestCase):
    def model_kwargs(self):
        return {
            "dim_x": 2,
            "dim_eta": 3,
            "dim_z": 2,
            "hidden_dims": [8, 8, 8],
            "use_bn": False,
            "residual_blocks": 1,
            "weight_mode": "barrier_put",
            "weight_alpha": 2.0,
            "weight_mode2": None,
            "weight_alpha2": 0.0,
            "weight_h": 0.05,
            "weight_normalize": True,
            "S0": 1.0,
            "K": 1.0,
            "B": 0.8,
        }

    def test_mt_and_ut_base_selection(self):
        mt_model = _make_cvae("mt", "base", **self.model_kwargs())
        transformed = _make_cvae("ut", "base", **self.model_kwargs())
        self.assertIs(type(mt_model), BaseCVAE)
        self.assertIs(type(transformed), UTCVAE)

    def test_mt_and_ut_weighted_selection(self):
        mt_model = _make_cvae("mt", "barr_weight", **self.model_kwargs())
        transformed = _make_cvae("ut", "barr_weight", **self.model_kwargs())
        self.assertIs(type(mt_model), CVAEBarrWeight)
        self.assertIs(type(transformed), UTCVAEBarrWeight)

    def test_parameterization_aliases(self):
        self.assertEqual(_normalize_target_parameterization("mt"), "mt")
        self.assertEqual(_normalize_target_parameterization("ut"), "ut")
        self.assertEqual(_normalize_target_parameterization("transformed"), "ut")
        with self.assertRaises(ValueError):
            _normalize_target_parameterization("unknown")

    def test_checkpoint_parameterization_detection(self):
        self.assertEqual(_checkpoint_target_parameterization({}), "mt")
        self.assertEqual(
            _checkpoint_target_parameterization(
                {"model_coordinates": ["X_T", "M_T"]}
            ),
            "mt",
        )
        self.assertEqual(
            _checkpoint_target_parameterization(
                {"model_coordinates": ["X_T", "U_T"]}
            ),
            "ut",
        )
        self.assertEqual(
            _checkpoint_target_parameterization({
                "model_state": {
                    "_extra_state": {"model_coordinates": ["X_T", "U_T"]}
                }
            }),
            "ut",
        )


if __name__ == "__main__":
    unittest.main()
