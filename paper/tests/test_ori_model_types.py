import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import pandas as pd


PAPER_DIR = Path(__file__).parents[1]
if str(PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_DIR))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex-tests")

with patch.object(torch.cuda, "is_available", return_value=True):
    import e_1_run_cvae
    import e_1_run_cvae_time_check

import e_3_CVAE_greek
import e_3_CVAE_greek_AD
import f_robust


class OriTrainingModelTypeTests(unittest.TestCase):
    def _resolver(self, module):
        resolver = getattr(module, "_resolve_model_config", None)
        if not callable(resolver):
            self.fail(f"{module.__name__} must define _resolve_model_config")
        return resolver

    def test_training_routes_bs_ori_to_matching_folder_and_two_targets(self):
        expected = ("/mnt/d/bs_ori/", "/mnt/d/bs_eta_basic.h5", 2)

        for module in (e_1_run_cvae, e_1_run_cvae_time_check):
            with self.subTest(module=module.__name__):
                self.assertEqual(self._resolver(module)("bs_ori"), expected)

    def test_training_routes_hes_ori_to_matching_folder_and_two_targets(self):
        expected = ("/mnt/d/hes_ori/", "/mnt/d/heston_eta_basic.h5", 2)

        for module in (e_1_run_cvae, e_1_run_cvae_time_check):
            with self.subTest(module=module.__name__):
                self.assertEqual(self._resolver(module)("hes_ori"), expected)

    def test_training_rejects_extra_targets_for_ori_models(self):
        calls = (
            (e_1_run_cvae.train_chunk, "bs_ori", 3),
            (e_1_run_cvae.train_chunk, "hes_ori", 4),
            (e_1_run_cvae_time_check.train_chunk_time_check, "bs_ori", 3),
            (e_1_run_cvae_time_check.train_chunk_time_check, "hes_ori", 4),
        )

        for train, model_type, dim_x in calls:
            with self.subTest(function=train.__name__, model_type=model_type):
                try:
                    train(model_type=model_type, dim_x=dim_x)
                except Exception as exc:
                    self.assertIsInstance(exc, ValueError)
                    self.assertIn("requires dim_x=2", str(exc))
                else:
                    self.fail("ori training must reject targets beyond [X_T, M_T]")


class OriEvaluationModelTypeTests(unittest.TestCase):
    def test_greek_backends_accept_ori_aliases(self):
        resolver = getattr(e_3_CVAE_greek, "_resolve_model_type", None)
        if not callable(resolver):
            self.fail("e_3_CVAE_greek must define _resolve_model_type")

        self.assertEqual(resolver("bs_ori"), "bs")
        self.assertEqual(resolver("hes_ori"), "hes")
        self.assertEqual(e_3_CVAE_greek_AD._resolve_model_type("bs_ori"), "bs")
        self.assertEqual(e_3_CVAE_greek_AD._resolve_model_type("hes_ori"), "hes")

    def test_robustness_backend_accepts_ori_aliases(self):
        try:
            bs_backend = f_robust._model_backend("bs_ori")
            hes_backend = f_robust._model_backend("hes_ori")
        except ValueError as exc:
            self.fail(f"ori model type must be accepted: {exc}")

        self.assertEqual(bs_backend, "bs")
        self.assertEqual(hes_backend, "hes")

    def test_hes_ori_sampling_plot_uses_heston_title(self):
        results = pd.DataFrame([{
            "model": "hes_ori",
            "variable": "r",
            "value": 0.05,
            "barr_type": "van",
            "option_type": "call",
            "eta": (),
            "closed": float("nan"),
            "fdm": 0.1,
            "reference": 0.1,
            "mc_results": {1_000: [0.1]},
            "cvae_results": {1_000: [0.1]},
            "mc_seconds_by_n": {1_000: [0.01]},
            "cvae_seconds_by_n": {1_000: [0.01]},
            "sampling_diagnostics_by_n": {1_000: []},
            "error": None,
        }])
        results.attrs["n_samples_list"] = (1_000,)

        with patch.object(f_robust.plt, "show"):
            try:
                figure, _ = f_robust.plot_sampling_comparison(
                    results, "hes_ori", "r", 0.05
                )
            except UnboundLocalError as exc:
                self.fail(f"hes_ori must select the Heston plot title: {exc}")

        try:
            self.assertEqual(
                figure._suptitle.get_text(),
                "Heston robustness | r=0.05",
            )
        finally:
            f_robust.plt.close(figure)


if __name__ == "__main__":
    unittest.main()
