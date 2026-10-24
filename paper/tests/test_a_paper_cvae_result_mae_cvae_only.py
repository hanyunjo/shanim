import json
import sys
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


PAPER_DIR = Path(__file__).parents[1]
NOTEBOOK_PATH = PAPER_DIR / "a_paper_cvae_result.ipynb"


def _mae_cell_source():
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    matches = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if "eta_mae_details" in "".join(cell.get("source", []))
        and "load_crealnvp_checkpoint" in "".join(cell.get("source", []))
    ]
    if len(matches) != 1:
        raise AssertionError("CVAE/NVP MAE cell must be unique")
    return matches[0]


def _mae_config_cell_source():
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    matches = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if "mae_target" in "".join(cell.get("source", []))
        and "mae_nvp_path" in "".join(cell.get("source", []))
        and "eta_mae_details" not in "".join(cell.get("source", []))
    ]
    if len(matches) != 1:
        raise AssertionError("MAE configuration cell must be unique")
    return matches[0]


class FakeCVAE:
    def __init__(self, *args, **kwargs):
        self.dim_x = int(kwargs["dim_x"])

    def to(self, device):
        return self

    def load_state_dict(self, state):
        return None

    def eval(self):
        return self

    def total_pricing(self, *args, **kwargs):
        prices = {"van_call": 0.1, "van_put": 0.1}
        if self.dim_x >= 2:
            prices.update({"barr_call": 0.1, "barr_put": 0.1})
        return prices


class CVAEOnlyMAECellTests(unittest.TestCase):
    def test_cvae_only_mode_needs_no_nvp_path_or_nvp_calls(self):
        closed_form = types.ModuleType("b_Closed_form")
        closed_form.BS_vanilla = lambda *args, **kwargs: 0.1
        closed_form.BS_barrier = lambda *args, **kwargs: 0.1

        generate = types.ModuleType("e_0_generate")
        generate.generate_BS_params = lambda n_sets, seed: np.tile(
            np.array([[0.05, 0.2, 1.0]], dtype=np.float32), (n_sets, 1)
        )
        generate.generate_heston_params = lambda n_sets, seed: None

        cvae_module = types.ModuleType("e_2_CVAE")
        cvae_module.CVAE = FakeCVAE

        def fail_nvp_call(*args, **kwargs):
            raise AssertionError("CVAE-only mode must not call NVP code")

        nvp_module = types.ModuleType("f_NVP")
        nvp_module.load_crealnvp_checkpoint = fail_nvp_call
        nvp_module.price_current_options = fail_nvp_call

        display_module = types.ModuleType("IPython.display")
        display_module.display = lambda value: None

        checkpoint = {
            "dim_x": 2,
            "dim_eta": 3,
            "dim_z": 1,
            "hidden_dims": [8, 8],
            "model_state": {},
            "eta_min": np.zeros(3, dtype=np.float32),
            "eta_max": np.ones(3, dtype=np.float32),
        }
        namespace = {
            "mae_target": "cvae",
            "mae_cvae_path": PAPER_DIR / "tests" / "bs_checkpoint.pt",
            "validate_samples": False,
            "mt_corr": False,
            "multiple": 0,
        }
        fake_modules = {
            "b_Closed_form": closed_form,
            "e_0_generate": generate,
            "e_2_CVAE": cvae_module,
            "f_NVP": nvp_module,
            "IPython.display": display_module,
        }

        with patch.dict(sys.modules, fake_modules), patch.object(
            torch, "load", return_value=checkpoint
        ), patch.object(Path, "is_file", return_value=True), redirect_stdout(StringIO()):
            try:
                exec(
                    compile(_mae_cell_source(), str(NOTEBOOK_PATH), "exec"),
                    namespace,
                )
            except NameError as exc:
                self.fail(f"CVAE-only mode must not require NVP globals: {exc}")

        self.assertTrue(namespace["mae_run_cvae"])
        self.assertFalse(namespace["mae_run_nvp"])
        self.assertFalse(any(
            "nvp" in column.lower()
            for column in namespace["eta_mae_details"].columns
        ))
        self.assertEqual(
            list(namespace["option_mae_summary"].columns),
            ["CVAE MAE", "CVAE MSE"],
        )


class NVPOnlyMAECellTests(unittest.TestCase):
    def test_default_nvp_config_needs_no_cvae_save_path(self):
        namespace = {}

        try:
            exec(
                compile(_mae_config_cell_source(), str(NOTEBOOK_PATH), "exec"),
                namespace,
            )
        except NameError as exc:
            self.fail(f"NVP-only config must not require CVAE globals: {exc}")

        self.assertEqual(namespace["mae_target"], "nvp")
        self.assertNotIn("mae_cvae_path", namespace)

    def test_nvp_only_mode_needs_no_cvae_path_or_cvae_calls(self):
        closed_form = types.ModuleType("b_Closed_form")
        closed_form.BS_vanilla = lambda *args, **kwargs: 0.1
        closed_form.BS_barrier = lambda *args, **kwargs: 0.1

        generate = types.ModuleType("e_0_generate")
        generate.generate_BS_params = lambda n_sets, seed: np.tile(
            np.array([[0.05, 0.2, 1.0]], dtype=np.float32), (n_sets, 1)
        )
        generate.generate_heston_params = lambda n_sets, seed: None

        class FailCVAEImport(types.ModuleType):
            def __getattr__(self, name):
                if name == "CVAE":
                    raise AssertionError("NVP-only mode must not import CVAE")
                raise AttributeError(name)

        cvae_module = FailCVAEImport("e_2_CVAE")

        nvp_checkpoint = {
            "dim_eta": 3,
            "eta_min": np.zeros(3, dtype=np.float32),
            "eta_max": np.ones(3, dtype=np.float32),
        }
        nvp_module = types.ModuleType("f_NVP")
        nvp_module.load_crealnvp_checkpoint = (
            lambda path, device: (object(), nvp_checkpoint)
        )
        nvp_module.price_current_options = lambda *args, **kwargs: {
            "van_call": 0.1,
            "van_put": 0.1,
            "barr_call": 0.1,
            "barr_put": 0.1,
        }

        display_module = types.ModuleType("IPython.display")
        display_module.display = lambda value: None

        namespace = {
            "mae_target": "nvp",
            "mae_nvp_path": PAPER_DIR / "tests" / "bs_nvp_checkpoint.pt",
            "validate_samples": False,
            "mt_corr": False,
            "nvp_multiple": 0,
        }
        fake_modules = {
            "b_Closed_form": closed_form,
            "e_0_generate": generate,
            "e_2_CVAE": cvae_module,
            "f_NVP": nvp_module,
            "IPython.display": display_module,
        }

        with patch.dict(sys.modules, fake_modules), patch.object(
            Path, "is_file", return_value=True
        ), redirect_stdout(StringIO()):
            try:
                exec(
                    compile(_mae_cell_source(), str(NOTEBOOK_PATH), "exec"),
                    namespace,
                )
            except NameError as exc:
                self.fail(f"NVP-only mode must not require CVAE globals: {exc}")

        self.assertFalse(namespace["mae_run_cvae"])
        self.assertTrue(namespace["mae_run_nvp"])
        self.assertFalse(any(
            "cvae" in column.lower()
            for column in namespace["eta_mae_details"].columns
        ))
        self.assertEqual(
            list(namespace["option_mae_summary"].columns),
            ["NVP MAE", "NVP MSE"],
        )


if __name__ == "__main__":
    unittest.main()
