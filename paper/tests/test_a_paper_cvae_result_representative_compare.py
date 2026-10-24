import json
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt


NOTEBOOK_PATH = Path(__file__).parents[1] / "a_paper_cvae_result.ipynb"
CELL_ID = "7810a735"


def _representative_cell_source():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    matches = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("id") == CELL_ID
    ]
    if len(matches) != 1:
        raise AssertionError("UI cell 20 must be unique")
    return matches[0]


class FakeCVAE:
    def __init__(self, dim_x):
        self.dim_x = dim_x


def _dump_result(path, n_list, base):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {
        n: [base + 0.001 * repeat_idx for repeat_idx in range(3)]
        for n in n_list
    }
    with path.open("wb") as file:
        pickle.dump(values, file)
    return values


def _dump_fixed_result(path, n_list, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {n: list(values) for n in n_list}
    with path.open("wb") as file:
        pickle.dump(result, file)


def _base_namespace(tmp_path, dim_x, n_list):
    result_base_path = tmp_path / "cvae" / "representative"
    mc_dir = tmp_path / "mc"
    nvp_checkpoint_path = tmp_path / "nvp" / "nvp_bs_test.pt"
    return {
        "cvae": FakeCVAE(dim_x),
        "model_type": "bs",
        "eta_raw": np.array([0.03, np.sqrt(0.05), 1.5], dtype=np.float32),
        "result_base_path": str(result_base_path),
        "multiple_suffix": "",
        "S0": 1.0,
        "B": 0.8,
        "K": 1.0,
        "r": 0.03,
        "T": 1.5,
        "validate_samples": False,
        "valid_resample": False,
        "mt_corr": False,
        "nvp_multiple": 50,
        "representative_n_list": n_list,
        "representative_nvp_repeats": 3,
        "representative_mc_result_dir": mc_dir,
        "representative_nvp_checkpoint_path_overrides": {
            "bs": nvp_checkpoint_path,
        },
    }


def _write_input_results(namespace, option_keys):
    n_list = namespace["representative_n_list"]
    cvae_base = Path(namespace["result_base_path"])
    mc_dir = Path(namespace["representative_mc_result_dir"])
    model_type = namespace["model_type"]
    mc_model_type = {"bs_bbc": "bs", "hes_bbc": "hes"}.get(
        model_type, model_type
    )
    for index, key in enumerate(option_keys):
        _dump_result(Path(f"{cvae_base}_{key}.pkl"), n_list, 0.10 + index * 0.01)
        _dump_result(
            mc_dir / f"mc_{mc_model_type}_{key}_results.pkl",
            n_list,
            0.11 + index * 0.01,
        )


def _write_nvp_cache(namespace, option_keys):
    n_list = namespace["representative_n_list"]
    checkpoint = Path(
        namespace["representative_nvp_checkpoint_path_overrides"]["bs"]
    )
    cache_base = checkpoint.with_suffix("")
    for index, key in enumerate(option_keys):
        _dump_result(Path(f"{cache_base}_{key}.pkl"), n_list, 0.12 + index * 0.01)


class RepresentativeComparisonCellTests(unittest.TestCase):
    def test_small_barrier_put_values_fill_the_vertical_axis(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            namespace = _base_namespace(Path(tmp_dir), dim_x=2, n_list=[10, 20])
            _write_input_results(namespace, ["barr_call", "barr_put"])
            _write_nvp_cache(namespace, ["barr_call", "barr_put"])

            cvae_put_path = Path(f"{namespace['result_base_path']}_barr_put.pkl")
            mc_put_path = (
                Path(namespace["representative_mc_result_dir"])
                / "mc_bs_barr_put_results.pkl"
            )
            checkpoint = Path(
                namespace["representative_nvp_checkpoint_path_overrides"]["bs"]
            )
            nvp_put_path = Path(f"{checkpoint.with_suffix('')}_barr_put.pkl")
            _dump_fixed_result(cvae_put_path, [10, 20], [0.0094, 0.0095, 0.0096])
            _dump_fixed_result(mc_put_path, [10, 20], [0.0091, 0.0092, 0.0093])
            _dump_fixed_result(nvp_put_path, [10, 20], [0.0097, 0.0098, 0.0099])

            with patch.object(plt, "show"):
                exec(
                    compile(_representative_cell_source(), str(NOTEBOOK_PATH), "exec"),
                    namespace,
                )

        figure = namespace["representative_comparison_figure"]
        barrier_put_axis = namespace["representative_comparison_axes"][1]
        try:
            rendered_y = []
            for line in barrier_put_axis.lines:
                rendered_y.extend(np.asarray(line.get_ydata(), dtype=float).ravel())
            for collection in barrier_put_axis.collections:
                for segment in collection.get_segments():
                    rendered_y.extend(np.asarray(segment, dtype=float)[:, 1])
            rendered_y = np.asarray(rendered_y)
            rendered_y = rendered_y[np.isfinite(rendered_y)]
            data_span = float(rendered_y.max() - rendered_y.min())
            axis_min, axis_max = barrier_put_axis.get_ylim()

            self.assertLessEqual(axis_max - axis_min, data_span * 1.11)
        finally:
            plt.close(figure)

    def test_cached_two_dimensional_model_plots_mc_cvae_and_nvp_for_barriers(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            namespace = _base_namespace(Path(tmp_dir), dim_x=2, n_list=[10, 20])
            namespace["model_type"] = "bs_bbc"
            _write_input_results(namespace, ["barr_call", "barr_put"])
            _write_nvp_cache(namespace, ["barr_call", "barr_put"])

            with patch.object(plt, "show"), patch.dict(
                sys.modules, {"f_NVP": None}
            ):
                exec(
                    compile(_representative_cell_source(), str(NOTEBOOK_PATH), "exec"),
                    namespace,
                )

        figure = namespace["representative_comparison_figure"]
        axes = namespace["representative_comparison_axes"]
        try:
            self.assertEqual(
                [axis.get_title() for axis in axes],
                ["Barrier Call", "Barrier Put"],
            )
            legend_labels = {
                text.get_text()
                for axis in axes
                for text in axis.get_legend().get_texts()
            }
            self.assertIn("MC (95% range)", legend_labels)
            self.assertIn("CVAE (95% range)", legend_labels)
            self.assertIn("CRealNVP (95% range)", legend_labels)
        finally:
            plt.close(figure)

    def test_missing_nvp_cache_computes_deterministic_repeats_and_saves_results(self):
        calls = []
        loaded_paths = []
        fake_module = types.ModuleType("f_NVP")

        def load_crealnvp_checkpoint(path, device=None):
            loaded_paths.append(Path(path))
            return object(), {
                "dim_eta": 3,
                "eta_min": np.zeros(3, dtype=np.float32),
                "eta_max": np.ones(3, dtype=np.float32),
            }

        def price_current_options(model, checkpoint, eta_raw, **kwargs):
            calls.append((kwargs["n_samples"], kwargs["seed"], kwargs["multiple"]))
            value = kwargs["n_samples"] / 1000 + kwargs["seed"] / 1_000_000
            return {
                "van_call": value,
                "van_put": value + 0.01,
                "barr_call": value + 0.02,
                "barr_put": value + 0.03,
            }

        fake_module.load_crealnvp_checkpoint = load_crealnvp_checkpoint
        fake_module.price_current_options = price_current_options

        with tempfile.TemporaryDirectory() as tmp_dir:
            namespace = _base_namespace(Path(tmp_dir), dim_x=1, n_list=[2, 4])
            checkpoint_path = Path(
                namespace["representative_nvp_checkpoint_path_overrides"]["bs"]
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.touch()
            _write_input_results(namespace, ["van_call", "van_put"])

            with patch.object(plt, "show"), patch.dict(
                sys.modules, {"f_NVP": fake_module}
            ):
                exec(
                    compile(_representative_cell_source(), str(NOTEBOOK_PATH), "exec"),
                    namespace,
                )

            expected_calls = [
                (n_samples, 1234 + repeat_idx, 50)
                for n_samples in (2, 4)
                for repeat_idx in range(3)
            ]
            self.assertEqual(calls, expected_calls)
            self.assertEqual(loaded_paths, [checkpoint_path])

            cache_base = checkpoint_path.with_suffix("")
            for key in ("van_call", "van_put"):
                cache_path = Path(f"{cache_base}_{key}.pkl")
                self.assertTrue(cache_path.is_file())
                with cache_path.open("rb") as file:
                    cached = pickle.load(file)
                self.assertEqual(sorted(cached), [2, 4])
                self.assertTrue(all(len(cached[n]) == 3 for n in (2, 4)))

        figure = namespace["representative_comparison_figure"]
        axes = namespace["representative_comparison_axes"]
        try:
            self.assertEqual(
                [axis.get_title() for axis in axes],
                ["Vanilla Call", "Vanilla Put"],
            )
        finally:
            plt.close(figure)

    def test_cached_one_dimensional_model_plots_only_vanilla_options(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            namespace = _base_namespace(Path(tmp_dir), dim_x=1, n_list=[10, 20])
            _write_input_results(namespace, ["van_call", "van_put"])
            _write_nvp_cache(namespace, ["van_call", "van_put"])

            with patch.object(plt, "show"):
                exec(
                    compile(_representative_cell_source(), str(NOTEBOOK_PATH), "exec"),
                    namespace,
                )

        figure = namespace["representative_comparison_figure"]
        axes = namespace["representative_comparison_axes"]
        try:
            self.assertEqual(len(axes), 2)
            self.assertEqual(
                [axis.get_title() for axis in axes],
                ["Vanilla Call", "Vanilla Put"],
            )
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
