import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


NOTEBOOK_PATH = Path(__file__).parents[1] / "a_paper_cvae_result.ipynb"


def _summary_rows(value_column, include_barrier):
    rows = []
    for barr_type, option_type in (
        ("van", "call"),
        ("van", "put"),
        ("barr", "call"),
        ("barr", "put"),
    ):
        value = 0.1 if barr_type == "van" or include_barrier else np.nan
        std = 0.01 if np.isfinite(value) else np.nan
        rows.append({
            "model": "hes",
            "variable": "r",
            "value": 0.05,
            "barr_type": barr_type,
            "option_type": option_type,
            "n_samples": 1_000,
            "reference": 0.1,
            "mc_mean": 0.1,
            "mc_std": 0.01,
            value_column: value,
            value_column.replace("mean", "std"): std,
        })
    return pd.DataFrame(rows)


class VanillaOnlyComparisonCellTest(unittest.TestCase):
    def test_dim_x_one_cvae_compares_only_vanilla_cvae_and_nvp_results(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        source = "".join(notebook["cells"][25]["source"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cvae_path = tmp_path / "cvae_summary.csv"
            nvp_path = tmp_path / "nvp_summary.csv"
            _summary_rows("cvae_mean", include_barrier=False).to_csv(
                cvae_path, index=False
            )
            _summary_rows("nvp_mean", include_barrier=True).to_csv(
                nvp_path, index=False
            )

            namespace = {
                "cvae_summary_path": cvae_path,
                "nvp_summary_path": nvp_path,
            }
            with patch.object(plt, "show"):
                exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)

        figures = namespace["combined_figures"]
        self.assertEqual(len(figures), 1)
        figure, axes = next(iter(figures.values()))
        try:
            self.assertEqual(len(axes), 2)
            self.assertEqual(
                [axis.get_title() for axis in axes],
                ["Vanilla Call", "Vanilla Put"],
            )
            legend_labels = {
                text.get_text()
                for axis in axes
                for text in axis.get_legend().get_texts()
            }
            self.assertIn("CVAE (95% range)", legend_labels)
            self.assertIn("CRealNVP (95% range)", legend_labels)
            self.assertEqual(set(namespace["cvae_summary"]["barr_type"]), {"van"})
            self.assertEqual(set(namespace["nvp_summary"]["barr_type"]), {"van"})
        finally:
            plt.close(figure)

    def test_dim_x_two_cvae_compares_only_barrier_cvae_and_nvp_results(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        source = "".join(notebook["cells"][25]["source"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cvae_path = tmp_path / "cvae_summary.csv"
            nvp_path = tmp_path / "nvp_summary.csv"
            _summary_rows("cvae_mean", include_barrier=True).to_csv(
                cvae_path, index=False
            )
            _summary_rows("nvp_mean", include_barrier=True).to_csv(
                nvp_path, index=False
            )

            namespace = {
                "cvae_summary_path": cvae_path,
                "nvp_summary_path": nvp_path,
            }
            with patch.object(plt, "show"):
                exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)

        figures = namespace["combined_figures"]
        self.assertEqual(len(figures), 1)
        figure, axes = next(iter(figures.values()))
        try:
            self.assertEqual(len(axes), 2)
            self.assertEqual(
                [axis.get_title() for axis in axes],
                ["Barrier Call", "Barrier Put"],
            )
            self.assertEqual(set(namespace["cvae_summary"]["barr_type"]), {"barr"})
            self.assertEqual(set(namespace["nvp_summary"]["barr_type"]), {"barr"})
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
