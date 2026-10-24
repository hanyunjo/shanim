import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "tmp2.ipynb"


def _notebook_code():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return "\n\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def _write_chunk(path, x_t_values):
    x_t_values = np.asarray(x_t_values, dtype=np.float64)
    paths = np.column_stack(
        [np.arange(len(x_t_values)), x_t_values, np.minimum(x_t_values, 0.0)]
    )
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("paths", data=paths)


class TrainingTailAnalysisNotebookTests(unittest.TestCase):
    def test_bs_analysis_reports_exact_cutoff_rates_and_saves_reusable_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chunk_dir = root / "chunks"
            output_dir = root / "output"
            matplotlib_dir = root / "matplotlib"
            chunk_dir.mkdir()
            matplotlib_dir.mkdir()

            _write_chunk(
                chunk_dir / "bs_chunk_000.h5",
                [0.0, np.log(20.0), np.log(40.0)],
            )
            _write_chunk(
                chunk_dir / "bs_chunk_001.h5",
                [np.log(200.0), -1.0, np.nan],
            )

            old_environment = os.environ.copy()
            os.environ.update(
                {
                    "BS_CHUNK_DIR": str(chunk_dir),
                    "HESTON_CHUNK_DIR": str(root / "missing_heston"),
                    "XT_ANALYSIS_MODELS": "bs",
                    "XT_ANALYSIS_CHUNK_COUNT": "2",
                    "XT_ANALYSIS_ROW_BATCH_SIZE": "2",
                    "XT_ANALYSIS_HISTOGRAM_BINS": "16",
                    "XT_ANALYSIS_OUTPUT_DIR": str(output_dir),
                    "MPLBACKEND": "Agg",
                    "MPLCONFIGDIR": str(matplotlib_dir),
                }
            )
            namespace = {}
            output = StringIO()
            try:
                with redirect_stdout(output):
                    exec(
                        compile(_notebook_code(), str(NOTEBOOK_PATH), "exec"),
                        namespace,
                    )
            finally:
                os.environ.clear()
                os.environ.update(old_environment)

            self.assertIn(
                "cutoff_summary", namespace,
                "tmp2.ipynb must expose the cutoff analysis table",
            )
            cutoff_summary = namespace["cutoff_summary"].set_index("multiple")
            self.assertEqual(int(cutoff_summary.loc[20, "exceed_count"]), 3)
            self.assertAlmostEqual(
                float(cutoff_summary.loc[20, "exceed_pct_of_finite"]), 60.0
            )
            self.assertEqual(int(cutoff_summary.loc[40, "exceed_count"]), 2)
            self.assertAlmostEqual(
                float(cutoff_summary.loc[40, "exceed_pct_of_finite"]), 40.0
            )
            self.assertEqual(int(cutoff_summary.loc[150, "exceed_count"]), 1)
            self.assertAlmostEqual(
                float(cutoff_summary.loc[150, "exceed_pct_of_finite"]), 20.0
            )

            overall = namespace["overall_summary"].iloc[0]
            self.assertEqual(int(overall["total_rows"]), 6)
            self.assertEqual(int(overall["finite_rows"]), 5)
            self.assertEqual(int(overall["nonfinite_count"]), 1)
            self.assertEqual(
                int(namespace["histogram_summary"]["count"].sum()), 5
            )
            self.assertEqual(
                namespace["quantile_summary"]["probability"].tolist(),
                [0.999, 0.9999, 0.99999],
            )
            self.assertTrue(
                (namespace["quantile_summary"]["method"]
                 == "histogram_linear_interpolation").all()
            )

            expected_files = {
                "xt_overall_summary.csv",
                "xt_cutoff_summary.csv",
                "xt_chunk_summary.csv",
                "xt_chunk_cutoff_summary.csv",
                "xt_quantile_summary.csv",
                "xt_histogram.csv",
                "xt_distribution.png",
            }
            self.assertEqual(
                {path.name for path in output_dir.iterdir()}, expected_files
            )
            saved_cutoffs = pd.read_csv(output_dir / "xt_cutoff_summary.csv")
            self.assertEqual(
                saved_cutoffs["multiple"].tolist(),
                [20, 30, 40, 50, 60, 70, 80, 90, 100, 150],
            )
            self.assertIn("[BS] 2/2개 1차 통계 완료", output.getvalue())
            self.assertIn("[BS] 2/2개 히스토그램 완료", output.getvalue())

    def test_heston_selection_does_not_access_bs_chunks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chunk_dir = root / "heston_chunks"
            output_dir = root / "output"
            matplotlib_dir = root / "matplotlib"
            chunk_dir.mkdir()
            matplotlib_dir.mkdir()
            _write_chunk(
                chunk_dir / "heston_chunk_000.h5",
                [0.0, np.log(40.0), np.log(200.0)],
            )

            old_environment = os.environ.copy()
            os.environ.update(
                {
                    "BS_CHUNK_DIR": str(root / "missing_bs"),
                    "HESTON_CHUNK_DIR": str(chunk_dir),
                    "XT_ANALYSIS_MODELS": "hes",
                    "XT_ANALYSIS_CHUNK_COUNT": "1",
                    "XT_ANALYSIS_ROW_BATCH_SIZE": "2",
                    "XT_ANALYSIS_HISTOGRAM_BINS": "16",
                    "XT_ANALYSIS_OUTPUT_DIR": str(output_dir),
                    "MPLBACKEND": "Agg",
                    "MPLCONFIGDIR": str(matplotlib_dir),
                }
            )
            namespace = {}
            try:
                with redirect_stdout(StringIO()):
                    exec(
                        compile(_notebook_code(), str(NOTEBOOK_PATH), "exec"),
                        namespace,
                    )
            finally:
                os.environ.clear()
                os.environ.update(old_environment)

            self.assertIn(
                "overall_summary", namespace,
                "tmp2.ipynb must analyze the selected Heston chunks",
            )
            self.assertEqual(namespace["overall_summary"]["model"].tolist(), ["hes"])
            cutoff_summary = namespace["cutoff_summary"].set_index("multiple")
            self.assertEqual(int(cutoff_summary.loc[40, "exceed_count"]), 2)
            self.assertTrue((output_dir / "xt_distribution.png").is_file())


if __name__ == "__main__":
    unittest.main()
