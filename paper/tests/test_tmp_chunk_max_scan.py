import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import h5py
import numpy as np


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "tmp.ipynb"


def _sixth_cell_source():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return "".join(notebook["cells"][5]["source"])


def _write_chunk(path, rows):
    with h5py.File(path, "w") as h5:
        h5.create_dataset("paths", data=np.asarray(rows, dtype=np.float64))


class ChunkMaximumNotebookCellTests(unittest.TestCase):
    def test_sixth_cell_finds_bs_and_heston_global_xt_maxima(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bs_dir = root / "bs"
            hes_dir = root / "hes"
            bs_dir.mkdir()
            hes_dir.mkdir()

            bs_rows = (
                [[0, 1.0, -0.2], [1, 3.0, -0.3]],
                [[2, -1.0, -0.4], [3, 7.0, -0.5]],
                [[4, np.inf, -0.6], [5, np.nan, -0.7]],
            )
            hes_rows = (
                [[10, 4.0, -0.2], [11, 5.0, -0.3]],
                [[12, 9.0, -0.4], [13, 6.0, -0.5]],
                [[14, -2.0, -0.6], [15, 8.0, -0.7]],
            )
            for index, rows in enumerate(bs_rows):
                _write_chunk(bs_dir / f"bs_chunk_{index:03d}.h5", rows)
            for index, rows in enumerate(hes_rows):
                _write_chunk(hes_dir / f"heston_chunk_{index:03d}.h5", rows)

            old_environment = os.environ.copy()
            os.environ.update(
                {
                    "BS_CHUNK_DIR": str(bs_dir),
                    "HESTON_CHUNK_DIR": str(hes_dir),
                    "CVAE_CHUNK_COUNT": "3",
                    "CVAE_CHUNK_WORKERS": "2",
                    "CVAE_CHUNK_ROW_BATCH_SIZE": "2",
                    "CVAE_SCAN_MODELS": "bs,hes",
                }
            )
            namespace = {}
            output = StringIO()
            try:
                with redirect_stdout(output):
                    exec(
                        compile(_sixth_cell_source(), str(NOTEBOOK_PATH), "exec"),
                        namespace,
                    )
            finally:
                os.environ.clear()
                os.environ.update(old_environment)

        self.assertIn(
            "scan_summary",
            namespace,
            "the sixth cell must expose a BS/Heston scan summary",
        )
        summary = namespace["scan_summary"].set_index("model")
        self.assertEqual(float(summary.loc["bs", "max_x_t"]), 7.0)
        self.assertEqual(int(summary.loc["bs", "chunk_index"]), 1)
        self.assertEqual(int(summary.loc["bs", "row_index"]), 1)
        self.assertEqual(int(summary.loc["bs", "eta_index"]), 3)
        self.assertEqual(int(summary.loc["bs", "nonfinite_x_t_count"]), 2)

        self.assertEqual(float(summary.loc["hes", "max_x_t"]), 9.0)
        self.assertEqual(int(summary.loc["hes", "chunk_index"]), 1)
        self.assertEqual(int(summary.loc["hes", "row_index"]), 0)
        self.assertEqual(int(summary.loc["hes", "eta_index"]), 12)
        self.assertEqual(int(summary.loc["hes", "nonfinite_x_t_count"]), 0)

        per_chunk = namespace["chunk_max_results"]
        self.assertEqual(len(per_chunk), 6)
        self.assertEqual(set(per_chunk["model"]), {"bs", "hes"})
        all_nonfinite = per_chunk[
            (per_chunk["model"] == "bs") & (per_chunk["chunk_index"] == 2)
        ].iloc[0]
        self.assertTrue(np.isnan(all_nonfinite["max_x_t"]))
        self.assertTrue(np.isnan(all_nonfinite["row_index"]))
        self.assertTrue(np.isnan(all_nonfinite["eta_index"]))
        self.assertIn("모든 chunk별 최대 X_T", output.getvalue())

    def test_reports_progress_after_every_completed_chunk(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bs_dir = Path(temp_dir) / "bs"
            bs_dir.mkdir()
            for index in range(3):
                _write_chunk(
                    bs_dir / f"bs_chunk_{index:03d}.h5",
                    [[index, float(index), -0.2]],
                )

            old_environment = os.environ.copy()
            os.environ.update(
                {
                    "BS_CHUNK_DIR": str(bs_dir),
                    "CVAE_CHUNK_COUNT": "3",
                    "CVAE_CHUNK_WORKERS": "2",
                    "CVAE_CHUNK_ROW_BATCH_SIZE": "1",
                    "CVAE_SCAN_MODELS": "bs",
                }
            )
            output = StringIO()
            try:
                with redirect_stdout(output):
                    exec(
                        compile(_sixth_cell_source(), str(NOTEBOOK_PATH), "exec"),
                        {},
                    )
            finally:
                os.environ.clear()
                os.environ.update(old_environment)

        progress_output = output.getvalue()
        self.assertIn("[BS] 1/3개 검사 완료", progress_output)
        self.assertIn("[BS] 2/3개 검사 완료", progress_output)
        self.assertIn("[BS] 3/3개 검사 완료", progress_output)

    def test_bs_selection_does_not_access_heston_chunks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bs_dir = root / "bs"
            bs_dir.mkdir()
            _write_chunk(bs_dir / "bs_chunk_000.h5", [[42, 6.5, -0.2]])

            old_environment = os.environ.copy()
            os.environ.update(
                {
                    "BS_CHUNK_DIR": str(bs_dir),
                    "HESTON_CHUNK_DIR": str(root / "missing_heston"),
                    "CVAE_CHUNK_COUNT": "1",
                    "CVAE_CHUNK_WORKERS": "1",
                    "CVAE_CHUNK_ROW_BATCH_SIZE": "1",
                    "CVAE_SCAN_MODELS": "bs",
                }
            )
            namespace = {}
            try:
                with redirect_stdout(StringIO()):
                    try:
                        exec(
                            compile(
                                _sixth_cell_source(), str(NOTEBOOK_PATH), "exec"
                            ),
                            namespace,
                        )
                    except FileNotFoundError as exc:
                        self.fail(f"BS-only scan accessed the Heston path: {exc}")
            finally:
                os.environ.clear()
                os.environ.update(old_environment)

        summary = namespace["scan_summary"]
        self.assertEqual(summary["model"].tolist(), ["bs"])
        self.assertEqual(float(summary.iloc[0]["max_x_t"]), 6.5)
        self.assertEqual(int(summary.iloc[0]["eta_index"]), 42)
        self.assertEqual(set(namespace["chunk_max_results"]["model"]), {"bs"})

    def test_heston_selection_does_not_access_bs_chunks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            hes_dir = root / "hes"
            hes_dir.mkdir()
            _write_chunk(hes_dir / "heston_chunk_000.h5", [[77, 8.5, -0.2]])

            old_environment = os.environ.copy()
            os.environ.update(
                {
                    "BS_CHUNK_DIR": str(root / "missing_bs"),
                    "HESTON_CHUNK_DIR": str(hes_dir),
                    "CVAE_CHUNK_COUNT": "1",
                    "CVAE_CHUNK_WORKERS": "1",
                    "CVAE_CHUNK_ROW_BATCH_SIZE": "1",
                    "CVAE_SCAN_MODELS": "hes",
                }
            )
            namespace = {}
            try:
                with redirect_stdout(StringIO()):
                    try:
                        exec(
                            compile(
                                _sixth_cell_source(), str(NOTEBOOK_PATH), "exec"
                            ),
                            namespace,
                        )
                    except FileNotFoundError as exc:
                        self.fail(f"Heston-only scan accessed the BS path: {exc}")
            finally:
                os.environ.clear()
                os.environ.update(old_environment)

        summary = namespace["scan_summary"]
        self.assertEqual(summary["model"].tolist(), ["hes"])
        self.assertEqual(float(summary.iloc[0]["max_x_t"]), 8.5)
        self.assertEqual(int(summary.iloc[0]["eta_index"]), 77)
        self.assertEqual(set(namespace["chunk_max_results"]["model"]), {"hes"})


if __name__ == "__main__":
    unittest.main()
