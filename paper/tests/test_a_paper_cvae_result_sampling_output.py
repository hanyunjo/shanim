import json
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import torch


PAPER_DIR = Path(__file__).parents[1]
if str(PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_DIR))

NOTEBOOK_PATH = PAPER_DIR / "a_paper_cvae_result.ipynb"


def _sampling_cell_source():
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    matches = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if "results_by_option" in "".join(cell.get("source", []))
        and "tail_rejected_sample_pct" in "".join(cell.get("source", []))
    ]
    if len(matches) != 1:
        raise AssertionError("CVAE sampling output cell must be unique")
    return matches[0].split("\nfor key in option_keys:", 1)[0]


class FakeCVAE:
    def __init__(self, dim_x):
        self.dim_x = dim_x

    def total_pricing(self, *args, **kwargs):
        n_samples = int(args[5])
        total_drawn = n_samples + 100
        result = {
            "van_call": 0.1,
            "van_put": 0.2,
            "vanilla_sample_count": n_samples,
            "total_drawn_samples": total_drawn,
            "used_sample_count": n_samples,
            "discarded_sample_count": 100,
            "discarded_sample_pct": 100.0 * 100 / total_drawn,
            "regenerated_sample_count": 100,
            "regenerated_sample_pct": 100.0 * 100 / n_samples,
            "support_invalid_count": 40 if self.dim_x >= 2 else None,
            "tail_rejected_sample_count": 60,
            "tail_rejected_sample_pct": 100.0 * 60 / total_drawn,
            "mt_corrected_sample_count": 0,
            "mt_corrected_sample_pct": 0.0,
        }
        if self.dim_x >= 2:
            result.update({"barr_call": 0.05, "barr_put": 0.15})
        return result


def _run_sampling_output(dim_x, validate_samples=True):
    namespace = {
        "np": np,
        "torch": torch,
        "cvae": FakeCVAE(dim_x),
        "eta_t": np.zeros(3),
        "B": 0.8,
        "K": 1.0,
        "r": 0.02,
        "T": 1.0,
        "validate_samples": validate_samples,
        "valid_resample": True,
        "mt_corr": False,
        "sigma": 0.2,
        "multiple": 10,
    }
    output = StringIO()
    with redirect_stdout(output):
        exec(compile(_sampling_cell_source(), str(NOTEBOOK_PATH), "exec"), namespace)
    return output.getvalue()


def _run_sampling_cell(cvae):
    namespace = {
        "np": np,
        "torch": torch,
        "cvae": cvae,
        "eta_t": np.zeros(3),
        "B": 0.8,
        "K": 1.0,
        "r": 0.02,
        "T": 1.0,
        "validate_samples": False,
        "valid_resample": False,
        "mt_corr": False,
        "sigma": 0.2,
        "multiple": 0,
    }
    with redirect_stdout(StringIO()):
        exec(compile(_sampling_cell_source(), str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace


class CVAESamplingOutputCellTests(unittest.TestCase):
    def test_barrier_output_separates_validation_and_multiple_ratios(self):
        output = _run_sampling_output(dim_x=2)

        self.assertIn("validation_rejected=40.0 (3.6364%)", output)
        self.assertIn("multiple_rejected=60.0 (5.4545%)", output)
        self.assertIn("regenerated=100.0 (10.0000%)", output)

    def test_vanilla_only_output_reports_multiple_ratio_without_validation(self):
        output = _run_sampling_output(dim_x=1)

        self.assertIn("validation_rejected=not checked", output)
        self.assertIn("multiple_rejected=60.0 (5.4545%)", output)
        self.assertIn("regenerated=100.0 (10.0000%)", output)

    def test_each_sample_size_reuses_seed_sequence_starting_at_1234(self):
        class SeedRecordingCVAE(FakeCVAE):
            def __init__(self):
                super().__init__(dim_x=1)
                self.seeds = []

            def total_pricing(self, *args, **kwargs):
                self.seeds.append(torch.initial_seed())
                return super().total_pricing(*args, **kwargs)

        cvae = SeedRecordingCVAE()
        _run_sampling_cell(cvae)

        expected_seeds = list(range(1234, 1284)) * 3
        self.assertEqual(cvae.seeds, expected_seeds)


if __name__ == "__main__":
    unittest.main()
