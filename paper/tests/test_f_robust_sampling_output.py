import os
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np


PAPER_DIR = Path(__file__).parents[1]
if str(PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_DIR))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex-tests")

import f_robust


class RobustnessSamplingOutputTests(unittest.TestCase):
    def test_validation_and_multiple_rejection_ratios_are_reported_separately(self):
        formatter = getattr(f_robust, "_format_sampling_rejection_text", None)
        if not callable(formatter):
            self.fail("sampling output must have a dedicated rejection formatter")

        diagnostics = [
            {
                "total_drawn_samples": 110,
                "support_invalid_count": 5,
                "tail_rejected_sample_count": 6,
                "tail_rejected_sample_pct": 100.0 * 6 / 110,
                "regenerated_sample_count": 10,
                "regenerated_sample_pct": 10.0,
            },
            {
                "total_drawn_samples": 120,
                "support_invalid_count": 12,
                "tail_rejected_sample_count": 8,
                "tail_rejected_sample_pct": 100.0 * 8 / 120,
                "regenerated_sample_count": 20,
                "regenerated_sample_pct": 20.0,
            },
        ]

        text = formatter(
            diagnostics,
            validate_samples=True,
            multiple=10,
        )

        self.assertEqual(
            text,
            "validation_rejected=8.5 (7.2727%) | "
            "multiple_rejected=7.0 (6.0606%) | "
            "regenerated=15.0 (15.0000%)",
        )

    def test_x_only_output_does_not_report_unapplied_validation(self):
        class XOnlyCVAE:
            dim_x = 1

            def total_pricing(self, *args, **kwargs):
                return {
                    "van_call": 0.1,
                    "van_put": 0.2,
                    "total_drawn_samples": 11,
                    "used_sample_count": 10,
                    "regenerated_sample_count": 1,
                    "regenerated_sample_pct": 10.0,
                    "support_invalid_count": None,
                    "tail_rejected_sample_count": 1,
                    "tail_rejected_sample_pct": 100.0 / 11.0,
                    "mt_corrected_sample_pct": 0.0,
                }

        checkpoint = {
            "eta_min": np.zeros(3, dtype=np.float32),
            "eta_max": np.ones(3, dtype=np.float32),
        }
        output = StringIO()

        with redirect_stdout(output):
            f_robust._cvae_cases(
                XOnlyCVAE(),
                checkpoint,
                "bs",
                (1.0, 1.0, 0.02, 0.2, 1.0),
                barrier=0.8,
                device="cpu",
                n_samples_list=(10,),
                repeats=1,
                validate_samples=True,
                valid_resample=True,
                mt_corr=False,
                multiple=10,
            )

        self.assertIn("validation_rejected=not checked", output.getvalue())
        self.assertIn("multiple_rejected=1.0 (9.0909%)", output.getvalue())


if __name__ == "__main__":
    unittest.main()
