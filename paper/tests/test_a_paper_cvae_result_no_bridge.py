import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


NOTEBOOK_PATH = Path(__file__).parents[1] / "a_paper_cvae_result.ipynb"


class _DeterministicRandom:
    @staticmethod
    def standard_normal(size):
        return np.zeros(size, dtype=float)

    @staticmethod
    def uniform(*args, **kwargs):
        raise AssertionError("no-bridge pricing must not sample bridge crossings")


FAKE_CP = SimpleNamespace(
    bool_=np.bool_,
    exp=np.exp,
    full=np.full,
    log=np.log,
    maximum=np.maximum,
    sqrt=np.sqrt,
    zeros=np.zeros,
    random=_DeterministicRandom(),
)


def _load_no_bridge_function():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][16]["source"])
    tree = ast.parse(source)
    function_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "mc_no_bridge_prices_gpu"
    ]
    assert len(function_nodes) == 1, (
        "17번째 셀에 mc_no_bridge_prices_gpu 함수가 정확히 하나 있어야 합니다."
    )
    module = ast.Module(body=function_nodes, type_ignores=[])
    namespace = {"cp": FAKE_CP, "np": np}
    exec(compile(module, str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace["mc_no_bridge_prices_gpu"]


class NoBridgePricingCellTest(unittest.TestCase):
    def assert_price_dict_close(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        for key, expected_value in expected.items():
            self.assertAlmostEqual(actual[key], expected_value)

    def test_bs_no_bridge_uses_only_discrete_monitoring_points(self):
        price = _load_no_bridge_function()

        result = price(
            "bs",
            (1.0, 1.0, 0.0, 0.2, 1.0),
            barrier=0.99,
            n_paths=4,
            dt=0.5,
        )

        expected_put = 1.0 - np.exp(-0.02)
        self.assert_price_dict_close(
            result,
            {
                "van_call": 0.0,
                "van_put": expected_put,
                "barr_call": 0.0,
                "barr_put": 0.0,
            },
        )

    def test_heston_no_bridge_returns_all_four_prices(self):
        price = _load_no_bridge_function()

        result = price(
            "hes",
            (1.0, 1.0, 0.1, 2.0, 0.0, 0.0, -0.7, 0.0, 1.0),
            barrier=0.8,
            n_paths=4,
            dt=0.5,
        )

        expected_call = 1.0 - np.exp(-0.1)
        self.assert_price_dict_close(
            result,
            {
                "van_call": expected_call,
                "van_put": 0.0,
                "barr_call": expected_call,
                "barr_put": 0.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
