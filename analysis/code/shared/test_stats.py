import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import stats as S  # noqa: E402


def test_rmse_mae_median_ae_hand_computed():
    predicted = [1, 2, 3]
    actual = [1, 2, 4]
    # errors = [0, 0, -1]
    assert S.rmse(predicted, actual) == np.sqrt(1 / 3)
    assert S.mae(predicted, actual) == 1 / 3
    assert S.median_ae(predicted, actual) == 0.0


def test_summarize_errors():
    out = S.summarize_errors([1, 2, 3], [1, 2, 4])
    assert out["n"] == 3
    assert out["rmse"] == pytest.approx(np.sqrt(1 / 3))
    assert out["mae"] == pytest.approx(1 / 3)
    assert out["median_ae"] == 0.0


def test_linear_fit_1d_exact_recovery_no_noise():
    x = np.linspace(0, 10, 20)
    y = 2 * x + 1
    intercept, slope = S.linear_fit_1d(x, y)
    assert abs(intercept - 1) < 1e-8
    assert abs(slope - 2) < 1e-8


def test_ols_fit_matches_polyfit():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = 3 * x - 2 + rng.normal(scale=0.01, size=200)
    design = np.column_stack([np.ones(len(x)), x])
    beta = S.ols_fit(design, y)
    ref_slope, ref_intercept = np.polyfit(x, y, 1)
    assert abs(beta[0] - ref_intercept) < 1e-6
    assert abs(beta[1] - ref_slope) < 1e-6
