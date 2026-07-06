"""Error statistics and least-squares fitting shared across the analysis modules.

numpy.linalg.lstsq, not statsmodels: fitting by formula string through statsmodels' patsy parser is
about 1000x slower in a loop that fits many (solvent, solute, formula) combinations, so every fitting
harness here builds a plain design matrix and calls ols_fit directly. Where a module still needs to
report a p-value or similar statsmodels-only diagnostic, keep a separate, explicitly-named
statsmodels code path as a test oracle, not the harness itself.
"""
import numpy as np


def rmse(predicted, actual):
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    return float(np.sqrt(np.mean(np.square(predicted - actual))))


def mae(predicted, actual):
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    return float(np.mean(np.abs(predicted - actual)))


def median_ae(predicted, actual):
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    return float(np.median(np.abs(predicted - actual)))


def summarize_errors(predicted, actual):
    """{"n", "rmse", "mae", "median_ae"} for one (predicted, actual) pair."""
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    return {
        "n": int(predicted.size),
        "rmse": rmse(predicted, actual),
        "mae": mae(predicted, actual),
        "median_ae": median_ae(predicted, actual),
    }


def ols_fit(design, response):
    """Least-squares coefficient vector for `design @ beta ~ response`. Callers build `design`
    (including any intercept column) and drop non-finite rows themselves -- this is a thin,
    single-purpose wrapper, not a formula parser."""
    design = np.asarray(design, dtype=float)
    response = np.asarray(response, dtype=float)
    beta, _residuals, _rank, _sv = np.linalg.lstsq(design, response, rcond=None)
    return beta


def linear_fit_1d(x, y):
    """(intercept, slope) of the least-squares line through (x, y)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    intercept, slope = ols_fit(design, y)
    return float(intercept), float(slope)
