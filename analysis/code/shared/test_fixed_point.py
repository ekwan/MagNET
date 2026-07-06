import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fixed_point as F  # noqa: E402


def test_decode_int32_with_missing_marker():
    values = np.array([10000, -2147483648, 25000], dtype=np.int32)
    out = F.decode_fixed_point(values)
    assert out[0] == 1.0
    assert np.isnan(out[1])
    assert out[2] == 2.5


def test_decode_int64_also_works():
    # leveling_effect's existing test feeds int64, not just int32 -- the shared decoder must accept
    # any integer dtype, not just int32.
    values = np.array([10000, -2147483648, 25000], dtype=np.int64)
    out = F.decode_fixed_point(values)
    assert out[0] == 1.0
    assert np.isnan(out[1])
    assert out[2] == 2.5


def test_float_input_passes_through():
    values = np.array([1.0, 2.5, float("nan")])
    out = F.decode_fixed_point(values)
    assert out[0] == 1.0
    assert out[1] == 2.5
    assert np.isnan(out[2])


def test_custom_scale_and_marker():
    values = np.array([100, -1], dtype=np.int32)
    out = F.decode_fixed_point(values, scale=100.0, missing_marker=-1)
    assert out[0] == 1.0
    assert np.isnan(out[1])
