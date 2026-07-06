"""Decodes the release's fixed-point integer encoding for shieldings and similar fields.

Values are stored as whole numbers equal to the real value times `scale`, with a reserved marker for
values that were never computed. This turns them back into real values (NaN where missing); float
input passes through untouched, so the same decoder reads either the released int32 encoding or an
older plain-float copy.
"""
import numpy as np

FIXED_POINT_SCALE = 1e4
MISSING_MARKER = -2147483648


def decode_fixed_point(values, scale=FIXED_POINT_SCALE, missing_marker=MISSING_MARKER):
    """Convert fixed-point integers back to real-valued floats, mapping the missing marker to NaN."""
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.integer):
        out = values.astype(np.float64) / scale
        out[values == missing_marker] = np.nan
        return out
    return np.asarray(values, dtype=np.float64)
