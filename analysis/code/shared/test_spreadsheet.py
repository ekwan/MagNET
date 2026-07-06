import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import spreadsheet as S  # noqa: E402


def test_site_atom_indices_single():
    assert S.site_atom_indices("21") == [20]


def test_site_atom_indices_multiple():
    assert S.site_atom_indices("17,19") == [16, 18]


def test_site_atom_indices_accepts_int_input():
    assert S.site_atom_indices(21) == [20]
