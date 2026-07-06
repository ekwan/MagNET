"""Helpers for parsing the experimental spreadsheets' conventions."""


def site_atom_indices(atom_numbers):
    """0-based atom indices for a site from a spreadsheet's 1-based comma-separated atom_numbers
    (e.g. "17,19" or "21"), the convention used by both the delta-22 and applications experimental
    spreadsheets."""
    return [int(x) - 1 for x in str(atom_numbers).split(",")]
