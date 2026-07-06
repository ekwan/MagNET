# ruff: noqa
"""MagNET predicts NMR shieldings using equivariant neural networks.

| function | what you get |
|---|---|
| `predict_shifts` | <sup>1</sup>H and <sup>13</sup>C chemical shifts (ppm) in a specific solvent using MagNET-Zero/MagNET-PCM |
| `predict_shieldings` | <sup>1</sup>H and <sup>13</sup>C shieldings (ppm) in the gas phase |
| `implicit_solvent_correction` | shieldings(PCM=chloroform) - shieldings(gas phase) |
| `explicit_solvent_correction` | shieldings(solute+solvents) - shieldings(solute) |

**Inputs and outputs**

- One molecule: array of `atomic_numbers` (e.g. 6 for carbon) and
  `coordinates` (an N-by-3 array of xyz positions in Angstrom, a numpy array or a nested list). You
  get one numpy array back, one value per atom.
- Many molecules: pass a list of each, and you get a list of arrays back.

**Geometries** MagNET-Zero and MagNET-PCM expect AIMNet2-optimized geometries.

**Supported Solvents**

| solvent | `predict_shifts` | `implicit_solvent_correction` | `explicit_solvent_correction` |
|---|:---:|:---:|:---:|
| tetrahydrofuran | ✓ | | |
| dichloromethane | ✓ | | |
| chloroform | ✓ | ✓ | ✓ |
| toluene | ✓ | | |
| benzene | ✓ | | ✓ |
| chlorobenzene | ✓ | | |
| acetone | ✓ | | |
| dimethylsulfoxide | ✓ | | |
| acetonitrile | ✓ | | |
| trifluoroethanol | ✓ | | |
| methanol | ✓ | | ✓ |
| water | ✓ | | ✓ |

`implicit_solvent_correction` predicts only the chloroform correction; `predict_shifts` linearly scales it to the other solvents.

**Shared options** (all four functions):

- `n_passes` (default `10`): average out equivariance error over `n_passes` forward passes
- `symmetrize` (default `True`): if True, average over `n_passes` on the input geometry and `n_passes` on the mirror image of the input geometry
- `device` (default `None`): where to run, a torch device or a string like `"cpu"` or `"cuda"`; uses GPU if available

"""
__docformat__ = "google"

from .api import (predict_shifts, predict_shieldings, implicit_solvent_correction,
                  explicit_solvent_correction)

__all__ = ["predict_shifts", "predict_shieldings", "implicit_solvent_correction",
           "explicit_solvent_correction"]
