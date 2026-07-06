---
license: mit
tags:
  - chemistry
  - nmr
  - chemical-shift-prediction
  - molecular-property-prediction
  - equiformer
  
---

# MagNET

MagNET is a family of neural networks for predicting NMR chemical shifts. This repository contains all models, datasets, and code to reproduce the data in the MagNET paper. 

### Contents

| item | contents |
|---|---|
| `magnet/` | the importable model package |
| `data/` | the *sigma* datasets and other auxiliary datasets |
| `analysis/` | scripts for reproducing figures and tables |
| `model_checkpoints/` | weights for the four MagNET models  |

### MagNET Models

| Model | Description | Details |
|---|---|---|
| **MagNET** | foundation model | <ul><li>for near-equilibrium geometries</li><li>trained on PBE0/pcSseg-1/gas shieldings</li></ul> |
| **MagNET-Zero** | high-quality gas-phase solute shieldings | <ul><li>use AIMNet2-optimized geometries</li><li>WP04/pcSseg-2 (<sup>1</sup>H shieldings)</li><li>ωB97X-D/pcSseg-2 (<sup>13</sup>C shieldings)</li></ul> |
| **MagNET-PCM** | implicit solvent corrections | <ul><li>shielding(PCM) - shielding(gas)</li><li>use AIMNet2-optimized geometries</li><li>computed at B3LYP-D3(BJ)/pcSseg-2/chloroform</li></ul> |
| **MagNET-x** | explicit solvent corrections | <ul><li>shielding(solute+solvent) - shielding(solute)</li><li>supports chloroform, benzene, methanol, and water</li><li>use classical MD geometries</li><li>trained at PBE0/pcSseg-1</li></ul> |

All models use Equiformer-V2 and have approximately 10M weights. Separate weights are given for <sup>1</sup>H and <sup>13</sup>C prediction. Input structures can contain H, C, N, O, F, Cl, and S (inference should not be performed on structures with unsupported elements).

### *sigma* Datasets

| Dataset | Details |
|---|---|
| **_sigma_-shake** | <ul><li>4.4M solutes from GDB-13/17 with functional group augmentation</li><li>stationary and quasiclassically perturbed structures at B3LYP-D3(BJ)/6-31G\*</li><li>PBE0/pcSseg-1/gas shieldings</li></ul> |
| **_sigma_-fresh** | <ul><li>10K representative natural product, drug-like, sugar, and peptide solutes dissolved in benzene, chloroform, methanol, and water</li><li>~10 computed poses/solute with many more solvated geometries available</li><li>462K poses have computed PBE0/pcSseg-1 shieldings</li></ul> |
| **_sigma_-pepper** (part 1) | <ul><li>GDB molecules with ≤ 10 heavy atoms</li><li>AIMNet2 stationary structures</li><li>PBE0/pcSseg-1 shieldings</li></ul> |
| **_sigma_-pepper** (part 2) | <ul><li>GDB molecules with ≤ 9 heavy atoms</li><li>AIMNet2 stationary structures</li><li>WP04/pcSseg-2/gas (<sup>1</sup>H shieldings)</li><li>ωB97X-D/pcSseg-2/gas (<sup>13</sup>C shieldings)</li></ul> |
| **_sigma_-concentrate** | <ul><li>50K random structures from *sigma*-shake</li><li>PCM(chloroform) corrections at B3LYP/pcSseg-2</li></ul> |

### Installing MagNET

1. Install MagNET and its dependencies.

   **Option A: Model Only via PyPI (won't work until publication)**

   ```bash
   pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu
   pip install magnet-nmr
   pip install torch_scatter torch_cluster --find-links https://data.pyg.org/whl/torch-2.5.0+cpu.html
   ```
   
   **Option B: Full Installation**
   
   ```bash
   git clone https://github.com/ekwan/MagNET
   cd MagNET
   pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu
   pip install -r magnet/requirements.txt
   pip install --no-deps ./magnet
   ```
   
   **<i>During the review process</i>**, the GitHub repository is private, so use a read-only token for cloning:
   
   `git clone https://<token>@github.com/ekwan/MagNET`

2. **Fetch Model Weights**

   ```bash
   pip install "huggingface_hub[cli]"
   hf download ekwan16/MagNET --local-dir . --include "model_checkpoints/*"
   ```
   
   The model weights will be downloaded into the working directory. No repository cloning is needed, as MagNET will automatically check the current directory for weights. Alternatively, you may pass the `checkpoints_dir` parameter to the inference methods.
   
   **<i>During the review process</i>**, the HuggingFace repository is private, so use a read-only token for downloading:
   
   `hf download ekwan16/MagNET --local-dir . --include "model_checkpoints/*" --token hf_xxx`
 
3. **Check the Installation**
 
   `python -c "import magnet"`
   
   This should work with no errors.
  
### Your First Prediction

Let's predict the <sup>1</sup>H and <sup>13</sup>C shifts of acetone in chloroform. `predict_shifts`
takes an AIMNet2-optimized geometry and runs MagNET-Zero, MagNET-PCM, and the paper's scaling for you.

```python
import numpy as np
import magnet

# acetone, (CH3)2C=O, on an AIMNet2-optimized geometry (Angstrom)
atomic_numbers = np.array([6, 6, 8, 6, 1, 1, 1, 1, 1, 1])   # supported elements: H C N O F Cl S
geometry = np.array([
    [ 1.2913, -0.5947, -0.0016],   # C  methyl
    [ 0.0029,  0.1931, -0.0010],   # C  carbonyl
    [-0.0174,  1.3994, -0.0003],   # O
    [-1.2743, -0.6189,  0.0004],   # C  methyl
    [-1.0822, -1.6899,  0.0014],   # H
    [-1.8597, -0.3513,  0.8791],   # H
    [-1.8605, -0.3530, -0.8783],   # H
    [ 1.3415, -1.2208,  0.8916],   # H
    [ 1.3170, -1.2669, -0.8614],   # H
    [ 2.1415,  0.0788, -0.0298],   # H
])

shifts = magnet.predict_shifts(atomic_numbers, geometry, solvent="chloroform")   # per-atom ppm
```

`shifts` has one value per atom (NaN where the atom is neither <sup>1</sup>H nor <sup>13</sup>C).
Averaging acetone's chemically-equivalent atoms by index:

```python
sites = {
    "carbonyl 13C": [1],
    "methyl 13C":   [0, 3],
    "methyl 1H":    [4, 5, 6, 7, 8, 9],
}
for name, indices in sites.items():
    print(f"{name:<13} {shifts[indices].mean():6.2f} ppm")
```

The predictions are close to experiment:

| site | MagNET | experiment (CDCl<sub>3</sub>) |
|---|---|---|
| carbonyl <sup>13</sup>C | 207.6 | 207.07 |
| methyl <sup>13</sup>C | 30.7 | 30.92 |
| methyl <sup>1</sup>H | 2.20 | 2.17 |

**Notes:**

- **Passes and symmetry.** `n_passes` (default 10) averages out equivariance error over the specified number of forward passes. If `symmetrize` (default True) is set, an additional `n_passes` are also performed on the mirror image of the input geometry (for a total of `2*n_passes`).
- **Geometry.** MagNET-Zero and MagNET-PCM require [AIMNet2](https://github.com/isayevlab/AIMNet2)-optimized geometries. Do not use other geometries.
- **Components.** Pass `return_components=True` to also get the MagNET-Zero shielding, the MagNET-PCM correction, and the scaling coefficients behind each shift, as a `dict`.
- **Other models.** For raw shieldings and solvent corrections, see the [API documentation](#api-documentation).
  
### Getting the datasets

The supporting datasets for this paper are large (42 GB) and are [archived on Hugging Face](https://huggingface.co/ekwan16/MagNET). To fetch everything:

    hf download ekwan16/MagNET --local-dir .

If you only want to download a single dataset, add
`--include "data/<name>/*"` with one of the names below.

| dataset | size | contents |
|---|---|---|
| `sigma-shake` | 2.2 GB | 4.4M GDB solutes, perturbed geometries, PBE0/pcSseg-1 shieldings |
| `sigma-fresh` | 30 GB | 10K solutes in explicit benzene, chloroform, methanol, and water |
| `sigma-pepper` | 501 MB | AIMNet2 GDB structures used to train MagNET-Zero and MagNET-PCM |
| `sigma-concentrate` | 17 MB | PCM corrections on 50K *sigma*-shake structures |
| `delta22` | 2.5 GB | experimental <sup>1</sup>H/<sup>13</sup>C shifts with matched DFT and MagNET predictions |
| `dft8k` | 12 MB | external benchmark of 7111 organics |
| `gdb_qcd` | 291 MB | quasiclassical dynamics of 2461 GDB molecules |
| `magnet_test_predictions` | 822 MB | MagNET predictions and DFT targets |
| `applications` | 5.1 GB | natural product MD geometries and experimental shifts |
| `supertestset_magnet_x` | 1.2 MB | explicit solvent test set for MagNET-x |

### Reproducing Figures and Tables

Each figure and table in the paper is produced by a notebook under `analysis/`.

1. Use the **Option B** checkout.
2. [Download](#getting-the-datasets) the datasets.
3. Install the analysis dependencies: `pip install -r requirements.txt`
4. Run the analysis scripts: `python reproduce.py`

If you want to reproduce specific items: `python reproduce.py fig3 s10`

The notebooks are stored without outputs. Running a notebook outputs its figures into `figures/` and its tables into `documents/`.

### Running the Tests

```bash
pip install -r requirements.txt "pytest>=7"
pytest
```

The tests use small synthetic fixtures. If the large files are present, then more comprehensive tests will run. The `magnet` package tests need the Option B checkout with the inference stack installed (`pip install -r magnet/requirements.txt`); without PyTorch they are skipped.

### API Documentation

You can render the API docs with [pdoc](https://pdoc.dev):

    pip install pdoc
    python build_api_docs.py

Open `api_docs/index.html`. This documents the `magnet` package (`magnet.html` is the public
API: `predict_shifts`, `predict_shieldings`, `implicit_solvent_correction`,
`explicit_solvent_correction`) and every analysis module under `analysis/code`.

### How to Cite

"Chemical Shift Prediction Beyond the Electronic Structure Limit."

Adams, K.; Wagen, C.C.; Wolford, J.; Sak, M.H.; Saurí, J.; Feng, Z.; Bhadauria, A.S.; Bailey, M.A.; Downs, J.S.; Li, S.Z.; Liu, A.I.; Smidt, T.; Paton, R.S.; Liu, R.Y.; Coley, C.W.\*; Kwan, E.E.\*

*submitted*, July 2026.

### License

- The original code, model weights, and datasets in this repository are released under the MIT License (see [`LICENSE`](LICENSE)).
- Third-party literature data redistributed here remains subject to its original publications' terms and should be
cited accordingly: CP3 (`data/cp3/`, Smith and Goodman), NS372
 (`data/ns372/`, Schattenberg and Kaupp), and DFT8K (`data/dft8k/`, Guan and Paton).