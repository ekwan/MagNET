import numpy as np
import torch
import torch_geometric
import torch_scatter

# The elements MagNET was trained on (the sigma datasets are GDB molecules plus F, S, and Cl
# substituents): hydrogen, carbon, nitrogen, oxygen, fluorine, sulfur, chlorine. Predictions for any
# other element are not meaningful, so predict_shieldings refuses them.
SUPPORTED_ELEMENTS = frozenset({1, 6, 7, 8, 9, 16, 17})


def yield_data(solute_atomic_numbers, geometries, atomic_numbers, shieldings = None, N_atoms_per_solvent = 3, solvent_distance_threshold = None, atom_type = 'H'):
    data = torch_geometric.data.Data(
        x = torch.as_tensor(atomic_numbers, dtype = torch.long), 
    )
    if shieldings is None:
        data.shieldings = torch.zeros(data.x.shape[0], dtype = torch.float)
    else:
        data.shieldings = torch.as_tensor(shieldings, dtype = torch.float)

    data.pos = torch.as_tensor(geometries, dtype = torch.float)
    
    solute = np.zeros(len(atomic_numbers), dtype = int)
    solute[0:len(solute_atomic_numbers)] = 1
    data.solute = torch.as_tensor(solute, dtype = torch.long)
    
    if atom_type == 'H':
        atom_type_mask = (atomic_numbers == 1) & (solute == 1) # H shieldings
    elif atom_type == 'C':
        atom_type_mask = (atomic_numbers == 6) & (solute == 1) # C shieldings
    data.atom_type_mask = torch.as_tensor(atom_type_mask, dtype = torch.bool)
        
    data.batch = torch.zeros(data.x.shape[0], dtype = torch.long)
    data.natoms = torch.unique_consecutive(data.batch, return_counts = True)[1] 
    
    N_solute_atoms = sum(data.solute).item()
    N_solvent_atoms = data.x.shape[0] - N_solute_atoms
    N_solvent_molecules = N_solvent_atoms // N_atoms_per_solvent
    unique_molecule_batch = np.zeros(N_solute_atoms, dtype = int)
    if N_solvent_molecules > 0:
        unique_molecule_batch = np.concatenate([
            unique_molecule_batch,
            np.concatenate([np.zeros(N_atoms_per_solvent, dtype = int) + 1 + i for i in range(N_solvent_molecules)], axis = 0), 
        ], axis = 0)
    data.unique_molecule_batch = torch.as_tensor(unique_molecule_batch, dtype = torch.long)
    
    if solvent_distance_threshold is not None:
        min_distance_to_solute_peratom = torch.linalg.norm(data.pos - (data.pos[data.solute > 0])[:, None, ...], dim = -1).min(axis = 0).values
        min_distance_to_solute_permolecule = torch_scatter.scatter_min(min_distance_to_solute_peratom, data.unique_molecule_batch)[0]
        remove_atom = np.array([m in torch.where((min_distance_to_solute_permolecule > solvent_distance_threshold))[0] for m in data.unique_molecule_batch])
        
        data.x = data.x[~remove_atom]
        if shieldings is None:
            data.shieldings = data.shieldings[~remove_atom]
        data.pos = data.pos[~remove_atom]
        data.solute = data.solute[~remove_atom]
        data.atom_type_mask = data.atom_type_mask[~remove_atom]
        data.batch = torch.zeros(data.x.shape[0], dtype = torch.long)
        data.natoms = torch.unique_consecutive(data.batch, return_counts = True)[1] 
        N_solute_atoms = sum(data.solute).item()
        N_solvent_atoms = data.x.shape[0] - N_solute_atoms
        N_solvent_molecules = N_solvent_atoms // N_atoms_per_solvent
        unique_molecule_batch = np.zeros(N_solute_atoms, dtype = int)
        if N_solvent_molecules > 0:
            unique_molecule_batch = np.concatenate([
                unique_molecule_batch,
                np.concatenate([np.zeros(N_atoms_per_solvent, dtype = int) + 1 + i for i in range(N_solvent_molecules)], axis = 0), 
            ], axis = 0)
        data.unique_molecule_batch = torch.as_tensor(unique_molecule_batch, dtype = torch.long)
    
    return data


def _predict_once(model_H, model_C, solute_atomic_numbers, geometry, atomic_numbers, N_atoms_per_solvent, solvent_distance_threshold, device):
    # one forward pass each for the 1H and 13C heads, combined into a single per-atom array
    for atom_type in ['H', 'C']:
        data = yield_data(
            solute_atomic_numbers = solute_atomic_numbers,
            geometries = geometry,
            atomic_numbers = atomic_numbers,
            shieldings = None,
            N_atoms_per_solvent = N_atoms_per_solvent if N_atoms_per_solvent is not None else 3,
            solvent_distance_threshold = solvent_distance_threshold,
            atom_type = atom_type,
        )
        data = data.to(device)

        with torch.no_grad():
            if atom_type == 'H':
                y_pred = model_H.forward(data).cpu()
            elif atom_type == 'C':
                y_pred = model_C.forward(data).cpu()
            # the masks live on `device`; move them to CPU to match y_pred (a no-op on CPU, and
            # required on CUDA, where indexing a CPU tensor with a device mask would raise)
            atom_type_mask = data.atom_type_mask.cpu()
            solute = data.solute.cpu()
            y_pred[~atom_type_mask] = 0.
            y_pred = y_pred[solute == 1].numpy()

        # combine H and C shieldings into one array
        if atom_type == 'H':
            y_pred_combined = y_pred
        elif atom_type == 'C':
            y_pred_combined[y_pred != 0.0] = y_pred[y_pred != 0.0]

    return y_pred_combined


def predict_shieldings(model_H, model_C, solute_atomic_numbers, geometry, atomic_numbers = None, N_atoms_per_solvent = None, solvent_distance_threshold = None, device = 'cpu', n_passes = 1, symmetrize = False):
    """Predict 1H/13C shieldings for the solute atoms.

    A single forward pass is NOT deterministic: each edge picks a random local
    reference frame (eqV2/edge_rot_mat.py), so with a finite spherical-harmonic grid
    the output varies by ~0.01 ppm (13C) between passes. Set n_passes (e.g. 20, as in
    the MagNET paper) to average that frame noise away.

    symmetrize=True also averages the prediction over the molecule and its mirror image.
    Isotropic shielding is parity-even (a molecule and its reflection have identical
    shieldings), but the SO(3)-only model does not enforce this and can disagree by
    ~0.3 ppm on 13C for large molecules; symmetrizing removes that spurious error.
    """
    solute_atomic_numbers = np.asarray(solute_atomic_numbers)
    if atomic_numbers is None:
        atomic_numbers = solute_atomic_numbers
    else:
        # coerce so the element comparisons in yield_data (atomic_numbers == 1) stay array-wise;
        # a bare Python list would compare to a scalar False and silently mis-mask
        atomic_numbers = np.asarray(atomic_numbers)

    # MagNET was only ever trained on these elements. On anything else it would emit a confident but
    # meaningless prediction, so refuse it rather than let bad numbers through (validate the full
    # system, which for MagNET-x includes the solvent atoms).
    unsupported = sorted(set(np.unique(atomic_numbers).tolist()) - SUPPORTED_ELEMENTS)
    if unsupported:
        raise ValueError(
            f"MagNET supports only the elements {sorted(SUPPORTED_ELEMENTS)} "
            f"(H, C, N, O, F, S, Cl); the input contains unsupported atomic numbers {unsupported}. "
            f"Exclude molecules with these elements before predicting.")

    geometry = np.asarray(geometry, dtype=float)
    geometries = [geometry]
    if symmetrize:
        reflected = geometry.copy()
        reflected[..., 0] = -reflected[..., 0]   # mirror across the yz-plane (an improper rotation)
        geometries.append(reflected)

    preds = []
    for geom in geometries:
        for _ in range(n_passes):
            preds.append(np.squeeze(_predict_once(model_H, model_C, solute_atomic_numbers, geom,
                                                  atomic_numbers, N_atoms_per_solvent,
                                                  solvent_distance_threshold, device)))
    # atleast_1d keeps a one-atom solute a (1,) array instead of a 0-d scalar after the per-pass squeeze
    return np.atleast_1d(np.mean(preds, axis=0))
