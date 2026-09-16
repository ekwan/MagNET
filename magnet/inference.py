import warnings

import numpy as np
import torch
import torch_geometric
import torch_scatter

# The elements MagNET was trained on (the sigma datasets are GDB molecules plus F, S, and Cl
# substituents): hydrogen, carbon, nitrogen, oxygen, fluorine, sulfur, chlorine. Predictions for any
# other element are not meaningful, so predict_shieldings refuses them.
SUPPORTED_ELEMENTS = frozenset({1, 6, 7, 8, 9, 16, 17})

# How many graphs go through one forward pass at most. The passes of one prediction are batched
# together, and a large solvated system at a high n_passes would otherwise have to fit in memory
# all at once. Chunking changes nothing about the answer.
MAX_BATCH_GRAPHS = 64


def resolve_mirror_average(mirror_average, symmetrize, caller):
    """Take whichever of the two names the caller used, and warn if it was the old one.

    `symmetrize` was renamed to `mirror_average`: the option averages a prediction over the
    molecule and its mirror image, where the old name read as though it averaged
    symmetry-equivalent nuclei. The old name still works.

    Args:
        mirror_average: what the caller passed under the new name.
        symmetrize: what the caller passed under the old name, or None if it did not.
        caller: the function name to name in the warning.

    Returns:
        The setting to use.
    """
    if symmetrize is None:
        return mirror_average
    warnings.warn(
        f"{caller}(symmetrize=...) is deprecated; use mirror_average=... instead. The option "
        f"averages the prediction over the molecule and its mirror image, which 'symmetrize' "
        f"read as though it meant averaging symmetry-equivalent nuclei.",
        DeprecationWarning, stacklevel = 3)
    return symmetrize


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


def _predict_batch(model_H, model_C, graphs, device):
    """Run every graph in `graphs` through both heads, one forward pass per head.

    Each entry of `graphs` is one geometry to predict for, as
    `(solute_atomic_numbers, geometry, atomic_numbers, N_atoms_per_solvent,
    solvent_distance_threshold)`. They need not be the same molecule: the graphs are concatenated
    into a single torch_geometric Batch, and the answers are split apart again by which graph each
    atom came from, so molecules of different sizes batch together as readily as repeated passes
    of one.

    The model recomputes `natoms` from `batch.batch`, so the per-graph `batch` and `natoms` that
    yield_data writes are dropped before concatenating and PyG assigns its own.
    `unique_molecule_batch` rides along unread: yield_data consumes it for solvent filtering
    before this point, and no model looks at it, so nothing has to be offset across graphs.

    Returns one (n_solute,) array per entry, in the order given.
    """
    n = len(graphs)
    combined = [None] * n
    for atom_type in ['H', 'C']:
        data_list = []
        for solute_atomic_numbers, geometry, atomic_numbers, per_solvent, threshold in graphs:
            data = yield_data(
                solute_atomic_numbers = solute_atomic_numbers,
                geometries = geometry,
                atomic_numbers = atomic_numbers,
                shieldings = None,
                N_atoms_per_solvent = per_solvent if per_solvent is not None else 3,
                solvent_distance_threshold = threshold,
                atom_type = atom_type,
            )
            # PyG owns `batch` on a Batch, and the model rebuilds `natoms` from it; keeping the
            # single-graph versions here would have them concatenated as ordinary attributes.
            del data.batch, data.natoms
            data_list.append(data)
        batch = torch_geometric.data.Batch.from_data_list(data_list).to(device)

        with torch.no_grad():
            y_pred = (model_H if atom_type == 'H' else model_C).forward(batch).cpu()
        # the masks live on `device`; move them to CPU to match y_pred (a no-op on CPU, and
        # required on CUDA, where indexing a CPU tensor with a device mask would raise)
        atom_type_mask = batch.atom_type_mask.cpu()
        solute = batch.solute.cpu()
        graph_of_atom = batch.batch.cpu()
        y_pred[~atom_type_mask] = 0.
        keep = solute == 1
        y_solute = y_pred[keep]
        graph_of_solute = graph_of_atom[keep]

        for i in range(n):
            y = np.squeeze(y_solute[graph_of_solute == i].numpy())
            # combine H and C shieldings into one array, exactly as the unbatched path does
            if atom_type == 'H':
                combined[i] = y
            elif atom_type == 'C':
                combined[i][y != 0.0] = y[y != 0.0]
    return combined


def _check_elements(atomic_numbers):
    """Refuse a system holding an element MagNET was never trained on.

    MagNET was only ever trained on SUPPORTED_ELEMENTS. On anything else it would emit a confident
    but meaningless prediction, so refuse it rather than let bad numbers through (validate the full
    system, which for MagNET-x includes the solvent atoms).

    Raises:
        ValueError: naming every atomic number that is not supported.
    """
    unsupported = sorted(set(np.unique(atomic_numbers).tolist()) - SUPPORTED_ELEMENTS)
    if unsupported:
        raise ValueError(
            f"MagNET supports only the elements {sorted(SUPPORTED_ELEMENTS)} "
            f"(H, C, N, O, F, S, Cl); the input contains unsupported atomic numbers {unsupported}. "
            f"Exclude molecules with these elements before predicting.")


def _geometries_of(geometry, mirror_average, n_passes):
    """Say which geometries one prediction runs over, the mirror image and the passes included."""
    geometry = np.asarray(geometry, dtype=float)
    geometries = [geometry]
    if mirror_average:
        reflected = geometry.copy()
        reflected[..., 0] = -reflected[..., 0]   # mirror across the yz-plane (an improper rotation)
        geometries.append(reflected)
    return [geom for geom in geometries for _ in range(n_passes)]


def predict_shieldings(model_H, model_C, solute_atomic_numbers, geometry, atomic_numbers = None, N_atoms_per_solvent = None, solvent_distance_threshold = None, device = 'cpu', n_passes = 1, mirror_average = False, symmetrize = None, max_batch_graphs = MAX_BATCH_GRAPHS):
    """Predict 1H/13C shieldings for the solute atoms of one molecule.

    A single forward pass is NOT deterministic: each edge picks a random local
    reference frame (eqV2/edge_rot_mat.py), so with a finite spherical-harmonic grid
    the output varies by ~0.01 ppm (13C) between passes. Set n_passes (e.g. 20, as in
    the MagNET paper) to average that frame noise away.

    mirror_average=True also averages the prediction over the molecule and its mirror image.
    Isotropic shielding is parity-even (a molecule and its reflection have identical
    shieldings), but the SO(3)-only model does not enforce this and can disagree by
    ~0.3 ppm on 13C for large molecules; averaging over the mirror image removes that
    spurious error.

    The passes are independent of one another, so they are run as one batch per head rather than
    one forward pass each. `predict_shieldings_batch` batches whole molecules together as well,
    and is what to call for more than one.

    `symmetrize` is the old name for `mirror_average` and still works, with a DeprecationWarning.
    """
    mirror_average = resolve_mirror_average(mirror_average, symmetrize, "predict_shieldings")
    return predict_shieldings_batch(
        model_H, model_C, [solute_atomic_numbers], [geometry],
        atomic_numbers_list = None if atomic_numbers is None else [atomic_numbers],
        N_atoms_per_solvent = N_atoms_per_solvent,
        solvent_distance_threshold = solvent_distance_threshold, device = device,
        n_passes = n_passes, mirror_average = mirror_average,
        max_batch_graphs = max_batch_graphs)[0]


def predict_shieldings_batch(model_H, model_C, solute_atomic_numbers_list, geometries_list, atomic_numbers_list = None, N_atoms_per_solvent = None, solvent_distance_threshold = None, device = 'cpu', n_passes = 1, mirror_average = False, symmetrize = None, max_batch_graphs = MAX_BATCH_GRAPHS):
    """Predict 1H/13C shieldings for many molecules at once.

    Every molecule's passes, and its mirror image where `mirror_average` is set, go through as one
    batch rather than one forward pass each, and molecules share those batches with one another.
    A 31-atom graph occupies very little of a GPU, so what costs the time is the number of forward
    passes and not the size of any one of them: batching across molecules is what fills the card.

    The molecules need not be the same size or the same shape. Each answer is split out by which
    graph its atoms came from, so a list of molecules comes back as a list of per-atom arrays in
    the order given, exactly as calling `predict_shieldings` on each would have.

    Args:
        model_H, model_C: the 1H and 13C models to run.
        solute_atomic_numbers_list: one array of solute atomic numbers per molecule.
        geometries_list: one (n, 3) coordinate array per molecule, in the same order.
        atomic_numbers_list: the whole system per molecule where it differs from the solute (the
            explicit-solvent path), or None where every system is its own solute.
        N_atoms_per_solvent, solvent_distance_threshold: the explicit-solvent options, applied to
            every molecule.
        device: where to run.
        n_passes: how many forward passes to average per geometry.
        mirror_average: whether to average over the mirror image as well.
        symmetrize: the old name for `mirror_average`, which still works and warns.
        max_batch_graphs: how many graphs go through one forward pass at most. It bounds memory
            and changes no answer.

    Returns:
        One (n_solute,) array per molecule, in the order given.

    Raises:
        ValueError: a molecule holds an element MagNET was not trained on, or the lists differ in
            length.
    """
    mirror_average = resolve_mirror_average(mirror_average, symmetrize, "predict_shieldings_batch")

    if len(solute_atomic_numbers_list) != len(geometries_list):
        raise ValueError(
            f"got {len(solute_atomic_numbers_list)} solutes and {len(geometries_list)} geometries; "
            f"they name the same molecules and must be the same length")
    if atomic_numbers_list is not None and len(atomic_numbers_list) != len(geometries_list):
        raise ValueError(
            f"got {len(atomic_numbers_list)} systems and {len(geometries_list)} geometries; "
            f"they name the same molecules and must be the same length")

    # every graph to run, and which molecule each one belongs to
    graphs, molecule_of_graph = [], []
    for i, (solute, geometry) in enumerate(zip(solute_atomic_numbers_list, geometries_list)):
        solute = np.asarray(solute)
        if atomic_numbers_list is None:
            whole = solute
        else:
            # coerce so the element comparisons in yield_data (atomic_numbers == 1) stay array-wise;
            # a bare Python list would compare to a scalar False and silently mis-mask
            whole = np.asarray(atomic_numbers_list[i])
        _check_elements(whole)
        for geom in _geometries_of(geometry, mirror_average, n_passes):
            graphs.append((solute, geom, whole, N_atoms_per_solvent, solvent_distance_threshold))
            molecule_of_graph.append(i)

    per_graph = []
    for start in range(0, len(graphs), max_batch_graphs):
        per_graph.extend(_predict_batch(model_H, model_C, graphs[start:start + max_batch_graphs],
                                        device))

    # average each molecule's own passes, and nothing else's
    gathered = [[] for _ in geometries_list]
    for prediction, molecule in zip(per_graph, molecule_of_graph):
        gathered[molecule].append(prediction)
    # atleast_1d keeps a one-atom solute a (1,) array instead of a 0-d scalar after the per-pass squeeze
    return [np.atleast_1d(np.mean(passes, axis=0)) for passes in gathered]
