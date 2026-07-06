"""
Data loading functions for delta22 DFT and NN data.

Loads and processes the delta22 HDF5 file (DFT and MagNET shieldings, timings, experimental
shifts) into the flat tables the analysis consumes. Read through analysis/code/delta22.py and
the delta-22 figure notebooks (fig3, fig4, si_figure_s08_panelsBCD, ...).
"""

import csv
import os
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm   # plain tqdm, not tqdm.auto: no ipywidgets/IProgress warning headless or in a notebook


GEOMETRY_TYPES = ["aimnet2", "pbe0_tz"]
IMPLICIT_SOLVATION_MODELS = ["pcm", "smd"]
EXPLICIT_SOLVATION_MODELS = ["desmond", "openMM"]

DEFAULT_PCM_REFERENCE_METHOD = "b3lyp_d3bj"
DEFAULT_PCM_REFERENCE_BASIS = "pcSseg2"

INDEX_COLUMNS = [
    "solute",
    "sap_geometry_type",
    "sap_nmr_method",
    "sap_basis",
    "nucleus",
    "site",
    "solvent"
]

# This dataset stores shieldings and coordinates as whole numbers equal to the real
# value times 10,000, with a reserved marker for values that were never computed.
# These helpers turn them back into real values, leaving blanks where data is missing.
_FIXED_POINT_SCALE = 1e4
_MISSING_MARKER = -2147483648

def _decode_fixed_point(values):
    values = np.asarray(values)
    if values.dtype == np.int32:
        out = values.astype(np.float64) / _FIXED_POINT_SCALE
        out[values == _MISSING_MARKER] = np.nan
        return out
    return np.asarray(values, dtype=np.float64)



def _parse_method_name(s: str) -> tuple[str, str, str, str]:
    """
    Helper for parsing tuples with method details into a method name, basis, 
    solvent model, and solvent name.
    """
    method, basis, solvent_model, solvent_name = next(csv.reader([s]))
    return method, basis, solvent_model, solvent_name


def _sort_and_parse_raw_shieldings(shield_df):
    """
    Helper for sorting the raw shieldings DataFrame to put 'none' solvents first
    and for parsing the shielding lists into true lists from strings.
    """
    def _custom_sort(value):
        """Helper function for sorting solvents with 'none' first."""
        if pd.isna(value) or str(value).lower() == 'none':
            return ('', str(value))
        return (str(value).lower(), str(value))


    def _parse_to_list(x):
        """Parse shielding arrays from string format to float lists."""
        value_list = x.strip("[]").replace('\n','').split(" ")
        value_list = [v for v in value_list if v]
        try:
            return [float(i) for i in value_list]
        except:
            return []

    def _to_float_list(x):
        """Return the shieldings as a plain list of floats. Values loaded from the HDF5 are already
        numeric arrays and are converted directly; the string form is only parsed when a caller hands
        one in (e.g. a live-inference override), since converting an array through str() truncates to
        print precision (~8 digits) instead of keeping full float64 precision."""
        if isinstance(x, str):
            return _parse_to_list(x)
        return np.asarray(x, dtype=float).tolist()

    sort_key = lambda x: x.map(_custom_sort) if x.name == "solvent" else x
    shield_df.sort_values(["solute","solvent"],
                                  key=sort_key,
                                  inplace=True)
    shield_df["shieldings"] = shield_df["shieldings"].apply(_to_float_list)
    return shield_df


def _fix_atom_numbers(atom_numbers, prefix="atom_", coerce_to=str, offset=0):
    """Utility function for converting strings and ints to a common list of strings format."""
    if isinstance(atom_numbers, int):
        atom_numbers = [f"{prefix}{int(atom_numbers)+offset}"]
    else:
        atom_numbers = [f"{prefix}{int(i)+offset}" for i in atom_numbers.split(",")]
    atom_numbers = [coerce_to(i) for i in atom_numbers]
    return atom_numbers


def _aggregate_shieldings_by_site(shieldings_data_df:pd.DataFrame, 
                                  site_atoms_df:pd.DataFrame, 
                                  verbose:bool=True):
    """Aggregate DFT/NN shieldings by site."""

    site_solutes = site_atoms_df.reset_index().solute.unique()

    if verbose:
        print("\033[93m", end="")
        solute_iter = tqdm(site_solutes, 
                           desc="Aggregating solute shieldings by site")
    else:
        solute_iter = site_solutes

    rows = []
    for solute in solute_iter:
        sites_df = site_atoms_df.query(f"solute == '{solute}'")
        data_df = shieldings_data_df.query(f"solute == '{solute}'").copy()

        # Building the frame from the stacked arrays (every row of this solute has the same number of
        # atoms) is far faster than .apply(pd.Series), which allocates a Series per row.
        shieldings_columns = pd.DataFrame(np.stack(data_df.shieldings.to_numpy()),
                                          index=data_df.index)
        shieldings_columns.columns = [f"atom_{i}" for i in range(1, len(shieldings_columns.columns)+1)]
        data_df = data_df.iloc[:,:-1]
        data_df = data_df.join(shieldings_columns)
        data_df.reset_index(inplace=True, drop=True)  # otherwise the index over the for loop won't work

        # Metadata columns (geometry type, NMR method, basis, solvation model, solvent) are the same
        # for every site of this solute; pulled out once as a plain object array so the per-site loop
        # does not rebuild a pandas Series per row.
        meta = data_df.iloc[:, 1:6].to_numpy(dtype=object)
        # the first row is always the stationary (unsolvated) one
        assert data_df.iloc[0, 5] == "none", \
            f"Invalid first row: {data_df.iloc[0, 2]} for solute {solute}"

        for (_, nucl, site), site_row in sites_df.iterrows():
            atom_numbers = _fix_atom_numbers(site_row.iloc[0])
            mean = data_df[atom_numbers].mean(axis=1).to_numpy()
            for i in range(len(meta)):
                geo_type, nmr_method, basis, implicit_solvation_model, solvent = meta[i]
                rows.append([solute, geo_type, nmr_method, basis, implicit_solvation_model,
                             solvent, nucl, site, mean[i]])

    result_df = pd.DataFrame(rows, columns=[
        "solute", 
        "sap_geometry_type", 
        "sap_nmr_method", 
        "sap_basis", 
        "implicit_solvation_model", 
        "solvent", 
        "nucleus", 
        "site", 
        "shielding"
    ])

    if verbose:
        print(f"\033[92mRows of site-aggregated shieldings data: {len(result_df)}\033[0m")

    return result_df

def _create_stationary_df(shieldings_df, solvents):
    """Create a dataframe for the stationary shieldings, tiled across."""
    stationary_df = shieldings_df.query("solvent=='none'")
    rows = []
    for _, row in stationary_df.iterrows():
        solute, geo_type, nmr_method, basis, _, _, nucl, site, stationary = row
        for solvent in solvents:
            row = [solute, geo_type, nmr_method, basis, nucl, site, solvent, stationary]
            rows.append(row)

    stationary_df = pd.DataFrame(rows, columns=[*INDEX_COLUMNS, "stationary"])
    stationary_df.set_index(INDEX_COLUMNS, inplace=True)
    stationary_df = stationary_df.sort_index()
    return stationary_df

def _create_dft_pcm_correction_df(sap_shieldings_df:pd.DataFrame,
                                  stationary_df:pd.DataFrame,
                                  pcm_reference_method:str=DEFAULT_PCM_REFERENCE_METHOD,
                                  pcm_reference_basis:str=DEFAULT_PCM_REFERENCE_BASIS,
                                  verbose:bool=True):
    """
    Create a dataframe for the DFT PCM corrections.
    """
    pcm_dft_shieldings_df = sap_shieldings_df.query("solvent != 'none'")

    pcm_dft_shieldings_df = pd.pivot_table(pcm_dft_shieldings_df,
                                           index=INDEX_COLUMNS,
                                           columns="implicit_solvation_model",
                                           values="shielding")

    pcm_dft_shieldings_df = pcm_dft_shieldings_df.merge(stationary_df,
                                                        on=INDEX_COLUMNS,
                                                        how='left')

    pcm_dft_corrections_df = pcm_dft_shieldings_df.copy()
    pcm_dft_corrections_df["pcm"] = pcm_dft_corrections_df["pcm"] - \
                                    pcm_dft_shieldings_df["stationary"]
    pcm_dft_corrections_df.drop(columns=["stationary"], inplace=True)
    pcm_dft_corrections_df.reset_index(inplace=True)

    # methods with no PCM shielding data of their own (e.g. double hybrids) borrow the reference
    # method's PCM correction below
    missing_methods = set(sap_shieldings_df['sap_nmr_method'].unique()) - \
                      set(pcm_dft_corrections_df['sap_nmr_method'].unique())

    if verbose:
        print("Methods without pcm shielding data:")
        for method in missing_methods:
            print(f" - {method}")

        print("\nThe following reference PCM correction data will be used:")
        print(f"  Reference Method: {pcm_reference_method}")
        print(f"  Reference Basis: {pcm_reference_basis}")

    # Get the data for the reference pcm corrections
    reference_corrections = pcm_dft_corrections_df.query(
        f"sap_nmr_method == '{pcm_reference_method}' and "
        f"sap_basis == '{pcm_reference_basis}'"
    ).copy().drop(columns=["sap_nmr_method","sap_basis"])

    for method in missing_methods:
        method_df = sap_shieldings_df.query(f"sap_nmr_method == '{method}'").copy()
        method_df.drop(
            columns=[
                "implicit_solvation_model",
                "solvent",
                "shielding"
            ], 
            inplace=True
        )

        method_pcm_df = method_df.merge(reference_corrections,
                                        on=[
                                            "solute",
                                            "sap_geometry_type",
                                            "nucleus",
                                            "site"
                                        ],
                                        how='left')

        pcm_dft_corrections_df = pd.concat([pcm_dft_corrections_df, 
                                            method_pcm_df],
                                            ignore_index=True)

    pcm_dft_corrections_df.set_index(INDEX_COLUMNS, inplace=True)

    if verbose:
        print("\nTotal rows in implicit corrections dataframe:", 
            pcm_dft_corrections_df.shape[0])
    
    return pcm_dft_corrections_df

def _create_nn_pcm_correction_df(zap_shieldings_df: pd.DataFrame,
                                 solvents:list,
                                 verbose: bool = True):
    """
    Create a DataFrame for NN PCM corrections.
    """
    pcm_corrections_df = zap_shieldings_df.query("solvent != 'none'").copy()

    # MagNET only computes a PCM correction for chloroform; copy it unchanged to every other
    # solvent (no per-solvent correction is computed here).
    chloroform_df = pcm_corrections_df[pcm_corrections_df["solvent"] == "chloroform"]
    new_rows = []
    for nucleus in chloroform_df["nucleus"].unique():
        nucleus_base = chloroform_df[chloroform_df["nucleus"] == nucleus]
        for solvent in solvents:
            if solvent == "chloroform":
                continue
            new_row = nucleus_base.copy()
            new_row["solvent"] = solvent
            new_rows.append(new_row)

    if new_rows:
        pcm_corrections_df = pd.concat([pcm_corrections_df, *new_rows], ignore_index=True)

    pcm_corrections_df = pd.pivot_table(
        pcm_corrections_df,
        index=INDEX_COLUMNS,
        columns="implicit_solvation_model",
        values="shielding"
    )

    # renamed to match the DFT implicit-corrections dataframe's column name
    pcm_corrections_df.columns = ["pcm"]

    return pcm_corrections_df

def _aggregate_qcd_by_site(qcd_raw_df:pd.DataFrame, 
                           site_atoms_df:pd.DataFrame, 
                           verbose:bool=True):
    """Aggregate QCD corrections by site."""

    if verbose:
        print("\033[93m", end="")
        solute_iter = tqdm(site_atoms_df.iterrows(), 
                           desc="Aggregating solute QCD corrections by site",
                           total=len(site_atoms_df))
    else:
        solute_iter = site_atoms_df.iterrows()

    rows = []
    for (solute, nucl, site), row in solute_iter:
        atom_numbers, *_ = row
        atom_numbers = _fix_atom_numbers(atom_numbers, prefix="", coerce_to=int)
        query = qcd_raw_df.query(f"solute == '{solute}' and atom_number.isin({atom_numbers})")
        correction = query.qcd_correction.mean()
        stderr = query.stderr.mean()
        rows.append([solute, nucl, site, correction, stderr])

    result_df = pd.DataFrame(rows, columns=["solute", "nucleus", "site", "qcd", "qcd_stderr"])
    if verbose:
        print(f"\033[92mRows of site-aggregated QCD corrections: {result_df.shape[0]}\033[0m")
    return result_df


def _tile_qcd_corrections(qcd_df, solvents):
    """Tile QCD corrections across solvents."""
    rows = []
    for _, row in qcd_df.iterrows():
        solute, nucl, site, qcd, _ = row
        for solvent in solvents:
            rows.append([solute, nucl, site, solvent, qcd])
    df = pd.DataFrame(rows, columns=["solute", "nucleus", "site", "solvent", "qcd"])
    df.set_index(["solute", "nucleus", "site", "solvent"], inplace=True)
    return df


def _collate_explicit_corrections(df):
    """
    Collate desmond and openMM corrections for each (solute, nucleus, site, solvent).
    Returns a DataFrame with columns ['desmond', 'openMM'].
    """
    def extract_corrections(group):
        desmond = group.loc[group['explicit_solvation_model'] == 'desmond', 'correction']
        openmm = group.loc[group['explicit_solvation_model'] == 'openMM', 'correction']
        return pd.Series({
            'desmond': desmond.iloc[0] if not desmond.empty else np.nan,
            'openMM': openmm.iloc[0] if not openmm.empty else np.nan
        })

    result_df = df.groupby(["solute", "nucleus", "site", "solvent"]).apply(extract_corrections, include_groups=False)
    result_df = pd.DataFrame(result_df)
    return result_df


def _build_combined_df(
    experimental_df,
    qcd_corrections_df,
    explicit_corrections_df,
    classical_df,
    stationary_df,
    implicit_corrections_df,
    index_columns,
):
    """Build the final combined dataframe with all corrections."""
    combined_corrections_df = pd.concat(
        [experimental_df, qcd_corrections_df, explicit_corrections_df, classical_df], axis=1
    )

    stationary_and_pcm_df = pd.concat([stationary_df, implicit_corrections_df], axis=1)

    # merged (not concatenated) so the non-nmr-specific data gets tiled across every nmr method
    combined_corrections_df.reset_index(inplace=True)
    stationary_and_pcm_df.reset_index(inplace=True)
    combined_df = combined_corrections_df.merge(
        stationary_and_pcm_df,
        on=["solute", "nucleus", "site", "solvent"],
        how='outer'
    )
    combined_df.set_index(index_columns, inplace=True)
    combined_df = combined_df[
        [
            'experimental', 
            'stationary',
            'qcd',
            'pcm',
            'desmond',
            'openMM',
            'desmond_vib',
            'openMM_vib'
        ]
    ]
    return combined_df


# Importable data-loading functions

def load_experiment_data(filepath: str, verbose: bool = True) -> pd.DataFrame:
    """
    Reads the experimental data from an excel file at the specified path.
    """
    if verbose:
        print("\033[93mLoading experimental data...\033[0m", end="", flush=True)
    experimental_df = pd.read_excel(filepath)
    experimental_df["nucleus"] = experimental_df.site.str.split("_").str[-1]
    experimental_df.set_index(["solute","nucleus","site"], inplace=True)

    # stack into column vectors indexed by solute/site/solvent, for vector regressions like
    # y = c0 + c1 * x1 + ...
    experimental_df = experimental_df.drop("atom_numbers", axis=1)
    experimental_df.reset_index(inplace=True)
    experimental_df.set_index(["solute","nucleus","site"], inplace=True)
    experimental_df = pd.DataFrame(experimental_df.stack(future_stack=True))
    experimental_df.columns = ["experimental"]
    experimental_df.index.names = ["solute", "nucleus", "site", "solvent"]
    experimental_df = experimental_df.sort_index()

    if verbose:
        print("\r\033[92mLoaded experimental data.    \033[0m")

    return experimental_df

def load_site_atom_data(filepath: str, verbose: bool = True) -> pd.DataFrame:
    """
    Loads atom number information for each solute/nucleus/site combination from
    the experimental Excel file. 
    
    Returns a DataFrame indexed by solute, nucleus, and site, with a column 
    'atom_numbers'.
    """
    if verbose:
        print("\033[93mLoading site atom data...\033[0m", end="", flush=True)
    experimental_df = pd.read_excel(filepath)
    experimental_df["nucleus"] = experimental_df.site.str.split("_").str[-1]
    result_df = experimental_df[["solute", "nucleus", "site", "atom_numbers"]]
    result_df = result_df.drop_duplicates()
    result_df.set_index(["solute", "nucleus", "site"], inplace=True)
    if verbose:
        print("\r\033[92mLoaded site atom data.       \033[0m")
    return result_df

def load_nmr_method_names(delta22_file_path:str) -> list[str]:
    """
    Returns the names of the nmr methods stored in the specified hdf5 file.
    """
    with h5py.File(delta22_file_path, 'r') as hdf5_file:
        names_data = np.array(hdf5_file["conventional_nmr_method_names"])

    nmr_method_tuples = [name.decode("ascii") for name in names_data]
    parsed_methods = [_parse_method_name(t) for t in nmr_method_tuples]
    columns = ["sap_nmr_method", "sap_basis", "solvent_model", "solvent_name"]
    return pd.DataFrame(parsed_methods, columns=columns)

def load_solvents_from_methods(delta22_file_path:str) -> list[str]:
    """
    Returns the list of solvents used in the nmr methods stored in the specified hdf5 file.
    """
    methods_df = load_nmr_method_names(delta22_file_path)
    solvents = methods_df['solvent_name'].unique().tolist()
    return [s for s in solvents if s.lower() != 'none']


# Geometry and topology accessors.
#
# These return the raw stored arrays (atom coordinates and solvent atom orderings) that the
# shielding loaders above do not expose. Coordinates are stored in the same fixed-point form as
# the shieldings (whole numbers equal to the real value in angstroms times 10,000) and are decoded
# back to angstroms here; the solvent atom orderings are plain atomic numbers and need no decoding.

def load_solutes(hdf5_filepath: str) -> list[str]:
    """
    Returns the names of the solutes stored in the file, in file order.
    """
    with h5py.File(hdf5_filepath, "r") as hdf5_file:
        return list(hdf5_file["solutes"].keys())


def load_solvent_atomic_numbers(hdf5_filepath: str,
                                solvent: str,
                                explicit_solvation_model: str = "openMM") -> np.ndarray:
    """
    Returns the atomic numbers of one solvent molecule, in the atom order used by the explicit
    solvent geometries. explicit_solvation_model is "openMM" (default) or "desmond"; the two
    orderings are stored separately. This is the reference topology for reading the solvent atoms
    in the solvated geometries from load_explicit_ensemble_geometries.
    """
    key = f"{explicit_solvation_model}_atomic_numbers"
    with h5py.File(hdf5_filepath, "r") as hdf5_file:
        return np.array(hdf5_file["solvents"][solvent][key])


def load_stationary_geometries(hdf5_filepath: str, solute: str) -> dict:
    """
    Returns the two resting geometries the conventional (stationary and PCM) shieldings were
    computed on, as a dict {geometry_type: coordinates (n_atoms, 3) in angstroms}. The geometry
    types are "aimnet2" and "pbe0_tz", matching GEOMETRY_TYPES.
    """
    with h5py.File(hdf5_filepath, "r") as hdf5_file:
        coords = _decode_fixed_point(hdf5_file["solutes"][solute]["stationary_and_pcm"]["geometries"][()])
    return {geometry_type: coords[i] for i, geometry_type in enumerate(GEOMETRY_TYPES)}


def load_qcd_geometries(hdf5_filepath: str, solute: str) -> dict:
    """
    Returns the gas-phase quasiclassical-dynamics geometries the QCD corrections were computed on,
    as a dict with "unperturbed" (n_atoms, 3) = the resting geometry and "perturbed"
    (n_trajectories, n_frames, n_atoms, 3) = the jiggled snapshots. Coordinates are in angstroms.
    The perturbed frame axes align with the perturbed shieldings used by load_average_qcd_corrections.
    """
    with h5py.File(hdf5_filepath, "r") as hdf5_file:
        qcd = hdf5_file["solutes"][solute]["qcd"]["gas_phase"]
        return {
            "unperturbed": _decode_fixed_point(qcd["unperturbed_geometry"][()]),
            "perturbed": _decode_fixed_point(qcd["perturbed_geometries"][()]),
        }


def load_explicit_ensemble_geometries(hdf5_filepath: str,
                                      solute: str,
                                      solvent: str,
                                      explicit_solvation_model: str = "openMM") -> np.ndarray:
    """
    Returns the explicit-solvent snapshot coordinates (n_frames, n_cluster_atoms, 3) in angstroms
    for one solute/solvent under one engine ("openMM" or "desmond"). The solute atoms come first
    (load_solutes order), then whole solvent molecules in the load_solvent_atomic_numbers order.
    The frame axis aligns with the perturbed shieldings used by the explicit-correction loaders.
    """
    with h5py.File(hdf5_filepath, "r") as hdf5_file:
        node = hdf5_file["solutes"][solute][explicit_solvation_model][solvent]
        return _decode_fixed_point(node["perturbed_ensemble_geometries"][()])


def load_explicit_solute_geometry(hdf5_filepath: str,
                                  solute: str,
                                  explicit_solvation_model: str = "openMM") -> np.ndarray:
    """
    Returns the isolated resting solute geometry (n_atoms, 3) in angstroms used as the gas-phase
    reference for the explicit-solvent corrections under one engine ("openMM" or "desmond").
    """
    with h5py.File(hdf5_filepath, "r") as hdf5_file:
        node = hdf5_file["solutes"][solute][explicit_solvation_model]["gas_phase"]
        return _decode_fixed_point(node["unperturbed_solute_geometry"][()])


def load_perturbed_shieldings(hdf5_filepath, solute, solvent,
                              explicit_solvation_model="openMM", shield_type="dft"):
    """The raw per-frame explicit-solvent shieldings (n_frames, n_atoms, 2) for one solute/solvent
    under one engine ("openMM" or "desmond"). shield_type is "dft" or "nn". The last axis is
    [isolated, solvated], so the per-frame solvent correction is solvated minus isolated. This is the
    frame-resolved data behind the convergence and autocorrelation panels of Figure S8; the
    site-aggregated correction over all frames comes from load_site_aggregated_explicit_shieldings.
    Frames that were not computed read as NaN."""
    key = f"perturbed_{shield_type}_shieldings"
    with h5py.File(hdf5_filepath, "r") as hdf5_file:
        node = hdf5_file["solutes"][solute][explicit_solvation_model][solvent]
        return _decode_fixed_point(node[key][()])


def load_stationary_and_pcm_dft_shieldings(hdf5_filepath:str,
                                           verbose:bool=True):
    """Load raw stationary and PCM DFT shieldings, one row per (solute, geometry, nmr method)."""
    if verbose:
        print("\033[93mLoading stationary and PCM DFT shieldings...\033[0m", end="", flush=True)
    with h5py.File(hdf5_filepath, 'r') as hdf5_file:
        nmr_method_names = np.array(hdf5_file["conventional_nmr_method_names"])
        nmr_method_names = [i.decode("utf-8") for i in nmr_method_names]

        dft_rows = []
        nan_dft_rows = 0
        for solute in hdf5_file["solutes"]:
            solute_group = hdf5_file["solutes"][solute]["stationary_and_pcm"]

            geometry_optimization_timings = np.array(solute_group["geometry_optimization_timings"])
            dft_shieldings = _decode_fixed_point(solute_group["conventional_shieldings"][()])
            dft_timings = np.array(solute_group["conventional_nmr_timings"])

            for nmr_method_index, nmr_method_name in enumerate(nmr_method_names):
                nmr_method, basis, solvent_model, solvent_name = _parse_method_name(nmr_method_name)

                for geometry_index, geometry_type in enumerate(GEOMETRY_TYPES):
                    method_geometry_time = geometry_optimization_timings[geometry_index]
                    method_nmr_time = dft_timings[nmr_method_index, geometry_index]
                    method_shieldings = dft_shieldings[nmr_method_index, geometry_index]
                    if np.isnan(method_shieldings).any():
                        nan_dft_rows += 1
                    dft_row = [
                        solute,
                        geometry_type,
                        nmr_method,
                        basis,
                        solvent_model,
                        solvent_name,
                        method_shieldings,
                        method_geometry_time,
                        method_nmr_time
                    ]
                    dft_rows.append(dft_row)

        if nan_dft_rows > 0 and verbose:
            print(f"Found {nan_dft_rows} total DFT rows with NaNs!")

    columns = [
        "solute",
        "sap_geometry_type",
        "sap_nmr_method",
        "sap_basis",
        "solvent_model",
        "solvent",
        "shieldings",
        "geometry_time",
        "nmr_time"
    ]
    dft_shieldings_df = pd.DataFrame(dft_rows, columns=columns)
    dft_shieldings_df = _sort_and_parse_raw_shieldings(dft_shieldings_df)
    if verbose:
        print("\r\033[92mLoaded stationary and PCM DFT shieldings.    \033[0m")
    return dft_shieldings_df

def load_stationary_and_pcm_nn_shieldings(hdf5_filepath:str,
                                          verbose:bool=True):
    """Load raw stationary and PCM NN (MagNET) shieldings, one row per (solute, geometry, nmr method)."""
    with h5py.File(hdf5_filepath, 'r') as hdf5_file:
        nn_rows = []
        nan_nn_rows = 0
        for solute in hdf5_file["solutes"]:
            atomic_numbers = np.array(hdf5_file["solutes"][solute]["atomic_numbers"])
            ch_mask = np.where((atomic_numbers == 1) | (atomic_numbers == 6))[0]

            solute_group = hdf5_file["solutes"][solute]["stationary_and_pcm"]
            geometry_optimization_timings = np.array(solute_group["geometry_optimization_timings"])
            nn_gas_shieldings = _decode_fixed_point(solute_group["nn_gas_shieldings"][()])
            nn_pcm_corrections = _decode_fixed_point(solute_group["nn_pcm_corrections"][()])
            nn_nmr_timings = np.array(solute_group["nn_nmr_timings"])

            if np.isnan(nn_gas_shieldings[ch_mask]).any():
                nan_nn_rows += 1
            if np.isnan(nn_pcm_corrections[ch_mask]).any():
                nan_nn_rows += 1
            
            nn_row_gas = [
                solute, 
                "aimnet2", 
                "MagNET",
                "N/A", 
                "gas", 
                "none", 
                nn_gas_shieldings, 
                geometry_optimization_timings[0], # AimNet2 timing
                nn_nmr_timings[0]                 # MagNET-Zero timing 
            ]
            nn_row_pcm = [
                solute, 
                "aimnet2", 
                "MagNET", 
                "N/A", 
                "pcm_correction", 
                "chloroform", 
                nn_pcm_corrections, 
                geometry_optimization_timings[0], # AimNet2 timing
                nn_nmr_timings[1]                 # MagNET-PCM timing 
            ]
            nn_rows.append(nn_row_gas)
            nn_rows.append(nn_row_pcm)

        if nan_nn_rows > 0 and verbose:
            print(f"Found {nan_nn_rows} total NN rows with NaNs!")

    columns = [
        "solute",
        "sap_geometry_type",
        "sap_nmr_method",
        "sap_basis",
        "solvent_model",
        "solvent",
        "shieldings",
        "geometry_time",
        "nmr_time"
    ]
    nn_shieldings_df = pd.DataFrame(nn_rows, columns=columns)
    nn_shieldings_df = _sort_and_parse_raw_shieldings(nn_shieldings_df)
    return nn_shieldings_df

def load_average_qcd_corrections(hdf5_filepath, 
                                 shield_type="dft", 
                                 verbose:bool=True):
    """Loads and computes the average QCD corrections."""
    if verbose:
        print("\033[93mLoading QCD data...\033[0m", end="", flush=True)
    rows = []
    with h5py.File(hdf5_filepath, "r") as hdf5_file:
        solutes = hdf5_file["solutes"].keys()
        for solute in solutes:
            qcd_group = hdf5_file["solutes"][solute]["qcd"]["gas_phase"]
            if shield_type == "dft":
                unperturbed = _decode_fixed_point(qcd_group["unperturbed_dft_shieldings"][()])
                perturbed = _decode_fixed_point(qcd_group["perturbed_dft_shieldings"][()])
            elif shield_type == "nn":
                unperturbed = _decode_fixed_point(qcd_group["unperturbed_nn_shieldings"][()])
                perturbed = _decode_fixed_point(qcd_group["perturbed_nn_shieldings"][()])
            else:
                raise ValueError("shield_type must be 'dft' or 'nn'")
            means = np.mean(perturbed, axis=1)
            n = means.shape[0]
            stdevs = np.std(means, axis=0)
            means = np.mean(means, axis=0)
            corrections = means - unperturbed
            stderrs = stdevs / np.sqrt(n)
            for i, (correction, error) in enumerate(zip(corrections, stderrs)):
                row = [solute, i + 1, correction, error]
                rows.append(row)
    df = pd.DataFrame(rows, columns=["solute", "atom_number", "qcd_correction", "stderr"])
    if verbose:
        print("\r\033[92mLoaded QCD data.    \033[0m")
    return df

def load_site_aggregated_explicit_shieldings(hdf5_filepath: str,
                                             site_atoms_df: pd.DataFrame,
                                             shield_type: str = "dft",
                                             verbose: bool = True):
    """Compute, per site/solvent/explicit-solvation-engine, the mean explicit-solvent correction
    (solvated minus isolated) over all frames, with its standard error."""
    if verbose:
        print("\033[93mLoading explicit shieldings data...\033[0m", end="", flush=True)

    new_rows = []
    with h5py.File(hdf5_filepath, "r") as hdf5_file:
        for (solute, nucl, site), row in site_atoms_df.iterrows():
            atom_numbers, *_ = row
            atom_indices = _fix_atom_numbers(atom_numbers, prefix="", coerce_to=int, offset=-1)

            # desmond and openMM may not cover the same solvents; take the union
            solvents = set()
            for explicit_solvation_model in EXPLICIT_SOLVATION_MODELS:
                group = hdf5_file[f"solutes/{solute}/{explicit_solvation_model}"]
                solvents.update([
                    k for k in group.keys()
                    if k != "gas_phase" and f"perturbed_{shield_type}_shieldings" in group[k]
                ])
            solvents = sorted(solvents)

            for explicit_solvation_model in EXPLICIT_SOLVATION_MODELS:
                group = hdf5_file[f"solutes/{solute}/{explicit_solvation_model}"]
                for solvent in solvents:
                    dataset = group.get(f"{solvent}/perturbed_{shield_type}_shieldings")
                    if dataset is None:
                        new_row = [solute, solvent, explicit_solvation_model, nucl, site, np.nan, 0, np.nan]
                        new_rows.append(new_row)
                        continue
                    dataset = _decode_fixed_point(dataset[()])

                    dataset = dataset[:,atom_indices,:]
                    dataset = np.mean(dataset, axis=1)
                    corrections = dataset[:,1] - dataset[:,0]
                    corrections = corrections[~np.isnan(corrections)]
                    correction = np.mean(corrections)
                    n = len(corrections)
                    stderr = np.std(corrections)/np.sqrt(n)
                    
                    new_row = [solute, solvent, explicit_solvation_model, nucl, site, correction, n, stderr]
                    new_rows.append(new_row)

    explicit_df = pd.DataFrame(new_rows)
    explicit_df.columns = [
        "solute", 
        "solvent", 
        "explicit_solvation_model", 
        "nucleus", 
        "site",
        "correction",
        "n",
        "stderr"
    ]

    if verbose:
        print("\r\033[92mLoaded explicit shieldings data.    \033[0m")

    return explicit_df

def load_classical_vibrational_corrections(hdf5_filepath: str, 
                                           site_atoms_df: pd.DataFrame, 
                                           shield_type: str = "dft",
                                           verbose: bool = True):
    """Compute classical vibrational corrections for both DFT and NN shielding values."""
    rows = []
    with h5py.File(hdf5_filepath, "r") as hdf5_file:
        solutes = hdf5_file["solutes"].keys()
        for solute in solutes:
            sites_df = site_atoms_df.query(f"solute == '{solute}'")
            for (solute, nucl, site), site_row in sites_df.iterrows():
                atom_numbers, *_ = site_row
                atom_numbers = _fix_atom_numbers(atom_numbers, prefix="", coerce_to=int, offset=-1)

                # desmond and openMM may not cover the same solvents; take the union
                solvents = set()
                for explicit_solvation_model in EXPLICIT_SOLVATION_MODELS:
                    group = hdf5_file[f"solutes/{solute}/{explicit_solvation_model}"]
                    solvents.update([
                        k for k in group.keys()
                        if k != "gas_phase" and f"perturbed_{shield_type}_shieldings" in group[k]
                    ])
                solvents = sorted(solvents)

                for explicit_solvation_model in EXPLICIT_SOLVATION_MODELS:
                    stationary_path = f"solutes/{solute}/{explicit_solvation_model}/gas_phase/unperturbed_{shield_type}_shieldings"
                    shieldings = hdf5_file[stationary_path]
                    shieldings = _decode_fixed_point(shieldings[()])
                    classical_stationary = np.mean(shieldings[atom_numbers])

                    for solvent in solvents:
                        path = f"solutes/{solute}/{explicit_solvation_model}/{solvent}/perturbed_{shield_type}_shieldings"
                        added = False
                        if path in hdf5_file:
                            shieldings = hdf5_file[path]
                            shieldings = _decode_fixed_point(shieldings[()])
                            shieldings = shieldings[:, atom_numbers, 0]  # solute shieldings

                            if np.all(np.isnan(shieldings)):
                                print(f"Found all NaNs in {path} for solute {solute}, nucleus {nucl}, site {site}, solvent {solvent}")
                            shieldings = shieldings[~np.isnan(shieldings)]

                            if len(shieldings) > 0:
                                explicit_mean = np.mean(shieldings)
                                correction = explicit_mean - classical_stationary
                                new_row = [solute, nucl, site, solvent, explicit_solvation_model, correction]
                                added = True
                        if not added:
                            new_row = [solute, nucl, site, solvent, explicit_solvation_model, np.nan]
                        rows.append(new_row)

    columns = ["solute", "nucleus", "site", "solvent", "flavor", "correction"]
    df = pd.DataFrame(rows, columns=columns)
    df = pd.pivot_table(df, values="correction", columns="flavor", index=["solute", "nucleus", "site", "solvent"])
    df.columns = [f"{col}_vib" for col in df.columns]
    return df

def load_delta22_dft_data(delta22_file_path: str, 
                          experimental_data_path: str,
                          pcm_reference_method: str = DEFAULT_PCM_REFERENCE_METHOD,
                          pcm_reference_basis: str = DEFAULT_PCM_REFERENCE_BASIS,
                          verbose: bool = True) -> pd.DataFrame:
    """
    Load and process DFT shielding data from the delta22 HDF5 file into one combined DataFrame.

    Returns a DataFrame indexed by solute, sap_geometry_type, sap_nmr_method, sap_basis, nucleus,
    site, solvent, with columns experimental, stationary, qcd, pcm, desmond, openMM, desmond_vib,
    openMM_vib.
    """
    experimental_df = load_experiment_data(experimental_data_path, verbose)
    site_atoms_df = load_site_atom_data(experimental_data_path, verbose)
    dft_solvents = load_solvents_from_methods(delta22_file_path)
    sap_shieldings_df = load_stationary_and_pcm_dft_shieldings(delta22_file_path, verbose)
    sap_shieldings_df = _aggregate_shieldings_by_site(sap_shieldings_df,
                                                      site_atoms_df)

    stationary_df = _create_stationary_df(sap_shieldings_df, dft_solvents)
    pcm_dft_corrections_df = _create_dft_pcm_correction_df(sap_shieldings_df,
                                                           stationary_df,
                                                           pcm_reference_method,
                                                           pcm_reference_basis,
                                                           verbose=False)

    qcd_data_df = load_average_qcd_corrections(delta22_file_path,
                                               "dft",
                                               verbose)
    qcd_data_df = _aggregate_qcd_by_site(qcd_data_df, site_atoms_df, verbose)
    qcd_corrections_df = _tile_qcd_corrections(qcd_data_df, dft_solvents)

    explicit_data_df = load_site_aggregated_explicit_shieldings(
        delta22_file_path,
        site_atoms_df,
        shield_type="dft",
        verbose=verbose
    )
    explicit_corrections_df = _collate_explicit_corrections(explicit_data_df)

    classical_df = load_classical_vibrational_corrections(
        delta22_file_path,
        site_atoms_df,
        shield_type="dft",
        verbose=verbose
    )

    combined_df = _build_combined_df(
        experimental_df,
        qcd_corrections_df,
        explicit_corrections_df,
        classical_df,
        stationary_df,
        pcm_dft_corrections_df,
        INDEX_COLUMNS
    )

    return combined_df


def load_delta22_nn_data(delta22_file_path: str,
                         experimental_data_path: str,
                         exclude_solutes=None,
                         verbose: bool = True,
                         nn_shieldings_override_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Load and process NN (MagNET) shielding data from the delta22 HDF5 file into one combined
    DataFrame, indexed by solute, sap_geometry_type, sap_nmr_method, sap_basis, nucleus, site,
    solvent, with columns experimental, stationary, qcd, pcm, desmond, openMM, desmond_vib,
    openMM_vib.

    MagNET only computes a PCM correction for chloroform; that value is copied unchanged to every
    other solvent.

    exclude_solutes drops solute names from the result, e.g. ["nitromethane"]. The composite-model
    fits exclude nitromethane because it is a stark outlier for MagNET (its rovibrational/QCD
    correction is anomalous).

    nn_shieldings_override_df, if given, is used in place of
    load_stationary_and_pcm_nn_shieldings(delta22_file_path) and must have the same columns/shape
    (one "gas" row and one "pcm_correction" row per solute). This lets a caller substitute
    freshly-computed shieldings (e.g. from live, symmetrized inference) for the ones stored in the
    HDF5, without touching the rest of the pipeline (QCD, explicit-solvent, and
    classical-vibrational corrections still come from the file).
    """
    experimental_df = load_experiment_data(experimental_data_path, verbose)
    site_atoms_df = load_site_atom_data(experimental_data_path, verbose)

    if nn_shieldings_override_df is not None:
        # same row-sorting / shieldings-parsing load_stationary_and_pcm_nn_shieldings applies, so
        # downstream code (_aggregate_shieldings_by_site) sees an identical shape regardless of
        # whether the shieldings came from the HDF5 or from a live override
        nn_shieldings_df = _sort_and_parse_raw_shieldings(nn_shieldings_override_df.copy())
    else:
        nn_shieldings_df = load_stationary_and_pcm_nn_shieldings(delta22_file_path,
                                                                 verbose)

    nn_shieldings_df = _aggregate_shieldings_by_site(nn_shieldings_df,
                                                     site_atoms_df,
                                                     verbose)

    nn_solvents = load_solvents_from_methods(delta22_file_path)
    stationary_df = _create_stationary_df(nn_shieldings_df, nn_solvents)
    pcm_nn_corrections_df = _create_nn_pcm_correction_df(nn_shieldings_df,
                                                         nn_solvents,
                                                         verbose=True)

    qcd_data_df = load_average_qcd_corrections(delta22_file_path, "nn", verbose)
    qcd_data_df = _aggregate_qcd_by_site(qcd_data_df, site_atoms_df, verbose)
    qcd_corrections_df = _tile_qcd_corrections(qcd_data_df, nn_solvents)

    explicit_data_df = load_site_aggregated_explicit_shieldings(
        delta22_file_path,
        site_atoms_df,
        shield_type="nn",
        verbose=verbose
    )
    explicit_corrections_df = _collate_explicit_corrections(explicit_data_df)

    classical_df = load_classical_vibrational_corrections(
        delta22_file_path,
        site_atoms_df,
        shield_type="nn",
        verbose=verbose
    )

    combined_df = _build_combined_df(
        experimental_df,
        qcd_corrections_df,
        explicit_corrections_df,
        classical_df,
        stationary_df,
        pcm_nn_corrections_df,
        INDEX_COLUMNS
    )

    if exclude_solutes:
        solute_index = combined_df.index.get_level_values("solute")
        combined_df = combined_df[~solute_index.isin(set(exclude_solutes))]

    return combined_df

def _compute_timings(df: pd.DataFrame) -> pd.DataFrame:
    """Sum a grouped shielding dataframe's geometry and NMR timing columns and return
    geometry_time, nmr_time, and their total as a Series."""
    geometry_time = df["geometry_time"].sum()
    nmr_time = df["nmr_time"].sum()
    total_time = geometry_time + nmr_time
    return pd.Series({"geometry_time":geometry_time, 
                      "nmr_time":nmr_time, 
                      "total_time":total_time})

def load_delta22_dft_timings(delta22_file_path: str) -> pd.DataFrame:
    """
    Sums DFT geometry-optimization and NMR computation times for every combination of geometry
    type, NMR method, basis, solvent model, and solvent name.
    """
    sap_dft_shieldings = load_stationary_and_pcm_dft_shieldings(delta22_file_path)

    groupings = [
        "sap_geometry_type",
        "sap_nmr_method",
        "sap_basis",
        "solvent_model",
        "solvent",
    ]
    return sap_dft_shieldings.groupby(groupings).apply(_compute_timings, 
                                       include_groups=False)

def load_delta22_nn_timings(delta22_file_path: str) -> pd.DataFrame:
    """
    Sums NN (MagNET) geometry-optimization and NMR computation times for every combination of
    geometry type, NMR method, basis, solvent model, and solvent name.
    """
    sap_nn_shieldings = load_stationary_and_pcm_nn_shieldings(delta22_file_path)

    groupings = [
        "sap_geometry_type",
        "sap_nmr_method",
        "sap_basis",
        "solvent_model",
        "solvent"
    ]
    return sap_nn_shieldings.groupby(groupings).apply(_compute_timings,
                                       include_groups=False)


if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    here = os.path.dirname(current_file_path)
    hdf5_filepath = os.path.join(here, "delta22.hdf5")
    experiment_data_filepath = os.path.join(here, "delta22_experimental.xlsx")

    print("LOADING DFT DATA:")
    print("="*60)
    dft_data = load_delta22_dft_data(hdf5_filepath, experiment_data_filepath)

    dft_data_display = dft_data.query(
        "sap_nmr_method == 'wp04' and "
        "sap_basis == 'pcSseg2' and "
        "sap_geometry_type == 'aimnet2'"
    ).copy().reset_index()
    dft_data_display.drop(columns=["sap_nmr_method", "sap_basis", "sap_geometry_type"], inplace=True)
    print("DFT Data Display:")
    print(dft_data_display.head())
    print("Solvents:")
    print(dft_data_display.reset_index().solvent.unique())

    print("LOADING NN DATA:")
    print("="*60)
    nn_data = load_delta22_nn_data(hdf5_filepath, experiment_data_filepath)
    print(nn_data.head())


