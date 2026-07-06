"""Reproduces the delta-22 composite-formula ablation study (the ablations.xlsx workbook).

Rebuilds `ablations.xlsx`: ~25 composite-formula variants (stationary geometry plus some combination
of implicit PCM, explicit Desmond, and rovibrational QCD corrections) across all 12 delta-22 solvents,
at two reference levels: DSD-PBEP86/pcSseg-3 (geometry PBE0/tz) and the MagNET-Zero training reference
(WP04/pcSseg-2 for 1H, wB97X-D/pcSseg-2 for 13C, AIMNet2 geometries). The fitting reuses delta22.py's
harness (same as si_figure_s06 at the DSD level, widened to the full formula set and both reference
levels); the openpyxl table-writing layer (color scales, %benefit formulas, best-model block)
matches the published screenshots.

Nitromethane: kept in (unlike scaling_factors.py, which drops it as an ML outlier). Per the SI, this
is a plain OLS fit, not ML, so all 22 solutes are used (10 test / 12 train per split).
"""
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule, FormulaRule

import delta22 as D

# ---------------------------------------------------------------------------
# formulas and reference-level configuration

# the full ablation formula set (one-term through three-term composites), Desmond explicit-solvent
# variant. This order is also the display order in "Multi-Term Models".
FORMULAS = [
    "stationary",
    "stationary + pcm",
    "stationary + desmond",
    "stationary + qcd",
    "stationary + desmond_vib",
    "stationary + pcm + qcd",
    "stationary + pcm + desmond_vib",
    "stationary + desmond + qcd",
    "stationary + desmond + desmond_vib",
    "stationary_plus_pcm",
    "stationary_plus_desmond",
    "stationary_plus_qcd",
    "stationary_plus_des_vib",
    "stationary_plus_pcm + qcd",
    "stationary_plus_pcm + desmond_vib",
    "stationary_plus_desmond + qcd",
    "stationary_plus_desmond + desmond_vib",
    "stationary_plus_qcd + pcm",
    "stationary_plus_qcd + desmond",
    "stationary_plus_des_vib + pcm",
    "stationary_plus_des_vib + desmond",
    "stationary_plus_pcm_plus_qcd",
    "stationary_plus_pcm_plus_des_vib",
    "stationary_plus_desmond_plus_qcd",
    "stationary_plus_desmond_plus_des_vib",
]

# display column order for the spreadsheet (distinct from D.DESMOND_SOLVENTS' iteration order)
ORDERED_SOLVENTS = [
    "chloroform", "tetrahydrofuran", "dichloromethane", "acetone", "acetonitrile",
    "dimethylsulfoxide", "trifluoroethanol", "methanol", "TIP4P", "benzene", "toluene",
    "chlorobenzene",
]

# the two reference levels the shipped workbook compares. "geometry" is the sap_geometry_type the
# stationary/pcm/etc columns were computed on; PCM's own reference level (used for the "PCM ="
# header line) is fixed at B3LYP-D3(BJ)/pcSseg-3 for both, matching load_delta22_dft_data's default.
# sheet_suffix_h/sheet_suffix_c differ within a level (the shipped workbook names the carbon sheet
# after wB97X-D, not WP04, since each nucleus uses its own MagNET-Zero training reference method).
REFERENCE_LEVELS = {
    "dsd": {
        "sheet_suffix_h": "DSD", "sheet_suffix_c": "DSD",
        "method_h": "dsd_pbep86", "basis_h": "pcSseg3", "geometry_h": "pbe0_tz",
        "method_c": "dsd_pbep86", "basis_c": "pcSseg3", "geometry_c": "pbe0_tz",
    },
    "wp04_wb97xd": {
        "sheet_suffix_h": "WP04", "sheet_suffix_c": "wB97XD",
        "method_h": "wp04", "basis_h": "pcSseg2", "geometry_h": "aimnet2",
        "method_c": "wb97xd", "basis_c": "pcSseg2", "geometry_c": "aimnet2",
    },
}
PCM_REFERENCE_METHOD = "b3lyp_d3bj"
PCM_REFERENCE_BASIS = "pcSseg3"

# vmin/vmax for the RMSE color scale, per nucleus (fixed, not data-dependent -- matches the shipped
# workbook so repeated runs render identically)
_VMIN_VMAX = {"H": (0.06, 0.28), "C": (1.2, 2.8)}

# per-nucleus labeled formula pairs the "story" sections of the sheet compare
PROTON_FORMULA_CONFIG = {
    "pair1_a": "stationary + pcm + qcd",
    "pair1_b": "stationary + desmond + qcd",
    "pair2_title": "benefit of qcd over desmond_vib",
    "pair2_a": "stationary + desmond_vib",
    "pair2_b": "stationary + qcd",
    "pair3_a": "stationary + desmond + desmond_vib",
    "pair3_b": "stationary + desmond + qcd",
    "best_multi": "stationary + desmond + qcd",
    "best_semi": "stationary_plus_qcd + desmond",
    "best_pars": "stationary_plus_desmond_plus_qcd",
}
CARBON_FORMULA_CONFIG = {
    "pair1_a": "stationary + pcm + desmond_vib",
    "pair1_b": "stationary + desmond + desmond_vib",
    "pair2_title": "benefit of desmond_vib over qcd",
    "pair2_a": "stationary + qcd",
    "pair2_b": "stationary + desmond_vib",
    "pair3_a": "stationary + desmond + qcd",
    "pair3_b": "stationary + desmond + desmond_vib",
    "best_multi": "stationary + desmond + desmond_vib",
    "best_semi": "stationary_plus_des_vib + desmond",
    "best_pars": "stationary_plus_desmond_plus_des_vib",
}
_FORMULA_CONFIG = {"H": PROTON_FORMULA_CONFIG, "C": CARBON_FORMULA_CONFIG}

# This ablation study fits plain OLS composite formulas, not an ML model. The canonical SI text
# for this ablation analysis states that "since no ML was used in this particular analysis, all 22
# molecules were considered, including nitromethane" -- other, ML-related analyses (e.g. scaling_factors.py's
# EXCLUDE_SOLUTES) drop nitromethane as an ML-training outlier; this module's default excludes nothing.
EXCLUDE_SOLUTES = ()


# ---------------------------------------------------------------------------
# fitting: reuses delta22.py's harness entirely

def _filtered_query_df(query_df_dft, method, basis, geometry, exclude_solutes=EXCLUDE_SOLUTES):
    """One (method, basis, geometry) slice of the DFT query table, composite columns added,
    any solutes in `exclude_solutes` dropped (default: none -- see EXCLUDE_SOLUTES above; this
    analysis is documented in the SI as using all 22 delta-22 solutes, since it's a plain OLS fit,
    not an ML model)."""
    sub = query_df_dft[(query_df_dft["sap_nmr_method"] == method)
                       & (query_df_dft["sap_basis"] == basis)
                       & (query_df_dft["sap_geometry_type"] == geometry)]
    sub = sub[~sub["solute"].isin(set(exclude_solutes))]
    return D.add_composite_columns(sub)


def ablation_rmse_table(query_df_dft, nucleus, method, basis, geometry, solutes,
                        formulas=FORMULAS, solvents=None, n_splits=250,
                        exclude_solutes=EXCLUDE_SOLUTES):
    """The formula x solvent test-RMSE table for one nucleus at one reference level: mean test RMSE
    over `n_splits` seeded splits (delta22.generate_solute_splits/run_fits), pivoted to formula rows
    x solvent columns, with an appended "Mean Test RMSE" column (the row mean across solvents).
    Row/column order matches FORMULAS / ORDERED_SOLVENTS. This is the DataFrame write_nucleus_table
    reads from."""
    solvents = list(solvents) if solvents is not None else ORDERED_SOLVENTS
    nuc_df = query_df_dft[query_df_dft["nucleus"] == nucleus]
    sub = _filtered_query_df(nuc_df, method, basis, geometry, exclude_solutes)
    results = D.run_fits(sub, solvents, formulas, n_splits, solutes)
    mean_rmse = results.groupby(["solvent", "formula"])["test_RMSE"].mean().reset_index()
    table = mean_rmse.pivot(index="formula", columns="solvent", values="test_RMSE")
    table = table.reindex(index=formulas, columns=solvents)
    table["Mean Test RMSE"] = table[solvents].mean(axis=1)
    return table


# ---------------------------------------------------------------------------
# openpyxl table-writing layer

_TABLE_COLUMNS = ORDERED_SOLVENTS + ["Mean Test RMSE"]
_WHITE_TEXT_THRESHOLD_FRACTION = 0.50
_PERCENT_COLOR_LIMIT = 0.40
_PERCENT_WHITE_TEXT_MIN_ABS = 0.25
_PERCENT_WHITE_TEXT_MAX_ABS = 0.40


def _canonical_formula_name(name):
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _build_formula_lookup(df):
    return {_canonical_formula_name(idx): idx for idx in df.index}


def _get_rmse(df, lookup, formula_name, col):
    key = _canonical_formula_name(formula_name)
    if key not in lookup:
        raise KeyError(f"Formula {formula_name!r} was not found in test RMSE dataframe index.")
    return float(df.loc[lookup[key], col])


def _write_benefit_formula_row(ws, row, ref_row, target_row, start_col, n_cols):
    # percent benefit = (reference - target) / reference, written as a live Excel formula so the
    # spreadsheet stays self-consistent if a cell above it is ever hand-edited
    for c in range(start_col, start_col + n_cols):
        ref = f"{get_column_letter(c)}{ref_row}"
        target = f"{get_column_letter(c)}{target_row}"
        ws.cell(row=row, column=c, value=f"=IFERROR(({ref}-{target})/{ref},0)")
        ws.cell(row=row, column=c).number_format = "0%"


def _format_method_label(method_name, basis_name):
    return f"{method_name.upper().replace('_', '-')}/{basis_name}"


def _write_color_scale_note(ws, vmin, vmax, threshold_fraction, pct_limit, pct_white_min, pct_white_max):
    note_col = 16  # column P
    threshold_value = vmin + threshold_fraction * (vmax - vmin)

    ws.cell(row=1, column=note_col, value="Color Scale Note").font = Font(bold=True)
    ws.cell(row=2, column=note_col, value=f"RMSE gradient uses fixed range [{vmin:.2f}, {vmax:.2f}] from notebook settings.")
    ws.cell(row=3, column=note_col, value=f"White text for RMSE cells below {threshold_fraction:.0%} of range (<= {threshold_value:.3f}).")
    ws.cell(row=4, column=note_col, value=f"%benefit uses fixed RdBu range [-{pct_limit:.0%}, +{pct_limit:.0%}], centered at 0%.")
    ws.cell(row=5, column=note_col, value=f"White text for %benefit where abs(value) is at least {pct_white_min:.0%}.")

    ws.cell(row=7, column=note_col, value="RMSE Legend").font = Font(bold=True)
    ws.cell(row=8, column=note_col, value=f"Low ({vmin:.2f})")
    ws.cell(row=9, column=note_col, value=f"Mid ({(vmin + vmax) / 2:.2f})")
    ws.cell(row=10, column=note_col, value=f"High ({vmax:.2f})")
    ws.cell(row=8, column=note_col + 1).fill = PatternFill(fill_type="solid", fgColor="2D004B")
    ws.cell(row=9, column=note_col + 1).fill = PatternFill(fill_type="solid", fgColor="CC4778")
    ws.cell(row=10, column=note_col + 1).fill = PatternFill(fill_type="solid", fgColor="F0F921")

    ws.cell(row=12, column=note_col, value="%Benefit Legend").font = Font(bold=True)
    ws.cell(row=13, column=note_col, value=f"Negative (-{pct_limit:.0%})")
    ws.cell(row=14, column=note_col, value="Zero")
    ws.cell(row=15, column=note_col, value=f"Positive (+{pct_limit:.0%})")
    ws.cell(row=13, column=note_col + 1).fill = PatternFill(fill_type="solid", fgColor="B2182B")
    ws.cell(row=14, column=note_col + 1).fill = PatternFill(fill_type="solid", fgColor="F7F7F7")
    ws.cell(row=15, column=note_col + 1).fill = PatternFill(fill_type="solid", fgColor="2166AC")

    ws.column_dimensions[get_column_letter(note_col)].width = 82
    ws.column_dimensions[get_column_letter(note_col + 1)].width = 14


def _apply_rmse_conditional_format(ws, rmse_rows, start_col, end_col, vmin, vmax):
    rule = ColorScaleRule(start_type="num", start_value=vmin, start_color="2D004B",
                          mid_type="num", mid_value=(vmin + vmax) / 2, mid_color="CC4778",
                          end_type="num", end_value=vmax, end_color="F0F921")
    for r in rmse_rows:
        ws.conditional_formatting.add(f"{get_column_letter(start_col)}{r}:{get_column_letter(end_col)}{r}", rule)


def _apply_percent_conditional_format(ws, pct_rows, start_col, end_col, pct_limit):
    rule = ColorScaleRule(start_type="num", start_value=-pct_limit, start_color="B2182B",
                          mid_type="num", mid_value=0, mid_color="F7F7F7",
                          end_type="num", end_value=pct_limit, end_color="2166AC")
    for r in pct_rows:
        ws.conditional_formatting.add(f"{get_column_letter(start_col)}{r}:{get_column_letter(end_col)}{r}", rule)


def _apply_percent_font_contrast(ws, pct_rows, start_col, end_col, pct_white_min, pct_white_max):
    min_s = f"{pct_white_min:.6f}"
    for r in pct_rows:
        for c in range(start_col, end_col + 1):
            addr = f"{get_column_letter(c)}{r}"
            rule = FormulaRule(formula=[f"ABS({addr})>={min_s}"], stopIfTrue=False, font=Font(color="FFFFFFFF"))
            ws.conditional_formatting.add(addr, rule)


def _apply_rmse_font_contrast(ws, rmse_rows, start_col, end_col, vmin, vmax, threshold_fraction):
    threshold_value = vmin + threshold_fraction * (vmax - vmin)
    for r in rmse_rows:
        for c in range(start_col, end_col + 1):
            cell = ws.cell(row=r, column=c)
            if isinstance(cell.value, (int, float)):
                cell.font = Font(color="FFFFFFFF" if float(cell.value) <= threshold_value else "FF000000")


def write_nucleus_table(ws, start_row, nucleus_name, stationary_desc, pcm_desc, test_rmse_df,
                        vmin, vmax, formula_config,
                        threshold_fraction=_WHITE_TEXT_THRESHOLD_FRACTION,
                        pct_limit=_PERCENT_COLOR_LIMIT,
                        pct_white_min=_PERCENT_WHITE_TEXT_MIN_ABS,
                        pct_white_max=_PERCENT_WHITE_TEXT_MAX_ABS):
    """Write one nucleus's ablation table into `ws` starting at `start_row`; returns the next free
    row. Layout: a header block, a "Multi-Term Models" section (reference stationary row; benefit of
    desmond over pcm; three labeled formula-pair comparisons from formula_config), then a "Best
    Model: Approach Comparison" section (SOTA/multi-term/semi-parsimonious/parsimonious, each vs the
    others). Every RMSE row gets a fixed-range color scale; every %benefit row is a live Excel
    formula with a diverging color scale. The layout matches the published workbook screenshots
    section-for-section."""
    lookup = _build_formula_lookup(test_rmse_df)
    c0, c1 = 1, 2
    c_end = c1 + len(_TABLE_COLUMNS) - 1
    rmse_rows, pct_rows = [], []

    def _rmse_row(row, label, formula):
        ws.cell(row=row, column=c0, value=label)
        rmse_rows.append(row)
        for i, col_name in enumerate(_TABLE_COLUMNS, start=c1):
            ws.cell(row=row, column=i, value=_get_rmse(test_rmse_df, lookup, formula, col_name))
            ws.cell(row=row, column=i).number_format = "0.000"

    def _benefit_row(row, ref_row, target_row):
        ws.cell(row=row, column=c0, value="%benefit")
        pct_rows.append(row)
        _write_benefit_formula_row(ws, row, ref_row, target_row, c1, len(_TABLE_COLUMNS))

    ws.cell(row=start_row, column=c0, value=nucleus_name).font = Font(bold=True)
    ws.cell(row=start_row, column=c1, value=f"stationary = {stationary_desc}")
    ws.cell(row=start_row + 1, column=c1, value=f"PCM = {pcm_desc}")

    ws.cell(row=start_row + 3, column=c0, value="Multi-Term Models").font = Font(bold=True)
    ws.cell(row=start_row + 4, column=c0, value="Test RMSEs (Solvent-Specific)").font = Font(bold=True)
    for i, col_name in enumerate(_TABLE_COLUMNS, start=c1):
        ws.cell(row=start_row + 4, column=i, value=col_name).font = Font(bold=True)
    ws.cell(row=start_row + 5, column=c0, value="formula").font = Font(bold=True)

    ws.cell(row=start_row + 7, column=c0, value="reference").font = Font(bold=True)
    _rmse_row(start_row + 8, "stationary", "stationary")

    ws.cell(row=start_row + 10, column=c0, value="benefit of desmond over pcm").font = Font(bold=True)
    _rmse_row(start_row + 11, "stationary + pcm", "stationary + pcm")
    _rmse_row(start_row + 12, "stationary + desmond", "stationary + desmond")
    _benefit_row(start_row + 13, start_row + 11, start_row + 12)

    _rmse_row(start_row + 15, formula_config["pair1_a"], formula_config["pair1_a"])
    _rmse_row(start_row + 16, formula_config["pair1_b"], formula_config["pair1_b"])
    _benefit_row(start_row + 17, start_row + 15, start_row + 16)

    ws.cell(row=start_row + 19, column=c0, value=formula_config["pair2_title"]).font = Font(bold=True)
    _rmse_row(start_row + 20, formula_config["pair2_a"], formula_config["pair2_a"])
    _rmse_row(start_row + 21, formula_config["pair2_b"], formula_config["pair2_b"])
    _benefit_row(start_row + 22, start_row + 20, start_row + 21)

    _rmse_row(start_row + 24, formula_config["pair3_a"], formula_config["pair3_a"])
    _rmse_row(start_row + 25, formula_config["pair3_b"], formula_config["pair3_b"])
    _benefit_row(start_row + 26, start_row + 24, start_row + 25)

    ws.cell(row=start_row + 29, column=c0, value="Best Model: Approach Comparison").font = Font(bold=True)
    _rmse_row(start_row + 31, "SOTA: stationary_plus_pcm", "stationary_plus_pcm")
    _rmse_row(start_row + 32, f"Multi-term: {formula_config['best_multi']}", formula_config["best_multi"])
    _rmse_row(start_row + 33, f"Semi-parsimonious: {formula_config['best_semi']}", formula_config["best_semi"])
    _rmse_row(start_row + 34, f"Parsimonious: {formula_config['best_pars']}", formula_config["best_pars"])

    ws.cell(row=start_row + 36, column=c0, value="%benefit Multi-term vs. SOTA")
    pct_rows.append(start_row + 36)
    _write_benefit_formula_row(ws, start_row + 36, start_row + 31, start_row + 32, c1, len(_TABLE_COLUMNS))
    ws.cell(row=start_row + 37, column=c0, value="%benefit Semi-parsimonious vs. SOTA")
    pct_rows.append(start_row + 37)
    _write_benefit_formula_row(ws, start_row + 37, start_row + 31, start_row + 33, c1, len(_TABLE_COLUMNS))
    ws.cell(row=start_row + 38, column=c0, value="%benefit Parsimonious vs. SOTA")
    pct_rows.append(start_row + 38)
    _write_benefit_formula_row(ws, start_row + 38, start_row + 31, start_row + 34, c1, len(_TABLE_COLUMNS))
    ws.cell(row=start_row + 40, column=c0, value="%benefit Semi-parsimonious vs. Multi-term")
    pct_rows.append(start_row + 40)
    _write_benefit_formula_row(ws, start_row + 40, start_row + 32, start_row + 33, c1, len(_TABLE_COLUMNS))
    ws.cell(row=start_row + 41, column=c0, value="%benefit Parsimonious vs. Multi-term")
    pct_rows.append(start_row + 41)
    _write_benefit_formula_row(ws, start_row + 41, start_row + 32, start_row + 34, c1, len(_TABLE_COLUMNS))

    ws.column_dimensions["A"].width = 52
    for col_idx in range(2, 2 + len(_TABLE_COLUMNS)):
        ws.column_dimensions[get_column_letter(col_idx)].width = 14

    _apply_rmse_conditional_format(ws, rmse_rows, c1, c_end, vmin, vmax)
    _apply_percent_conditional_format(ws, pct_rows, c1, c_end, pct_limit)
    _apply_rmse_font_contrast(ws, rmse_rows, c1, c_end, vmin, vmax, threshold_fraction)
    _apply_percent_font_contrast(ws, pct_rows, c1, c_end, pct_white_min, pct_white_max)
    _write_color_scale_note(ws, vmin, vmax, threshold_fraction, pct_limit, pct_white_min, pct_white_max)

    return start_row + 45


# ---------------------------------------------------------------------------
# top-level pipeline

def build_ablations_workbook(query_df_dft, solutes, output_path,
                             reference_levels=("dsd", "wp04_wb97xd"), n_splits=250,
                             exclude_solutes=EXCLUDE_SOLUTES, verbose=True):
    """Computes the ablation RMSE tables for both nuclei at each requested reference level and
    writes them into one workbook, one sheet per (nucleus, reference level) -- 4 sheets for the
    default two reference levels, matching the shipped ablations_updated.xlsx: "Proton (DSD)",
    "Carbon (DSD)", "Proton (WP04)", "Carbon (wB97XD)". Pass a single-element reference_levels to
    produce the plain "Proton"/"Carbon" two-sheet form for one reference level. query_df_dft is
    delta22.load_query_df_dft(...)'s output (unfiltered by method/basis); this function does the
    per-reference-level filtering itself."""
    wb = Workbook()
    first = True
    for level_key in reference_levels:
        level = REFERENCE_LEVELS[level_key]
        for nucleus, method_key, basis_key, geom_key, suffix_key in (
            ("H", "method_h", "basis_h", "geometry_h", "sheet_suffix_h"),
            ("C", "method_c", "basis_c", "geometry_c", "sheet_suffix_c"),
        ):
            method, basis, geometry = level[method_key], level[basis_key], level[geom_key]
            if verbose:
                print(f"fitting {nucleus} at {method}/{basis} ({level_key}), {n_splits} splits/solvent...",
                     flush=True)
            table = ablation_rmse_table(query_df_dft, nucleus, method, basis, geometry, solutes,
                                        n_splits=n_splits, exclude_solutes=exclude_solutes)
            nucleus_name = "Proton" if nucleus == "H" else "Carbon"
            sheet_title = (f"{nucleus_name} ({level[suffix_key]})" if len(reference_levels) > 1
                          else nucleus_name)
            ws = wb.active if first else wb.create_sheet()
            ws.title = sheet_title
            first = False
            write_nucleus_table(
                ws=ws, start_row=1, nucleus_name=nucleus_name,
                stationary_desc=_format_method_label(method, basis),
                pcm_desc=_format_method_label(PCM_REFERENCE_METHOD, PCM_REFERENCE_BASIS),
                test_rmse_df=table, vmin=_VMIN_VMAX[nucleus][0], vmax=_VMIN_VMAX[nucleus][1],
                formula_config=_FORMULA_CONFIG[nucleus],
            )
    wb.save(output_path)
    if verbose:
        print(f"wrote {output_path}", flush=True)
    return output_path
