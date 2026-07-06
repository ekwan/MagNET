"""SI Figure S2: literature Diels-Alder barriers for ethylene + butadiene.

Published quantum-chemistry predictions span ~15-45 kcal/mol (experiment ~23.3). The barriers are a
literature compilation (paper reference 26); the table below is the data (no dataset/reader). This
file holds the table and its method-family grouping; the bar chart is drawn inline in the notebook.
"""
# experimental activation barrier for the ethylene + butadiene Diels-Alder reaction, kcal/mol
EXPERIMENTAL_BARRIER = 23.3

# (method // geometry, predicted barrier in kcal/mol), compiled from the literature (reference 26).
# The label before "//" is the energy method; it sets the bar's group and label.
DATA = [
    ("HF/3-21G//HF/3-21G", 35.9),
    ("HF/6-31G*//HF/6-31G*", 45.0),
    ("HF/6-31G**//HF/6-31G**", 45.9),
    ("CASSCF(6,6)/3-21G// CASSCF(6,6)/3-21G", 37.3),
    ("CASSCF(6,6)/6-31G*//B3LYP/6-31G*", 43.8),
    ("MRMP2-CAS(6,6)/6-31G*//B3LYP/6-31G*", 28.6),
    ("CASSCF(6,6)/6-31G*// CASSCF(6,6)/6-31G*", 43.8),
    ("CASSCF(6,6)/6-31G(d,p)//CASSCF(6,6)/6-31(d,p)", 44.5),
    ("CASSCF(6,6)-MP2/6-31G(d,p)//CASSCF(6,6)/6-31(d,p)", 39.4),
    ("CASSCF(6,6)/6-311+G(d,p)//CASSCF(6,6)/6-31(d,p)", 37.0),
    ("CASSCF(6,6)-MP2/6-311+G(d,p)//CASSCF(6,6)/6-31(d,p)", 40.9),
    ("CASSCF(6,6)/6-31G*//CASSCF(6,6)/6-31G*", 25.0),
    ("CASSCF(6,6)/6-31G*//CASSCF(6,6)/6-31G*", 43.6),
    ("MR-AQCC/6-31G*//MR-AQCC/6-31G*", 25.7),
    ("MR-AQCC/6-31G**//MR-AQCC/6-31G**", 25.3),
    ("MR-AQCC/6-31G**//MR-AQCC/6-31G**", 24.2),
    ("MR-AQCC/6-311G(2d,p)//MR-AQCC/6-311G(2d,p)", 23.7),
    ("MP2/6-31G*//HF/6-31G*", 16.6),
    ("MP2/6-31G*//MP2/6-31G*", 18.5),
    ("MP2/6-31G*//B3LYP/6-31G*", 20.4),
    ("MP2/6-31G**//MP2/6-31G**", 15.9),
    ("MP2/6-311G(d,p)//B3LYP/6-31G", 17.5),
    ("MP3/6-31G*//HF/6-31G*", 26.9),
    ("MP3/6-311G(d,p)//B3LYP/6-31G", 28.2),
    ("MP4(SDQ)/6-31G*//HF/6-31G*", 29.0),
    ("MP4(SDTQ)/6-31G*//HF/6-31G*", 21.9),
    ("MP4(SDTQ)/6-31G*//MP2/6-31G*", 22.4),
    ("MP4(SDTQ)/6-311G(d,p)//B3LYP/6-31G", 22.8),
    ("B3LYP/6-31G*//B3LYP/6-31G*", 23.1),
    ("B3LYP/6-31G*//B3LYP/6-31G*", 24.9),
    ("B3LYP/6-31G**//B3LYP/6-31G**", 22.4),
    ("B3LYP/6-311+G(2d,p)//B3LYP/6-311+G(2d,p)", 27.2),
    ("BPW91/6-31G*//BPW91/6-31G*", 26.2),
    ("MPW1K/6-31+G**//MPW1K/6-31+G*", 24.4),
    ("KMLYP/6-31G*//KMLYP/6-31G*", 21.1),
    ("KMLYP/6-311G//KMLYP/6-31G", 22.4),
    ("OLYP/6-31G(d)//OLYP/6-31G(d)", 26.7),
    ("OLYP/6-311+G(2d,p)//OLYP/6-311+G(2d,p)", 30.1),
    ("OLYP/6-311G(2df,2pd)// OLYP/6-311G(2df,2pd)", 29.2),
    ("O3LYP/6-31G(d)//OLYP/6-31G(d)", 26.8),
    ("O3LYP/6-311+G(2d,p)//OLYP/6-311+G(2d,p)", 30.1),
    ("O3LYP/6-311G(2df,2pd)// OLYP/6-311G(2df,2pd)", 29.3),
    ("M06-2x/6-31G(d)", 20.4),
    ("M06-2x/cc-pVTZ", 23.2),
    ("ωB97X-D/6-31G(d)", 21.7),
    ("ωB97X-D/cc-pVTZ", 25.1),
    ("B2PLYP/cc-pVDZ", 24.0),
    ("QCISD(T)/6-31G*//CASSCF(6,6)/6-31G*", 25.5),
    ("QCISD(T)/6-31G*//B3LYP/6-31G*", 25.0),
    ("CCSD/6-311G(d,p)//B3LYP/6-31G", 30.8),
    ("CCSD(T)/6-311G(d,p)//B3LYP/6-31G", 25.7),
    ("CCSD(T)/6-31G**//B3LYP/6-31G**", 27.6),
    ("G2MP2/6-311+G(3df,2p)", 24.6),
    ("G2MS/6-311+G(2df,2p)", 23.9),
    ("CBS-QB3", 22.9),
]

# the method-family groups, in the left-to-right order the figure lays them out
CATEGORY_ORDER = ["HF", "MP2", "MP3", "MP4", "DFT", "CASSCF",
                  "Multireference", "Coupled Cluster", "Composite"]


def energy_method(method_string):
    """The energy method (the part before '//') of a 'method//geometry' string. This is what the
    bar is labeled with and categorized by; a string with no '//' is returned unchanged."""
    return method_string.split("//")[0].strip() if "//" in method_string else method_string


def categorize_method(method):
    """The method family ('HF', 'MP2', 'DFT', 'CASSCF', ...) of an energy method, by the same rules
    as the original figure. Used to group the bars."""
    m = method.upper()
    if m.startswith("HF"):
        return "HF"
    if m.startswith("MP2"):
        return "MP2"
    if m.startswith("MP3"):
        return "MP3"
    if m.startswith("MP4"):
        return "MP4"
    if any(k in m for k in ("B3LYP", "BPW91", "MPW1K", "KMLYP", "OLYP", "O3LYP", "M06",
                            "ωB97X", "B2PLYP")):
        return "DFT"
    if "CASSCF" in m:
        return "CASSCF"
    if "CCSD" in m or "QCISD" in m:
        return "Coupled Cluster"
    if m.startswith("G2") or "CBS" in m:
        return "Composite"
    if "MR-AQCC" in m or "MRMP2" in m:
        return "Multireference"
    return "Other"


def ordered_layout(data=DATA, group_spacing=2):
    """Lay the bars out grouped by method family and sorted by barrier within each group, with a gap
    of group_spacing between groups. Returns a dict with parallel lists methods (the energy-method
    labels), energies, positions (x of each bar), and group_labels / group_boundaries (the
    (start, end) bar position of each group), in CATEGORY_ORDER.
    """
    grouped = {}
    for method_string, energy in data:
        method = energy_method(method_string)
        grouped.setdefault(categorize_method(method), []).append((method, energy))
    for entries in grouped.values():
        entries.sort(key=lambda me: me[1])

    methods, energies, positions = [], [], []
    group_labels, group_boundaries = [], []
    pos = 0
    for i, category in enumerate(CATEGORY_ORDER):
        if category not in grouped:
            continue
        if i > 0 and methods:   # gap before each group after the first present one
            pos += group_spacing
        start = pos
        for method, energy in grouped[category]:
            methods.append(method)
            energies.append(energy)
            positions.append(pos)
            pos += 1
        group_labels.append(category)
        group_boundaries.append((start, pos - 1))
    return {"methods": methods, "energies": energies, "positions": positions,
            "group_labels": group_labels, "group_boundaries": group_boundaries}
