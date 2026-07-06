"""Build magnet_test_predictions.hdf5 from the raw MagNET inference pickles.

This regenerates the released archive so the packaging is reproducible end to end. It reads Keir
Adams' per-test-set result pickles (each a {molecule_key: tuple} dict) and writes one canonical HDF5
using the release's int32 fixed-point scheme (round(value * 1e4), NaN sentinel) with content-addressed
dedup: any stored block that is byte-identical to one already written (the DFT targets are identical
across a test set's model variants) is stored once and referenced.

Usage:
    python build_magnet_test_predictions.py <source_dir_of_pickles> <output.hdf5>

The reader `magnet_test_predictions_reader.py` inverts this exactly (round-trip error <= 5e-5 ppm).
"""
import os
import glob
import pickle
import hashlib
import numpy as np
import h5py

SCALE = 1e4
NAN_SENTINEL = np.iinfo(np.int32).min


def _enc_float(a):
    a = np.asarray(a, dtype=np.float64)
    return np.where(np.isfinite(a), np.round(a * SCALE), NAN_SENTINEL).astype(np.int32)


def _kind(elem):
    if isinstance(elem, (str, np.str_)):
        return "str"
    a = np.asarray(elem)
    if a.dtype == bool:
        return "bool"
    if np.issubdtype(a.dtype, np.floating):
        return "float"
    if np.issubdtype(a.dtype, np.integer):
        return "int"
    if a.dtype.kind in ("U", "S"):
        return "str"
    raise ValueError(f"unhandled element dtype {a.dtype}")


class _Writer:
    """Writes groups into an open h5py.File, sharing byte-identical blocks."""

    def __init__(self, h):
        self.h = h
        self.store = {}
        self.n_dedup = 0

    def _put(self, grp, name, arr, **kw):
        b = np.ascontiguousarray(arr)
        key = hashlib.sha1(b.tobytes()).hexdigest() + f":{b.dtype.str}:{b.shape}"
        if key in self.store:
            grp.attrs[name + "_ref"] = self.store[key]
            self.n_dedup += 1
            return
        ds = grp.create_dataset(name, data=b, **kw)
        self.store[key] = ds.name

    def encode_pickle(self, grp, data):
        """Write one result pickle's {molecule_key: tuple} dict into `grp`, column by column."""
        keys = list(data.keys())
        n = len(keys)
        tuples = [data[k] for k in keys]
        tlen = len(tuples[0])
        grp.attrs["n"] = n
        grp.attrs["tuple_len"] = tlen
        k0 = keys[0]
        if isinstance(k0, tuple):
            grp.attrs["key_kind"] = "tuple"
            self._put(grp, "keys", np.asarray(keys, dtype=np.int64), compression="gzip", compression_opts=4)
        elif isinstance(k0, (str, np.str_)):
            grp.attrs["key_kind"] = "str"
            grp.create_dataset("keys", data=np.asarray(keys, dtype=object),
                               dtype=h5py.string_dtype(), compression="gzip", compression_opts=4)
        else:
            grp.attrs["key_kind"] = "int"
            self._put(grp, "keys", np.asarray(keys, dtype=np.int64), compression="gzip", compression_opts=4)
        kinds = [_kind(tuples[0][p]) for p in range(tlen)]
        grp.attrs["pos_kinds"] = np.asarray(kinds, dtype=h5py.string_dtype())
        for p in range(tlen):
            kind = kinds[p]
            grp.attrs[f"p{p}_kind"] = kind
            if kind == "str":
                vals = np.asarray([str(t[p]) for t in tuples], dtype=object)
                grp.create_dataset(f"p{p}_str", data=vals, dtype=h5py.string_dtype(),
                                   compression="gzip", compression_opts=4)
                continue
            arrs = [np.asarray(t[p]) for t in tuples]
            shapes = np.asarray([a.shape for a in arrs], dtype=np.int32)
            sizes = np.asarray([a.size for a in arrs], dtype=np.int64)
            offsets = np.zeros(n + 1, dtype=np.int64)
            offsets[1:] = np.cumsum(sizes)
            flat = np.concatenate([a.ravel() for a in arrs]) if n else np.array([])
            self._put(grp, f"p{p}_shapes", shapes, compression="gzip", compression_opts=4)
            self._put(grp, f"p{p}_offsets", offsets, compression="gzip", compression_opts=4)
            if kind == "float":
                self._put(grp, f"p{p}_data", _enc_float(flat), compression="gzip", compression_opts=4, shuffle=True)
            elif kind == "int":
                self._put(grp, f"p{p}_data", flat.astype(np.int64), compression="gzip", compression_opts=4, shuffle=True)
            elif kind == "bool":
                self._put(grp, f"p{p}_data", flat.astype(np.int8), compression="gzip", compression_opts=4)


def build_file(source_dir, outfile):
    """Encode every *.pickle result file in `source_dir` into `outfile` as one HDF5 archive.

    Returns (n_files_read, n_blocks_deduplicated).
    """
    files = sorted(glob.glob(os.path.join(source_dir, "*.pickle")))
    with h5py.File(outfile, "w") as h:
        h.attrs["format"] = "magnet_test_predictions v2 (dedup)"
        h.attrs["scale"] = SCALE
        h.attrs["nan_sentinel"] = NAN_SENTINEL
        w = _Writer(h)
        for f in files:
            name = os.path.basename(f)[:-7]
            if name.startswith("results_"):
                name = name[len("results_"):]
            w.encode_pickle(h.create_group(name), pickle.load(open(f, "rb")))
    return len(files), w.n_dedup


if __name__ == "__main__":
    import sys
    src, out = sys.argv[1], sys.argv[2]
    nfiles, ndedup = build_file(src, out)
    print(f"wrote {out} from {nfiles} pickles ({ndedup} shared blocks), "
          f"{os.path.getsize(out) / 1e9:.3f} GB")
