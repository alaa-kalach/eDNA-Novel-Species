#!/usr/bin/env python3
"""
All-against-all VSEARCH identity for every in-tree French Guiana sequence, once per marker.

Same alignment and identity definition as B2 (baselines/b2_vsearch.py): global alignment,
plus strand, no masking, VSEARCH's default identity (--iddef 2: matching columns / alignment
columns, terminal gaps ignored). Because B2 searches exhaustively (maxaccepts 0, maxrejects 0),
the best hit of a query against any reference = the max of this matrix over the reference
rows, so every split can be scored from the matrix without re-running VSEARCH.

Used by models/group_cutoffs.py (per-group novelty cut-offs).

Usage (repo root):  python data/pairwise_identity.py            # Teleo   (~1 min)
                    python data/pairwise_identity.py --marker=full   # long 12S (~10 min)
Needs VSEARCH (same lookup as the baselines: $VSEARCH, PATH, or tools/vsearch[.exe]).
Output: data/pairwise_identity_teleo.npz | data/pairwise_identity_full12s.npz
        seq_id (n,), ID (n, n) float32 identity in 0-1 (diagonal = 1)
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "baselines"))
from common import find_vsearch  # noqa: E402


def load_sequences(marker):
    d = pd.read_csv(DATA / "guiana_teleo_clean.csv", dtype={"Specimen identification code": str})
    d = d[d["in_tree"]].rename(columns={"Specimen identification code": "seq_id"})
    d = d.drop_duplicates("seq_id").set_index("seq_id")
    if marker == "teleo":
        seq = d["teleo"]
    else:
        f = pd.read_csv(DATA / "guiana_full12s.csv", dtype={"seq_id": str}).set_index("seq_id")
        seq = f["seq"].reindex(d.index)
        if seq.isna().any():
            raise SystemExit(f"{seq.isna().sum()} in-tree sequences missing from guiana_full12s.csv")
    seq = seq.astype(str).str.strip().str.upper()
    return seq.sort_index()


def main():
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    marker = {"teleo": "teleo", "full": "full12s", "full12s": "full12s"}[opts.get("marker", "teleo")]
    seq = load_sequences(marker)
    ids = list(seq.index)
    pos = {s: i for i, s in enumerate(ids)}
    vs = find_vsearch()
    with tempfile.TemporaryDirectory() as tmp:
        fa, out = os.path.join(tmp, "all.fa"), os.path.join(tmp, "pairs.tsv")
        with open(fa, "w", newline="\n") as fh:
            for i, s in enumerate(seq.values):
                fh.write(f">s{i}\n{s}\n")                      # neutral names (ids may contain spaces)
        p = subprocess.run([vs, "--allpairs_global", fa, "--acceptall", "--qmask", "none",
                            "--userout", out, "--userfields", "query+target+id",
                            "--threads", str(os.cpu_count() or 1), "--quiet"],
                           capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(p.stderr[-2000:])
        pr = pd.read_csv(out, sep="\t", header=None, names=["q", "t", "id"])
    n = len(ids)
    ID = np.full((n, n), np.nan, dtype=np.float32)
    qi, ti = pr.q.str[1:].astype(int).values, pr.t.str[1:].astype(int).values
    ID[qi, ti] = pr.id.values / 100.0
    ID[ti, qi] = pr.id.values / 100.0
    np.fill_diagonal(ID, 1.0)
    if np.isnan(ID).any():
        raise SystemExit(f"{np.isnan(ID).sum()} pairs missing from VSEARCH output")
    path = DATA / f"pairwise_identity_{marker}.npz"
    np.savez_compressed(path, seq_id=np.array(ids), ID=ID)
    print(f"wrote {path}  ({n} sequences, median length {seq.str.len().median():.0f} bp)")


if __name__ == "__main__":
    main()
