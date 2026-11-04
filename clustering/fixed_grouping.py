#!/usr/bin/env python3
"""
Fixed-threshold grouping (the field's standard OTU practice) as a control for the
reference-calibrated barcode gap (clustering/gap_grouping.py).

Question: does calibrating the cut-off on the known species beat the generic identity
thresholds eDNA studies normally use?

RULES, FIXED BEFORE RUNNING
  1. Sequences to group = those VSEARCH (B2) calls novel (outputs/results_vsearch[_extra].csv),
     exactly the set gap_grouping.py --detector=vsearch grouped (clusters_gap_align_vs*.csv).
  2. Distance = the SAME 1 - alignment identity as gap_align (data/pairwise_align_identity_teleo.npz,
     built by gap_grouping.py). Only the cut-off differs from gap_align.
  3. Standard thresholds: identity 100% (exact sequence / ASV), 99%, 98%, 97% (classic OTU),
     each with single and average linkage. Cut at distance 1 - identity.
     PRIMARY control = 97%, single linkage.
  4. Sensitivity only (NOT a result; it looks at the answer): 96% and 95%, to show how close the
     calibrated cut-off gets to the best fixed cut-off chosen in hindsight.
  5. Naming = gap_grouping.name_groups (majority nearest reference species), so naming outcomes
     are comparable too.

Usage (repo root):
    python clustering/fixed_grouping.py
    python clustering/score_clusters.py --out=_fixed gap_align_vs fixed97_single_vs fixed98_single_vs \
        fixed99_single_vs fixed100_single_vs fixed97_average_vs fixed98_average_vs fixed99_average_vs \
        fixed100_average_vs fixed96_single_vs fixed95_single_vs fixed96_average_vs fixed95_average_vs
Output: outputs/results/clusters_fixed<id>_<linkage>_vs[_extra].csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "clustering"))
from gap_grouping import CACHE, DATA, OUT, group, identity_matrix, load, name_groups  # noqa: E402

IDENTITIES = [100, 99, 98, 97, 96, 95]       # 96 / 95 = sensitivity only (rule 4)
LINKAGES = ["single", "average"]


def main():
    if not CACHE.exists():
        sys.exit("missing data/pairwise_align_identity_teleo.npz - run clustering/gap_grouping.py first")
    clean, _ = load()
    info = clean.drop_duplicates("species_name").set_index("species_name")
    uniq = sorted(clean.dna.unique())
    pos = {s: i for i, s in enumerate(uniq)}
    DA = identity_matrix(uniq)                  # read from the cache, identical to gap_align
    for called_file, split_file, tag in [("results_vsearch.csv", "splits.csv", ""),
                                         ("results_vsearch_extra.csv", "splits_extra.csv", "_extra")]:
        called = pd.read_csv(ROOT / "outputs" / called_file, dtype={"seq_id": str})
        called = called[called.predicted_species == "novel"]
        splits = pd.read_csv(DATA / split_file)
        splits["seq_id"] = splits.seq_id.astype(str)
        rows = {}
        for k in sorted(splits.split_id.unique()):
            s = splits[splits.split_id == k]
            ref = s[s.role == "reference"].seq_id.values
            nov = called[called.split_id == k].seq_id.values
            if len(nov) == 0:
                continue
            ref_sp = clean.loc[ref, "species_name"].values
            ri = np.array([pos[x] for x in clean.loc[ref, "dna"]])
            ni = np.array([pos[x] for x in clean.loc[nov, "dna"]])
            Dnov, Dnr = DA[np.ix_(ni, ni)], DA[np.ix_(ni, ri)]
            for ident in IDENTITIES:
                t = 1 - ident / 100 + 1e-9         # identical sequences (d = 0) always merge
                for method in LINKAGES:
                    lab = group(Dnov, method, t)
                    guess = name_groups(lab, Dnr, ref_sp)
                    rows.setdefault(f"fixed{ident}_{method}", []).append(
                        pd.DataFrame({"seq_id": nov, "split_id": k, "cluster_id": lab,
                                      "guess_genus": info.loc[guess, "Genus"].values,
                                      "guess_family": info.loc[guess, "Family"].values}))
            print(f"split {k:2d} | {len(nov)} VSEARCH-novel sequences grouped", flush=True)
        for name, r in rows.items():
            pd.concat(r).to_csv(OUT / f"clusters_{name}_vs{tag}.csv", index=False)
    print(f"wrote clusters_fixed<id>_<linkage>_vs[_extra].csv for {len(IDENTITIES) * len(LINKAGES)} settings")


if __name__ == "__main__":
    main()
