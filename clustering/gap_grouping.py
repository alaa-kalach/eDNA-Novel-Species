#!/usr/bin/env python3
"""
Barcode-gap grouping, calibrated on the KNOWN species of each split (ABGD / ASAP idea).

Problem it targets: HDBSCAN (clustering/cluster.py) was never tuned and splits new species into
too many groups. Here the reference library itself tells us how different two sequences of the
SAME species usually are, and that sets how the new sequences are grouped.

RULES, FIXED BEFORE RUNNING
  1. Sequences to group = those called "novel" by B4 (outputs/results_kmer[_extra].csv), as for
     every other clustering file, so only the grouping step differs.
  2. Distance (main) = 1 - identity of a global pairwise alignment (Biopython PairwiseAligner:
     match 1, mismatch -1, gap open -2, gap extend -1; identity = matches / aligned columns,
     end gaps not counted), like VSEARCH identity. Secondary: the raw k-mer distance used by B4.
  3. Calibration, per split, on REFERENCE sequences only (true species known): cluster the
     reference sequences with single or average linkage, cut at t in 0, 0.0025, ..., 0.15
     (alignment) or at the 0-50% quantiles of reference distances in 1% steps (k-mers); keep the
     (linkage, t) with the highest ARI against the true reference species. Ties -> single before
     average, then smaller t.
  4. Group the novel-called sequences with that linkage and cut-off.
  5. Name each group: every member's nearest reference sequence (same distance) gives a species;
     the group takes the most common one (ties -> alphabetically first).
  Comparison (clustering/score_clusters.py): ARI and B-cubed F1 vs kmer (HDBSCAN) and
  ft_taxonomy_withraw; naming outcomes reported too.

OPTIONS (added later; the defaults reproduce the original run exactly)
  --detector=vsearch   group the sequences VSEARCH (B2, the best detector) calls novel instead of
                       B4's; output names get a "_vs" suffix. Default --detector=b4 (no suffix).
  --emb=ft_treecap90,ft_taxonomy   ALSO group on base + model distances, to test whether the tree-
                       trained model helps the barcode gap (taxonomy model = control). Rule, fixed
                       before running: D = D_base / median_base + D_model / median_model (w = 1, as
                       in placement_kmer_tree.py), each median = median distance between two
                       REFERENCE sequences in that space; base = alignment or raw k-mers. Cut-offs
                       = 0-50% quantiles of the reference D (as for gap_kmer); calibration, grouping
                       and naming exactly as rules 3-5. Spaces: gap_align_<emb>, gap_kmer_<emb>.
  --seeds=0,1,2,3,4    model seeds for --emb (default all 5); seed N > 0 adds "_sN" to the name.

Teleo only, like the other clustering runs. Needs: pip install biopython
Usage (repo root):
    python clustering/gap_grouping.py
    python clustering/score_clusters.py kmer ft_taxonomy_withraw gap_align gap_kmer phylo_tree
    python clustering/gap_grouping.py --detector=vsearch --emb=ft_treecap90,ft_taxonomy
Output: outputs/results/clusters_gap_align[_extra].csv, clusters_gap_kmer[_extra].csv,
        clusters_gap_{align,kmer}_<emb>[_sN][_vs][_extra].csv with --emb,
        outputs/results/gap_grouping_settings[_vs].csv (chosen linkage and cut-off per split),
        data/pairwise_align_identity_teleo.npz (cache of all pairwise identities, rebuilt if missing)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs" / "results"
EMB = ROOT / "outputs" / "embeddings"
DETECTORS = {"b4": ("results_kmer", ""), "vsearch": ("results_vsearch", "_vs")}
CACHE = DATA / "pairwise_align_identity_teleo.npz"
T_ALIGN = np.round(np.arange(0, 0.1501, 0.0025), 4)
Q_KMER = np.arange(0, 0.51, 0.01)


def load():
    clean = pd.read_csv(DATA / "guiana_teleo_clean.csv")
    clean["seq_id"] = clean["Specimen identification code"].astype(str)
    clean = clean[clean.in_tree].drop_duplicates("seq_id").set_index("seq_id")
    clean["dna"] = clean.teleo.str.strip().str.lower()
    k = np.load(DATA / "kmer_features_k3to6.npz", allow_pickle=True)
    X = pd.DataFrame(k["X"], index=k["seq_id"].astype(str))
    return clean, X[~X.index.duplicated()]


def identity_matrix(uniq):
    """1 - identity between every pair of distinct sequences (cached)."""
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        if list(z["seqs"]) == list(uniq):
            return z["D"]
    from Bio import Align
    al = Align.PairwiseAligner()
    al.mode, al.match_score, al.mismatch_score = "global", 1, -1
    al.open_gap_score, al.extend_gap_score = -2, -1
    n, t0 = len(uniq), time.time()
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a = al.align(uniq[i], uniq[j])[0]
            r0, r1 = a[0], a[1]
            lo = 0
            while lo < len(r0) and (r0[lo] == "-" or r1[lo] == "-"):
                lo += 1
            hi = len(r0)
            while hi > lo and (r0[hi - 1] == "-" or r1[hi - 1] == "-"):
                hi -= 1
            cols = hi - lo
            m = sum(r0[k] == r1[k] and r0[k] != "-" for k in range(lo, hi))
            D[i, j] = D[j, i] = 1 - m / cols if cols else 1.0
        if i % 100 == 0:
            print(f"  aligned {i}/{n} sequences ({time.time() - t0:.0f}s)", flush=True)
    np.savez(CACHE, seqs=np.array(uniq), D=D)
    return D


def kmer_dist(A, B):
    return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T, 0))


def calibrate(Dref, species, grid):
    """(linkage, cut-off, ARI) with the best ARI on the reference sequences."""
    cond = squareform(Dref, checks=False)
    best = None
    for method in ["single", "average"]:
        Z = linkage(cond, method)
        for t in grid:
            ari = adjusted_rand_score(species, fcluster(Z, t, "distance"))
            if best is None or ari > best[2] + 1e-12:
                best = (method, float(t), ari)
    return best


def group(Dnov, method, t):
    if len(Dnov) < 2:
        return np.zeros(len(Dnov), int)
    return fcluster(linkage(squareform(Dnov, checks=False), method), t, "distance") - 1


def name_groups(lab, Dnr, ref_sp):
    nearest = ref_sp[Dnr.argmin(1)]
    guess = np.empty(len(lab), dtype=object)
    for c in np.unique(lab):
        m = lab == c
        vc = pd.Series(nearest[m]).value_counts()
        guess[m] = sorted(vc.index[vc == vc.max()])[0]
    return guess


def emb_file(emb, seed, k):
    return EMB / f"embeddings_{emb}{'' if seed == 0 else f'_s{seed}'}_split{k}.npz"


def ref_median(D):
    return np.median(D[np.triu_indices(len(D), 1)])


def model_spaces(base, emb, seed, k, ref, nov):
    """{name: (Dref, Dnov, Dnr, grid)} for base + model distance (see OPTIONS --emb)."""
    z = np.load(emb_file(emb, seed, k), allow_pickle=True)
    V = pd.DataFrame(z["emb"], index=z["seq_id"].astype(str))
    Vr, Vn = V.loc[ref].values, V.loc[nov].values
    Mref = kmer_dist(Vr, Vr)
    mm = ref_median(Mref)
    out = {}
    for bname, (Bref, Bnov, Bnr) in base.items():
        mb = ref_median(Bref)
        Dref = Bref / mb + Mref / mm
        Dnov = Bnov / mb + kmer_dist(Vn, Vn) / mm
        Dnov = (Dnov + Dnov.T) / 2                 # remove tiny rounding differences
        np.fill_diagonal(Dnov, 0)
        np.fill_diagonal(Dref, 0)
        Dnr = Bnr / mb + kmer_dist(Vn, Vr) / mm
        grid = np.quantile(Dref[np.triu_indices(len(ref), 1)], Q_KMER)
        out[f"{bname}_{emb}{'' if seed == 0 else f'_s{seed}'}"] = (Dref, Dnov, Dnr, grid)
    return out


def main():
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    det = opts.get("detector", "b4")
    if det not in DETECTORS:
        sys.exit(f"--detector must be one of {sorted(DETECTORS)}")
    called_stem, det_tag = DETECTORS[det]
    embs = [e for e in opts.get("emb", "").split(",") if e]
    seeds = [int(x) for x in opts.get("seeds", "0,1,2,3,4").split(",")]
    missing = [emb_file(e, s, 1) for e in embs for s in seeds if not emb_file(e, s, 1).exists()]
    if missing:
        sys.exit("missing embeddings, e.g. " + str(missing[0].relative_to(ROOT)))

    clean, X = load()
    info = clean.drop_duplicates("species_name").set_index("species_name")
    uniq = sorted(clean.dna.unique())
    pos = {s: i for i, s in enumerate(uniq)}
    DA = identity_matrix(uniq)
    OUT.mkdir(parents=True, exist_ok=True)
    settings = []
    for called_file, split_file, tag in [(f"{called_stem}.csv", "splits.csv", ""),
                                         (f"{called_stem}_extra.csv", "splits_extra.csv", "_extra")]:
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
            Xr, Xn = X.loc[ref].values, X.loc[nov].values
            Dk_ref = kmer_dist(Xr, Xr)
            base = {"gap_align": (DA[np.ix_(ri, ri)], DA[np.ix_(ni, ni)], DA[np.ix_(ni, ri)]),
                    "gap_kmer": (Dk_ref, kmer_dist(Xn, Xn), kmer_dist(Xn, Xr))}
            spaces = {"gap_align": (*base["gap_align"], T_ALIGN),
                      "gap_kmer": (*base["gap_kmer"],
                                   np.quantile(Dk_ref[np.triu_indices(len(ref), 1)], Q_KMER))}
            for e in embs:
                for sd in seeds:
                    spaces.update(model_spaces(base, e, sd, k, ref, nov))
            for name, (Dref, Dnov, Dnr, grid) in spaces.items():
                method, t, ari = calibrate(Dref, ref_sp, grid)
                lab = group(Dnov, method, t)
                guess = name_groups(lab, Dnr, ref_sp)
                rows.setdefault(name, []).append(
                    pd.DataFrame({"seq_id": nov, "split_id": k, "cluster_id": lab,
                                  "guess_genus": info.loc[guess, "Genus"].values,
                                  "guess_family": info.loc[guess, "Family"].values}))
                settings.append(dict(space=name, split=k, linkage=method, cutoff=t, reference_ari=ari,
                                     n_novel_called=len(nov), n_groups=len(np.unique(lab))))
            a = {x["space"]: x for x in settings if x["split"] == k}
            print(f"split {k:2d} | {len(nov)} novel seqs | " + " | ".join(
                f"{n}: {a[n]['linkage']} -> {a[n]['n_groups']} groups" for n in list(spaces)[:4]), flush=True)
        for name, r in rows.items():
            pd.concat(r).to_csv(OUT / f"clusters_{name}{det_tag}{tag}.csv", index=False)
    pd.DataFrame(settings).to_csv(OUT / f"gap_grouping_settings{det_tag}.csv", index=False)
    print(f"wrote {len(rows)} grouping files (clusters_<space>{det_tag}[_extra].csv) and "
          f"gap_grouping_settings{det_tag}.csv in outputs/results/")


if __name__ == "__main__":
    main()
