#!/usr/bin/env python3
"""
Why does the VSEARCH nearest match place new species as well as tree placement?

Tree placement (models/phylo_placement.py) can only use ONE sequence per reference species (the
Fish Tree has one tip per species); its built-in control ("alignment nearest") uses the same one
sequence per species. VSEARCH (baselines/b2_vsearch.py) searches EVERY reference sequence. So two
things differ between VSEARCH and the tree pipeline: how much of the library is used, and how
the alignment is made. This check separates them by running VSEARCH twice, changing only the
reference set:

  VSEARCH all             every reference sequence              (= the B2 baseline's nearest match)
  VSEARCH 1 per species   only the representatives tree placement uses
  tree placement          representatives, on the Fish Tree     (phylo_placement_per_seq.csv)
  alignment 1 per species representatives, MAFFT alignment      (same file, the tree's control)

RULES, FIXED BEFORE RUNNING
  Representatives: exactly the rule of phylo_placement.py - per reference species, its most common
    reference sequence (ties -> alphabetically first).
  VSEARCH: exactly baselines/common.py vsearch_best_hits (exhaustive global alignment, ties at
    the best identity -> species with most tied hits, then alphabetical).
  Sequences: truly novel test sequences, Teleo, all 30 splits (placement outcomes) and the known
    test sequences (known_sp, side check).
  Outcomes per split: family (PRIMARY), family_easy, genus_hard, tree_error (My, lower = better),
    known_sp. Paired Wilcoxon over the 30 splits + bootstrap 95% CI.
  PRIMARY TEST: tree placement minus VSEARCH 1 per species, family = does the tree beat a good
    aligner when both see the same sequences?
  DECISION, fixed before running:
    - tree > VSEARCH 1 per species (95% CI above 0): the tree adds information, and VSEARCH wins
      only by using more of the library -> tree placement with ALL reference sequences is worth
      trying (the "rescue" experiment).
    - otherwise: the tree adds nothing over a good aligner on this marker -> no rescue; report the
      negative result.
  Secondary: VSEARCH all minus VSEARCH 1 per species (value of the full library), VSEARCH 1 per
    species minus alignment 1 per species (value of pairwise vs fixed-MAFFT alignment), tree minus
    VSEARCH all. Also checks that "VSEARCH all" reproduces outputs/results_vsearch*.csv.

Usage (repo root):  python models/placement_vsearch_check.py [--threads=8]
Needs:  VSEARCH (tools/vsearch.exe, PATH, or $VSEARCH), outputs/results/phylo_placement_per_seq.csv,
        outputs/results_vsearch[_extra].csv (for the reproduction check), data/global_tree_distances.npz
Output: outputs/results/placement_vsearch_check_per_seq.csv
        outputs/results/placement_vsearch_check_per_split.csv
        outputs/results/placement_vsearch_check_tests.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baselines"))
from common import find_vsearch, load_data, vsearch_best_hits  # noqa: E402

DATA, OUT = ROOT / "data", ROOT / "outputs" / "results"
METHODS = {"VSEARCH all": "vs_all", "VSEARCH 1 per species": "vs_rep",
           "tree placement": "phylo_species", "alignment 1 per species": "align_species"}
OUTCOMES = ["family", "family_easy", "genus_hard", "tree_error", "known_sp"]
PAIRS = [("tree placement", "VSEARCH 1 per species"),          # primary (family)
         ("VSEARCH all", "VSEARCH 1 per species"),
         ("VSEARCH 1 per species", "alignment 1 per species"),
         ("tree placement", "VSEARCH all"),
         ("tree placement", "alignment 1 per species")]


def representatives(ref):
    """phylo_placement.py rule: most common reference sequence per species, ties -> alphabetical."""
    keep = []
    for sp, g in ref.groupby("species_name", sort=True):
        vc = g.teleo.value_counts()
        seq = sorted(vc.index[vc == vc.max()])[0]
        keep.append(g.index[g.teleo == seq][0])          # one row carrying that sequence
    return ref.loc[keep]


def run_vsearch(threads):
    vs = find_vsearch()
    rows = []
    for split_file in ["splits.csv", "splits_extra.csv"]:
        seqs, splits = load_data(split_file)
        for k in sorted(splits.split_id.unique()):
            s = splits[splits.split_id == k].set_index("seq_id")
            ref = seqs.loc[s.index[s.role == "reference"]]
            test = seqs.loc[s.index[s.role != "reference"]]
            reps = representatives(ref)
            a = vsearch_best_hits(test, ref, vs, threads)
            r = vsearch_best_hits(test, reps, vs, threads)
            rows.append(pd.DataFrame({"split_id": k, "seq_id": test.index, "role": s.loc[test.index, "role"].values,
                                      "difficulty": s.loc[test.index, "difficulty"].values,
                                      "vs_all": a.best_species.values, "vs_all_id": a.best_id.values,
                                      "vs_rep": r.best_species.values, "vs_rep_id": r.best_id.values,
                                      "n_ref": len(ref), "n_rep": len(reps)}))
            print(f"split {k:2d} | {len(ref)} reference seqs -> {len(reps)} representatives | "
                  f"{len(test)} test seqs", flush=True)
    return pd.concat(rows, ignore_index=True)


def reproduction_check(d):
    files = [ROOT / "outputs" / f for f in ["results_vsearch.csv", "results_vsearch_extra.csv"]]
    if not all(f.exists() for f in files):
        print("reproduction check skipped: outputs/results_vsearch*.csv not found")
        return
    b2 = pd.concat([pd.read_csv(f, dtype={"seq_id": str}) for f in files])
    m = d.merge(b2[["split_id", "seq_id", "nearest_species"]], on=["split_id", "seq_id"])
    same = (m.vs_all == m.nearest_species).mean()
    print(f"reproduction check: 'VSEARCH all' = B2 nearest_species for {same:.2%} of {len(m)} test sequences")


def score(d, clean, info, row, age):
    true = clean.loc[d.seq_id, "species_name"].values
    out = d[["split_id", "seq_id", "role", "difficulty"]].copy()
    for name, col in METHODS.items():
        g = d[col].values
        if pd.isna(g).any():
            sys.exit(f"{name}: some test sequences have no guess")
        out[f"{name}|family"] = info.loc[g, "Family"].values == info.loc[true, "Family"].values
        out[f"{name}|genus"] = info.loc[g, "Genus"].values == info.loc[true, "Genus"].values
        out[f"{name}|species"] = g == true
        out[f"{name}|tree_error"] = age[[row[x] for x in g], [row[x] for x in true]]
    rows = []
    for k, g in out.groupby("split_id"):
        nov = (g.role == "test_novel").values
        easy, hard = nov & (g.difficulty == "easy").values, nov & (g.difficulty == "hard").values
        for name in METHODS:
            f, ge = g[f"{name}|family"].values, g[f"{name}|genus"].values
            rows.append(dict(split=k, method=name, family=f[nov].mean(), family_easy=f[easy].mean(),
                             genus_hard=ge[hard].mean(), tree_error=g[f"{name}|tree_error"].values[nov].mean(),
                             known_sp=g[f"{name}|species"].values[~nov].mean()))
    return out, pd.DataFrame(rows)


def tests(ps, rng):
    out = []
    for a, b in PAIRS:
        A, B = ps[ps.method == a].set_index("split"), ps[ps.method == b].set_index("split")
        for o in OUTCOMES:
            d = (A[o] - B[o]).dropna()
            boots = [rng.choice(d.values, len(d)).mean() for _ in range(5000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            p = wilcoxon(d.values).pvalue if (d != 0).any() else 1.0
            better, worse = ((d < 0), (d > 0)) if o == "tree_error" else ((d > 0), (d < 0))
            out.append(dict(a=a, b=b, outcome=o, primary=((a, b) == PAIRS[0] and o == "family"),
                            mean_a=A[o].mean(), mean_b=B[o].mean(), diff=d.mean(), ci_lo=lo, ci_hi=hi,
                            a_better=int(better.sum()), b_better=int(worse.sum()), n=len(d), wilcoxon_p=p))
    return pd.DataFrame(out)


def main():
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    threads = int(opts.get("threads", 0))
    pf = OUT / "phylo_placement_per_seq.csv"
    tf = DATA / "global_tree_distances.npz"
    for f in (pf, tf):
        if not f.exists():
            sys.exit(f"missing {f.relative_to(ROOT)}")
    clean = pd.read_csv(DATA / "guiana_teleo_clean.csv")
    clean["seq_id"] = clean["Specimen identification code"].astype(str)
    clean = clean[clean.in_tree].drop_duplicates("seq_id").set_index("seq_id")
    info = clean.drop_duplicates("species_name").set_index("species_name")
    z = np.load(tf)
    row, age = {s: i for i, s in enumerate(z["species"])}, z["D"] * float(z["scale"]) / 2

    d = run_vsearch(threads)
    reproduction_check(d)
    ph = pd.read_csv(pf, dtype={"seq_id": str})[["split_id", "seq_id", "phylo_species", "align_species"]]
    d = d.merge(ph, on=["split_id", "seq_id"], how="left")
    if d.phylo_species.isna().any():
        sys.exit("phylo_placement_per_seq.csv is missing some test sequences - rerun models/phylo_placement.py")
    per_seq, ps = score(d, clean, info, row, age)
    t = tests(ps, np.random.default_rng(0))

    print(f"\nTeleo, {ps.split.nunique()} splits; placement outcomes on truly novel sequences "
          "(tree_error in My, lower = better); known_sp on known sequences")
    print(f"representatives per split: {d.groupby('split_id').n_rep.first().mean():.0f} of "
          f"{d.groupby('split_id').n_ref.first().mean():.0f} reference sequences on average")
    print(ps.groupby("method")[OUTCOMES].mean().loc[list(METHODS)].round(3).to_string())
    print("\npaired comparisons (Wilcoxon over splits; diff = a - b)")
    for r in t.itertuples():
        tag = "  <- PRIMARY" if r.primary else ""
        print(f"  {r.a} vs {r.b} | {r.outcome:11s} {r.mean_a:.3f} vs {r.mean_b:.3f} diff {r.diff:+.3f} "
              f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] better/worse {r.a_better}/{r.b_better} p = {r.wilcoxon_p:.3g}{tag}")
    p = t[t.primary].iloc[0]
    verdict = ("tree beats VSEARCH on the same sequences -> rescue experiment worth trying"
               if p.ci_lo > 0 else "tree does NOT beat VSEARCH on the same sequences -> no rescue; negative result")
    print(f"\nDECISION (fixed in advance): tree - VSEARCH 1 per species, family = {p['diff']:+.3f} "
          f"[{p.ci_lo:+.3f}, {p.ci_hi:+.3f}] -> {verdict}")

    kw = dict(index=False, lineterminator="\n")
    OUT.mkdir(parents=True, exist_ok=True)
    d.drop(columns=["n_ref", "n_rep"]).to_csv(OUT / "placement_vsearch_check_per_seq.csv", **kw)
    ps.to_csv(OUT / "placement_vsearch_check_per_split.csv", **kw)
    t.to_csv(OUT / "placement_vsearch_check_tests.csv", **kw)
    print("\nwrote outputs/results/placement_vsearch_check_per_seq.csv, _per_split.csv, _tests.csv")


if __name__ == "__main__":
    main()
