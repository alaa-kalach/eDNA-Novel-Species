#!/usr/bin/env python3
"""
CONFIRMATION on independent data: does the tree break VSEARCH's cross-genus ties better than
VSEARCH's own tie rule?

Exploratory finding on French Guiana (models/tree_tiebreak.py, chosen AFTER seeing the results):
when VSEARCH's best hits are tied between species of different genera, tree placement put the new
species in the correct family more often (53.1% vs 47.8%, +6.6 points per split, p = 0.016).
Because that subset was picked after looking, it is tested here ONCE, on fish from around the
world with no French Guiana species, with every rule fixed before running.

DATA (exactly Leen's global check, models/phylo_global_check.py)
  Global Teleo sequences from models/pretrain.py load_global (in the tree, not French Guiana, no
  sequence identical to a French Guiana one); seq_id = g0, g1, ... in that order; roles from
  held_out_set (rng(0): 10% of species novel, one held-out sequence of every other species with
  >= 2 distinct sequences known, the rest reference). Both functions are copied below so this
  script does not need the tree tools; the script STOPS unless its test sequences and species
  match outputs/results/phylo_global_check_per_seq.csv exactly.
  Tree guesses = phylo_species in that file (EPA-ng placement on the Fish Tree, already run).
  Family: data/global_taxonomy.csv; genus = first word of the species name (as Leen's script);
  novel sequences without a family are left out (as Leen's script).

RULES, FIXED BEFORE RUNNING (no tuning; nothing is chosen on these data)
  VSEARCH: same settings as baselines/common.py vsearch_best_hits (exhaustive global alignment of
    each novel sequence against EVERY reference sequence, --top_hits_only = all hits tied at the
    best identity). VSEARCH's answer = the B2 rule: best identity; ties -> species with most tied
    hits, then alphabetical.
  Tied candidates = reference species among the hits at the best identity (margin 0).
  Ambiguous = the tied candidates belong to more than one GENUS.
  Tie-break rule = the tree's guess on ambiguous sequences, VSEARCH's answer otherwise
    (the "free" variant at margin 0, as in the exploratory finding).
  PRIMARY TEST: on the ambiguous novel sequences, family correct of the tree's guess minus
    VSEARCH's answer; mean over novel SPECIES (each species = mean of its sequences), 95%
    bootstrap CI over species (2000 resamples, rng(0); as phylo_global_check.py).
  DECISION: CONFIRMED if the 95% CI is above 0; otherwise NOT confirmed (also if too few ties to
    test: fewer than 2 ambiguous species -> NOT confirmed, untestable).
  Secondary (reported, not used to decide): genus correct and tree error on the ambiguous
    sequences; the tie-break rule vs VSEARCH vs tree on ALL novel sequences (family,
    family_easy, genus_hard, tree error).

Usage (repo root):  python models/tiebreak_global_check.py [--threads=8]
Needs:  VSEARCH (tools/vsearch.exe, PATH, or $VSEARCH)
        data/global_teleo.csv, data/guiana_teleo_clean.csv, data/global_taxonomy.csv,
        data/global_tree_distances.npz, outputs/results/phylo_global_check_per_seq.csv
Output: outputs/results/tiebreak_global_per_seq.csv   (one row per novel sequence)
        outputs/results/tiebreak_global_summary.csv   (all outcomes, with CIs)
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baselines"))
from common import _run, find_vsearch  # noqa: E402

DATA, OUT = ROOT / "data", ROOT / "outputs" / "results"
PHYLO = OUT / "phylo_global_check_per_seq.csv"


# ------------------------------------------------- copied from Leen's code (do not change)
def load_global():
    """models/pretrain.py load_global(marker='teleo')."""
    g = pd.read_csv(DATA / "global_teleo.csv")
    fg_seqs = set(pd.read_csv(DATA / "guiana_teleo_clean.csv")["teleo"].str.strip().str.lower())
    return g[g.in_tree & ~g.in_guiana & ~g.teleo.isin(fg_seqs)].reset_index(drop=True)


def held_out_set(g):
    """models/phylo_global_check.py held_out_set (same roles as select_treeloss.py)."""
    rng = np.random.default_rng(0)
    species = np.array(sorted(g.species_name.unique()))
    novel = set(rng.choice(species, size=len(species) // 10, replace=False))
    role = np.where(g.species_name.isin(novel), "test_novel", "reference").astype(object)
    for sp, grp in g[~g.species_name.isin(novel)].groupby("species_name"):
        seqs = sorted(grp.teleo.unique())
        if len(seqs) >= 2:
            role[grp.index[grp.teleo == seqs[rng.integers(len(seqs))]]] = "test_known"
    return role
# -------------------------------------------------------------------------------------------


def top_hits(query, ref, vsearch, threads):
    """All reference hits tied at each query's best identity (vsearch_best_hits settings)."""
    with tempfile.TemporaryDirectory() as tmp:
        qf, rf, of = (os.path.join(tmp, x) for x in ("q.fa", "r.fa", "hits.tsv"))
        for df, path in [(query, qf), (ref, rf)]:
            with open(path, "w", newline="\n") as fh:
                for sid, s in zip(df.seq_id, df.teleo):
                    fh.write(f">{sid}\n{s.upper()}\n")
        _run([vsearch, "--usearch_global", qf, "--db", rf, "--id", "0.0",
              "--maxaccepts", "0", "--maxrejects", "0", "--top_hits_only", "--minwordmatches", "0",
              "--strand", "plus", "--qmask", "none", "--dbmask", "none",
              "--userout", of, "--userfields", "query+target+id",
              "--output_no_hits", "--threads", str(threads), "--quiet"])
        h = pd.read_csv(of, sep="\t", header=None, names=["q", "t", "id"], dtype={"q": str, "t": str})
    h = h[h.t != "*"].copy()
    h["sp"] = ref.set_index("seq_id").species_name.reindex(h.t).values
    return h


def species_ci(d, species, rng):
    """Mean over species of d (each species = mean of its sequences) + 95% bootstrap CI."""
    s = pd.Series(d).groupby(np.asarray(species)).mean()
    if len(s) < 2:
        return s.mean() if len(s) else np.nan, np.nan, np.nan, len(s)
    boots = [rng.choice(s.values, len(s)).mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return s.mean(), lo, hi, len(s)


def main():
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    threads = int(opts.get("threads", 0))
    for f in [PHYLO, DATA / "global_teleo.csv", DATA / "global_taxonomy.csv", DATA / "global_tree_distances.npz"]:
        if not f.exists():
            sys.exit(f"missing {f.relative_to(ROOT)}")

    g = load_global()
    g["teleo"] = g.teleo.str.lower()
    g["seq_id"] = [f"g{i}" for i in range(len(g))]
    g["role"] = held_out_set(g)

    # the reconstruction must match Leen's saved global check exactly
    ph = pd.read_csv(PHYLO)
    mine = g[g.role != "reference"][["seq_id", "species_name", "role"]].reset_index(drop=True)
    theirs = ph[["seq_id", "species_name", "role"]].reset_index(drop=True)
    if not mine.equals(theirs):
        sys.exit("the rebuilt held-out set does NOT match phylo_global_check_per_seq.csv "
                 "(different data files?) - stopping, nothing written")
    print(f"held-out set matches Leen's global check: {len(g)} sequences | reference "
          f"{(g.role == 'reference').sum()} | novel {(g.role == 'test_novel').sum()} seqs / "
          f"{g[g.role == 'test_novel'].species_name.nunique()} species")

    tax = pd.read_csv(DATA / "global_taxonomy.csv").drop_duplicates("species_name").set_index("species_name")
    fam_of = tax["Family"].to_dict()
    gen_of = lambda s: s.split()[0]  # noqa: E731
    z = np.load(DATA / "global_tree_distances.npz")
    trow, scale = {s: i for i, s in enumerate(z["species"])}, float(z["scale"])

    ref, nov = g[g.role == "reference"], g[g.role == "test_novel"]
    h = top_hits(nov, ref, find_vsearch(), threads)
    rows = []
    for q, grp in h.groupby("q", sort=False):
        counts = grp.sp.value_counts()
        answer = sorted(counts.index[counts == counts.max()])[0]          # B2 tie rule
        tied = sorted(set(grp.sp))
        rows.append(dict(seq_id=q, vsearch=answer, best_identity=grp.id.max() / 100,
                         n_tied_species=len(tied), n_tied_genera=len({gen_of(s) for s in tied}),
                         tied_species="; ".join(tied)))
    v = pd.DataFrame(rows)
    d = nov[["seq_id", "species_name"]].merge(v, on="seq_id", how="left").merge(
        ph[["seq_id", "phylo_species"]], on="seq_id")
    if d.vsearch.isna().any():
        sys.exit(f"{d.vsearch.isna().sum()} novel sequences got no VSEARCH hit")
    d["family"] = d.species_name.map(fam_of)
    d["ambiguous"] = d.n_tied_genera > 1
    d["tiebreak"] = np.where(d.ambiguous, d.phylo_species, d.vsearch)
    ref_genera = {gen_of(s) for s in ref.species_name}
    d["easy"] = [gen_of(s) not in ref_genera for s in d.species_name]
    for m in ["vsearch", "phylo_species", "tiebreak"]:
        d[f"{m}|family"] = (d[m].map(fam_of).values == d.family.values) & d.family.notna().values
        d[f"{m}|genus"] = [gen_of(a) == gen_of(b) for a, b in zip(d[m], d.species_name)]
        d[f"{m}|tree_error"] = z["D"][[trow[x] for x in d[m]], [trow[x] for x in d.species_name]] * scale / 2

    rng = np.random.default_rng(0)
    has_fam = d.family.notna().values
    amb = d.ambiguous.values & has_fam
    out = []

    def add(subset, what, a, b, outcome, mask):
        diff = (d.loc[mask, f"{a}|{outcome}"].astype(float).values - d.loc[mask, f"{b}|{outcome}"].astype(float).values)
        mean, lo, hi, n = species_ci(diff, d.loc[mask, "species_name"].values, rng)
        out.append(dict(subset=subset, comparison=what, outcome=outcome, n_seqs=int(mask.sum()), n_species=n,
                        mean_a=d.loc[mask, f"{a}|{outcome}"].astype(float).mean(),
                        mean_b=d.loc[mask, f"{b}|{outcome}"].astype(float).mean(),
                        diff_species_mean=mean, ci_lo=lo, ci_hi=hi))

    add("ambiguous (cross-genus exact ties)", "tree - VSEARCH", "phylo_species", "vsearch", "family", amb)  # PRIMARY
    add("ambiguous (cross-genus exact ties)", "tree - VSEARCH", "phylo_species", "vsearch", "genus", amb)
    add("ambiguous (cross-genus exact ties)", "tree - VSEARCH", "phylo_species", "vsearch", "tree_error", amb)
    for name, mask in [("all novel", has_fam), ("all novel, no same-genus reference", has_fam & d.easy.values)]:
        add(name, "tie-break rule - VSEARCH", "tiebreak", "vsearch", "family", mask)
        add(name, "tie-break rule - tree", "tiebreak", "phylo_species", "family", mask)
    hard = has_fam & ~d.easy.values
    add("all novel, same-genus reference", "tie-break rule - VSEARCH", "tiebreak", "vsearch", "genus", hard)
    add("all novel", "tie-break rule - VSEARCH", "tiebreak", "vsearch", "tree_error", has_fam)
    res = pd.DataFrame(out)

    print(f"\nnovel sequences with a family: {has_fam.sum()} | cross-genus exact ties (ambiguous): "
          f"{amb.sum()} seqs ({amb.mean() / has_fam.mean():.1%}) from {d.loc[amb, 'species_name'].nunique()} species")
    print("\n(tree_error in My: negative diff = tree / rule better; species-level means, 95% bootstrap CI over species)")
    for r in res.itertuples():
        tag = "  <- PRIMARY" if r.Index == 0 else ""
        print(f"  {r.subset:38s} {r.comparison:26s} {r.outcome:10s} {r.mean_a:7.3f} vs {r.mean_b:7.3f} "
              f"diff {r.diff_species_mean:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] "
              f"({r.n_seqs} seqs, {r.n_species} species){tag}")
    p = res.iloc[0]
    if p.n_species < 2 or np.isnan(p.ci_lo):
        verdict = "NOT confirmed (too few cross-genus ties to test)"
    else:
        verdict = "CONFIRMED" if p.ci_lo > 0 else "NOT confirmed"
    print(f"\nDECISION (fixed in advance): tree - VSEARCH on cross-genus ties, family = "
          f"{p.diff_species_mean:+.3f} [{p.ci_lo:+.3f}, {p.ci_hi:+.3f}], {p.n_species} species -> {verdict}")

    kw = dict(index=False, lineterminator="\n")
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "tiebreak_global_per_seq.csv", **kw)
    res.to_csv(OUT / "tiebreak_global_summary.csv", **kw)
    print("\nwrote outputs/results/tiebreak_global_per_seq.csv and tiebreak_global_summary.csv")


if __name__ == "__main__":
    main()
