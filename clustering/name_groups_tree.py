#!/usr/bin/env python3
"""
Name each group of new sequences by a VOTE OF ITS MEMBERS' TREE PLACEMENTS.

The pipeline so far: VSEARCH calls new sequences -> barcode-gap grouping (gap_grouping.py
--detector=vsearch) -> each group named by the most common nearest-alignment species of its
members. Tree placement (models/phylo_placement.py) places single sequences better than alignment.
Here the two are joined: a group takes the species most of its members are PLACED next to on the
Fish Tree. No new run: groups and per-sequence placements are already on disk.

RULES, FIXED BEFORE RUNNING
  Sequences: those VSEARCH calls new (the gap_align_vs groups), Teleo, all 30 splits.
  Naming outcomes are scored on the truly novel ones only (as clustering/score_clusters.py).
  Member guesses, both from outputs/results/phylo_placement_per_seq.csv (same alignment, so the
    only difference is the tree):
      tree       phylo_species  (species nearest the EPA-ng attachment point)
      alignment  align_species  (nearest reference sequence by identity, same MAFFT alignment)
  Group name = the most common member guess (ties -> alphabetically first), the same rule as
    gap_grouping.py; its genus and family are the group's guess. Secondary rule "family first":
    the most common member FAMILY, then the most common genus among members guessing that family
    (no species, so no tree error).
  Methods, all on the same sequences:
    gap_align groups + tree vote            <- the new pipeline
    gap_align groups + alignment vote       <- same groups, no tree (PRIMARY comparison)
    gap_align groups + current naming       (guess_family / guess_genus already in the cluster file)
    gap_align groups + tree vote (family first)
    tree / alignment per sequence           (no grouping: does pooling a group's members help?)
    treecap90 groups (gap_align_ft_treecap90_vs, 5 seeds averaged) + current naming / + tree vote
  Outcomes per split: family (PRIMARY), family_easy (no same-genus species in the reference),
    genus_hard (one is), tree_error (My between guessed and true species; lower = better).
  PRIMARY TEST: tree vote minus alignment vote on the gap_align groups, family, paired Wilcoxon
    over the 30 splits + bootstrap 95% CI. Everything else is secondary.

Usage (repo root):  python clustering/name_groups_tree.py
Needs:  outputs/results/clusters_gap_align_vs[_extra].csv and clusters_gap_align_ft_treecap90[_sN]_vs[_extra].csv
        (python clustering/gap_grouping.py --detector=vsearch --emb=ft_treecap90,ft_taxonomy)
        outputs/results/phylo_placement_per_seq.csv (models/phylo_placement.py)
        data/global_tree_distances.npz (models/tree_distances.py)
Output: outputs/results/group_naming_per_seq.csv   (gap_align_vs groups: each sequence's guesses)
        outputs/results/group_naming_per_split.csv (split x method; treecap90 groups seed-averaged)
        outputs/results/group_naming_tests.csv     (paired comparisons)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs" / "results"
MAIN_GROUPS = "gap_align_vs"
TREE_GROUPS = "gap_align_ft_treecap90_vs"        # seeds: _s1.._s4 inserted before _vs
OUTCOMES = ["family", "family_easy", "genus_hard", "tree_error"]
NEW, BASE = "gap_align groups + tree vote", "gap_align groups + alignment vote"
PAIRS = [(NEW, BASE),                                                   # primary (family)
         (NEW, "gap_align groups + current naming"),
         (NEW, "tree per sequence"),
         ("gap_align groups + tree vote (family first)", NEW),
         (NEW, "treecap90 groups + current naming"),
         ("treecap90 groups + tree vote", NEW),
         ("gap_align groups + alignment vote", "alignment per sequence")]


def load():
    clean = pd.read_csv(DATA / "guiana_teleo_clean.csv")
    clean["seq_id"] = clean["Specimen identification code"].astype(str)
    clean = clean[clean.in_tree].drop_duplicates("seq_id").set_index("seq_id")
    splits = pd.concat([pd.read_csv(DATA / "splits.csv"), pd.read_csv(DATA / "splits_extra.csv")])
    splits["seq_id"] = splits.seq_id.astype(str)
    tf = DATA / "global_tree_distances.npz"
    if not tf.exists():
        sys.exit(f"missing {tf} - run: python models/tree_distances.py")
    z = np.load(tf)
    row = {s: i for i, s in enumerate(z["species"])}
    pf = OUT / "phylo_placement_per_seq.csv"
    if not pf.exists():
        sys.exit(f"missing {pf.relative_to(ROOT)} - run: python models/phylo_placement.py")
    ph = pd.read_csv(pf, dtype={"seq_id": str}).set_index(["split_id", "seq_id"])
    return clean, splits, row, z["D"] * float(z["scale"]) / 2, ph


def load_groups(name):
    files = [OUT / f"clusters_{name}.csv", OUT / f"clusters_{name}_extra.csv"]
    found = [pd.read_csv(f, dtype={"seq_id": str}) for f in files if f.exists()]
    return pd.concat(found, ignore_index=True) if found else None


def vote(labels, guesses):
    """Most common guess per group (ties -> alphabetically first), returned per member."""
    out = np.empty(len(labels), dtype=object)
    for c in np.unique(labels):
        m = labels == c
        vc = pd.Series(guesses[m]).value_counts()
        out[m] = sorted(vc.index[vc == vc.max()])[0]
    return out


def vote_family_first(labels, fam, gen):
    """Most common family, then the most common genus among members guessing that family."""
    f_out, g_out = np.empty(len(labels), dtype=object), np.empty(len(labels), dtype=object)
    for c in np.unique(labels):
        m = labels == c
        vc = pd.Series(fam[m]).value_counts()
        f = sorted(vc.index[vc == vc.max()])[0]
        vg = pd.Series(gen[m & (fam == f)]).value_counts()
        f_out[m], g_out[m] = f, sorted(vg.index[vg == vg.max()])[0]
    return f_out, g_out


class Scorer:
    def __init__(self, clean, splits, row, age):
        self.clean, self.row, self.age = clean, row, age
        self.info = clean.drop_duplicates("species_name").set_index("species_name")
        self.role = splits.set_index(["split_id", "seq_id"])[["role", "difficulty"]]

    def outcomes(self, g, fam, gen, species=None):
        """g: rows of one split (seq_id, split_id); fam/gen guesses; species guess for tree error."""
        r = self.role.loc[list(zip(g.split_id, g.seq_id))]
        nov = (r.role == "test_novel").values
        easy, hard = nov & (r.difficulty == "easy").values, nov & (r.difficulty == "hard").values
        t = self.clean.loc[g.seq_id]
        f_ok, g_ok = np.asarray(fam) == t.Family.values, np.asarray(gen) == t.Genus.values
        m = lambda x: x.mean() if len(x) else np.nan  # noqa: E731
        err = np.nan
        if species is not None:
            e = self.age[[self.row[s] for s in species], [self.row[s] for s in t.species_name]]
            err = m(e[nov])
        return dict(family=m(f_ok[nov]), family_easy=m(f_ok[easy]), genus_hard=m(g_ok[hard]), tree_error=err)


def name_split(g, ph, info):
    """Every naming of one split's groups: {method suffix: (family, genus, species or None)}."""
    key = list(zip(g.split_id, g.seq_id))
    lab = g.cluster_id.values
    tree_sp, aln_sp = ph.loc[key, "phylo_species"].values, ph.loc[key, "align_species"].values
    out = {}
    for tag, sp in [("tree", tree_sp), ("alignment", aln_sp)]:
        v = vote(lab, sp)
        out[f"{tag} vote"] = (info.loc[v, "Family"].values, info.loc[v, "Genus"].values, v)
        out[f"{tag} per sequence"] = (info.loc[sp, "Family"].values, info.loc[sp, "Genus"].values, sp)
    f, gg = vote_family_first(lab, info.loc[tree_sp, "Family"].values, info.loc[tree_sp, "Genus"].values)
    out["tree vote (family first)"] = (f, gg, None)
    out["current naming"] = (g.guess_family.values, g.guess_genus.values, None)
    return out


def run(clean, splits, row, age, ph):
    sc = Scorer(clean, splits, row, age)
    info = sc.info
    main = load_groups(MAIN_GROUPS)
    if main is None:
        sys.exit(f"missing clusters_{MAIN_GROUPS}*.csv - run gap_grouping.py --detector=vsearch --emb=ft_treecap90,ft_taxonomy")
    rows, per_seq = [], []
    for k, g in main.groupby("split_id"):
        g = g.reset_index(drop=True)
        names = name_split(g, ph, info)
        for suffix, (f, gg, sp) in names.items():
            method = suffix if suffix.endswith("per sequence") else f"gap_align groups + {suffix}"
            rows.append(dict(split=k, method=method, n_seeds=1, **sc.outcomes(g, f, gg, sp)))
        per_seq.append(g[["split_id", "seq_id", "cluster_id"]].assign(
            tree_vote=names["tree vote"][2], alignment_vote=names["alignment vote"][2],
            tree_per_seq=names["tree per sequence"][2], current_family=g.guess_family.values))

    seeds = [(0, TREE_GROUPS)] + [(s, TREE_GROUPS.replace("_vs", f"_s{s}_vs")) for s in range(1, 10)]
    tg_rows = []
    for s, name in seeds:
        tg = load_groups(name)
        if tg is None:
            continue
        for k, g in tg.groupby("split_id"):
            g = g.reset_index(drop=True)
            names = name_split(g, ph, info)
            for suffix in ["current naming", "tree vote"]:
                f, gg, sp = names[suffix]
                tg_rows.append(dict(split=k, method=f"treecap90 groups + {suffix}", seed=s,
                                    **sc.outcomes(g, f, gg, sp)))
    if tg_rows:
        tg = pd.DataFrame(tg_rows)
        n = tg.groupby("method").seed.nunique()
        tg = tg.groupby(["split", "method"], as_index=False)[OUTCOMES].mean()
        tg["n_seeds"] = tg.method.map(n)
        rows += tg.to_dict("records")
    else:
        print(f"note: no clusters_{TREE_GROUPS}*.csv - treecap90 groups skipped")
    return pd.DataFrame(rows), pd.concat(per_seq, ignore_index=True)


def tests(ps, rng):
    out = []
    for a, b in PAIRS:
        A, B = ps[ps.method == a].set_index("split"), ps[ps.method == b].set_index("split")
        if A.empty or B.empty:
            continue
        for o in OUTCOMES:
            d = (A[o] - B[o]).dropna()
            if len(d) < 2:
                continue
            boots = [rng.choice(d.values, len(d)).mean() for _ in range(5000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            p = wilcoxon(d.values).pvalue if (d != 0).any() else 1.0
            better, worse = ((d < 0), (d > 0)) if o == "tree_error" else ((d > 0), (d < 0))
            out.append(dict(a=a, b=b, outcome=o, primary=((a, b) == PAIRS[0] and o == "family"),
                            mean_a=A[o].mean(), mean_b=B[o].mean(), diff=d.mean(), ci_lo=lo, ci_hi=hi,
                            a_better=int(better.sum()), b_better=int(worse.sum()), n=len(d), wilcoxon_p=p))
    return pd.DataFrame(out)


def main():
    clean, splits, row, age, ph = load()
    ps, per_seq = run(clean, splits, row, age, ph)
    t = tests(ps, np.random.default_rng(0))
    order = [NEW, BASE, "gap_align groups + current naming", "gap_align groups + tree vote (family first)",
             "treecap90 groups + tree vote", "treecap90 groups + current naming",
             "tree per sequence", "alignment per sequence"]
    tab = ps.groupby("method")[OUTCOMES].mean()
    print(f"VSEARCH-called sequences, Teleo, {ps.split.nunique()} splits; naming scored on truly novel ones")
    print("mean over splits (tree_error in My, lower = better):")
    print(tab.loc[[m for m in order if m in tab.index]].round(3).to_string())
    print("\npaired comparisons (Wilcoxon over splits; diff = a - b)")
    for r in t.itertuples():
        tag = "  <- PRIMARY" if r.primary else ""
        print(f"  {r.a} vs {r.b} | {r.outcome:11s} {r.mean_a:.3f} vs {r.mean_b:.3f} diff {r.diff:+.3f} "
              f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] better/worse {r.a_better}/{r.b_better} p = {r.wilcoxon_p:.3g}{tag}")
    kw = dict(index=False, lineterminator="\n")
    OUT.mkdir(parents=True, exist_ok=True)
    per_seq.to_csv(OUT / "group_naming_per_seq.csv", **kw)
    ps.to_csv(OUT / "group_naming_per_split.csv", **kw)
    t.to_csv(OUT / "group_naming_tests.csv", **kw)
    print("\nwrote outputs/results/group_naming_per_seq.csv, group_naming_per_split.csv, group_naming_tests.csv")


if __name__ == "__main__":
    main()
