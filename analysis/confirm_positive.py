#!/usr/bin/env python3
"""
Confirm the four positive results before adding anything new.

Every comparison is paired over the 30 splits (Wilcoxon signed-rank + bootstrap 95% CI of the
mean difference, 5000 resamples, seed 0), as in the rest of the repo. diff = a - b.

DECISION RULE, FIXED BEFORE RUNNING
  A claim is CONFIRMED if every one of its PRIMARY tests favours "a" with a 95% CI that excludes 0
  and a Holm-adjusted p < 0.05 (Holm over that claim's primary tests). Secondary tests are Holm-
  adjusted within the claim and reported, but do not decide.

CLAIM 1  Long 12S beats Teleo  (a = long 12S, b = Teleo; same test sequences, checked per split)
  Detection, VSEARCH / B2 rule ("global" in group_cutoffs[_full]_per_split.csv)
    PRIMARY   close_recall (novel within 3 Teleo differences of a reference; same sequences both markers)
    secondary auroc, auroc_hard, f1, novel_recall, known_acc, known_fp (lower = better)
  Placement of novel species, "alignment nearest" (no tree; phylo_placement[_full]_per_split.csv)
    PRIMARY   family
    secondary family_easy, genus_hard, tree_error (lower = better); same four for "phylo"

CLAIM 2  Reference-calibrated grouping beats standard fixed thresholds
  (a = gap_align_vs, b = fixed<id>_<linkage>_vs; cluster_scores_fixed_per_split.csv)
    PRIMARY   ARI vs fixed 97% single linkage
    secondary ARI, AMI, bcubed_f1 vs 100 / 99 / 98 / 97 % x single / average;
              richness error |groups - true species among the grouped sequences| (lower = better)
    sensitivity (printed, no test): fixed 96% and 95% = best fixed cut-off in hindsight

CLAIM 3  Tree supervision helps alignment-free (k-mer) methods  (collects tests ALREADY on disk)
  raw+treecap90 vs raw (w = 1), Teleo and long 12S (placement_kmer_tree[_full]_tests.csv):
    family, family_easy, tree_error
  isolation trend, raw+treecap90 vs raw (when_tree_helps[_full]_tests.csv):
    sequence quartiles Q4 - Q1 and tree bins >= 60 My - < 30 My, family and tree_error
  control: raw+treecap90 vs raw+taxonomy (same outcomes) -> is it the TREE, or any hierarchy?
  NOTE: these were secondary analyses and have been seen. Teleo = discovery, long 12S = replication on
  the second marker (same species and splits). Report them as such; PRIMARY here = the long-12S
  replication of the Q4 - Q1 family trend.

CLAIM 4  Open-set identification beats forced assignment  (a = VSEARCH / B2, b = B1 forced)
    PRIMARY   balanced_correct = mean(known species named correctly, novel sequences called novel)
    secondary novel_misnamed (lower = better), known_acc, overall_correct

Usage (repo root):  python analysis/confirm_positive.py
Needs for claim 2:  python clustering/fixed_grouping.py, then score_clusters.py --out=_fixed ...
                    (claim 2 is skipped with a message if its file is missing)
Output: outputs/results/confirm_claims.csv  (one row per test), verdicts printed
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
DATA, OUTS, RES = ROOT / "data", ROOT / "outputs", ROOT / "outputs" / "results"
LOWER_BETTER = {"known_fp", "tree_error", "tree_error_easy", "richness_error", "novel_misnamed"}
rng = np.random.default_rng(0)


def paired(a, b, outcome):
    """a, b: Series indexed by split. Returns the test dict (diff = a - b)."""
    d = (a - b).dropna()
    boots = [rng.choice(d.values, len(d)).mean() for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = wilcoxon(d.values).pvalue if (d != 0).any() else 1.0
    sign = -1 if outcome in LOWER_BETTER else 1
    return dict(mean_a=a.loc[d.index].mean(), mean_b=b.loc[d.index].mean(), diff=d.mean(), ci_lo=lo, ci_hi=hi,
                a_better=int((sign * d > 0).sum()), b_better=int((sign * d < 0).sum()), n=len(d), wilcoxon_p=p,
                favours_a=bool(sign * lo > 0 if sign > 0 else sign * hi > 0))


def holm(p):
    p = np.asarray(p, float)
    order, m = np.argsort(p), len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def add(rows, claim, primary, a, b, outcome, marker, res):
    rows.append(dict(claim=claim, primary=primary, a=a, b=b, outcome=outcome, marker=marker, **res))


# ---------------------------------------------------------------- claim 1
def claim1(rows):
    gt = pd.read_csv(RES / "group_cutoffs_per_split.csv")
    gf = pd.read_csv(RES / "group_cutoffs_full_per_split.csv")
    gt, gf = (g[g.method == "global"].set_index("split") for g in (gt, gf))
    for c in ["n_test", "n_novel", "n_close"]:
        if not gt[c].equals(gf.loc[gt.index, c]):
            raise SystemExit(f"claim 1: {c} differs between markers - not the same test sequences")
    for o in ["close_recall", "auroc", "auroc_hard", "f1", "novel_recall", "known_acc", "known_fp"]:
        add(rows, 1, o == "close_recall", "long 12S", "Teleo", o, "VSEARCH detection",
            paired(gf[o], gt[o], o))
    pt = pd.read_csv(RES / "phylo_placement_per_split.csv")
    pf = pd.read_csv(RES / "phylo_placement_full_per_split.csv")
    for meth in ["alignment nearest", "phylo"]:
        A = pf[pf.method == meth].set_index("split")
        B = pt[pt.method == meth].set_index("split")
        for o in ["family", "family_easy", "genus_hard", "tree_error"]:
            add(rows, 1, meth == "alignment nearest" and o == "family", "long 12S", "Teleo", o,
                f"placement: {meth}", paired(A[o], B[o], o))


# ---------------------------------------------------------------- claim 2
def load_clusters(name):
    fs = [RES / f"clusters_{name}.csv", RES / f"clusters_{name}_extra.csv"]
    return pd.concat([pd.read_csv(f, dtype={"seq_id": str}) for f in fs if f.exists()])


def richness(name, species):
    df = load_clusters(name)
    out = {}
    for k, g in df.groupby("split_id"):
        lab = g.cluster_id.values.copy()
        n_noise = int((lab < 0).sum())                       # noise = groups of one
        n_groups = len(np.unique(lab[lab >= 0])) + n_noise
        out[k] = abs(n_groups - species.loc[g.seq_id].nunique())
    return pd.Series(out)


def claim2(rows, species):
    f = RES / "cluster_scores_fixed_per_split.csv"
    if not f.exists():
        print("claim 2 skipped: run clustering/fixed_grouping.py and score_clusters.py --out=_fixed first")
        return
    per = pd.read_csv(f)
    A = per[per.method == "gap_align_vs"].set_index("split")
    rich_a = richness("gap_align_vs", species)
    for ident in [97, 98, 99, 100]:
        for link in ["single", "average"]:
            name = f"fixed{ident}_{link}_vs"
            B = per[per.method == name].set_index("split")
            if B.empty:
                print(f"claim 2: {name} missing from {f.name}")
                continue
            for o in ["ARI", "AMI", "bcubed_f1"]:
                add(rows, 2, (ident, link, o) == (97, "single", "ARI"), "gap_align_vs", name, o, "Teleo",
                    paired(A[o], B[o], o))
            add(rows, 2, False, "gap_align_vs", name, "richness_error", "Teleo",
                paired(rich_a, richness(name, species), "richness_error"))
    sens = per[per.method.isin(["gap_align_vs", "fixed96_single_vs", "fixed95_single_vs",
                                "fixed96_average_vs", "fixed95_average_vs"])]
    print("\nclaim 2 sensitivity (hindsight fixed cut-offs, NOT a result) - mean over splits:")
    print(sens.groupby("method")[["ARI", "AMI", "bcubed_f1"]].mean().round(3).to_string())


# ---------------------------------------------------------------- claim 3
def claim3(rows):
    for marker, tag in [("Teleo", ""), ("long 12S", "_full")]:
        t = pd.read_csv(OUTS / f"placement_kmer_tree{tag}_tests.csv")
        t = t[(t.a == "raw+treecap90") & (t.w.astype(float) == 1.0)]
        for b in ["raw", "raw+taxonomy"]:
            for o in ["family", "family_easy", "tree_error"]:
                r = t[(t.b == b) & (t.outcome == o)]
                if len(r) != 1:
                    print(f"claim 3: no unique row raw+treecap90 vs {b} {o} ({marker})")
                    continue
                r = r.iloc[0]
                sign = -1 if o in LOWER_BETTER else 1
                add(rows, 3, False, "raw+treecap90", b, o, f"{marker}, overall",
                    dict(mean_a=r.mean_a, mean_b=r.mean_b, diff=r["diff"], ci_lo=r.ci_lo, ci_hi=r.ci_hi,
                         a_better=r.a_better, b_better=r.b_better, n=r.n, wilcoxon_p=r.wilcoxon_p,
                         favours_a=bool(sign * r.ci_lo > 0 if sign > 0 else sign * r.ci_hi > 0)))
        w = pd.read_csv(RES / f"when_tree_helps{tag}_tests.csv")
        w = w[(w.a == "raw+treecap90") & w.bin.astype(str).str.startswith("trend")]
        for b in ["raw", "raw+taxonomy"]:
            for iso in ["seq", "tree"]:
                for o in ["family", "tree_error"]:
                    r = w[(w.b == b) & (w.isolation == iso) & (w.outcome == o)]
                    if len(r) != 1:
                        continue
                    r = r.iloc[0]
                    # trend favours "a" if the gain grows with isolation: >0 for family, <0 for tree_error
                    sign = -1 if o in LOWER_BETTER else 1
                    prim = marker == "long 12S" and b == "raw" and iso == "seq" and o == "family"
                    add(rows, 3, prim, "raw+treecap90", b, o, f"{marker}, trend {r.bin.split(': ')[1]}",
                        dict(mean_a=np.nan, mean_b=np.nan, diff=r["diff"], ci_lo=r.ci_lo, ci_hi=r.ci_hi,
                             a_better=np.nan, b_better=np.nan, n=r.n, wilcoxon_p=r.wilcoxon_p,
                             favours_a=bool(sign * r.ci_lo > 0 if sign > 0 else sign * r.ci_hi > 0)))


# ---------------------------------------------------------------- claim 4
def claim4(rows, species):
    def load(stem):
        return pd.concat([pd.read_csv(OUTS / f"{stem}{t}.csv", dtype={"seq_id": str}) for t in ["", "_extra"]])
    splits = pd.concat([pd.read_csv(DATA / "splits.csv"), pd.read_csv(DATA / "splits_extra.csv")])
    splits["seq_id"] = splits.seq_id.astype(str)
    role = splits.set_index(["split_id", "seq_id"]).role
    per = {}
    for name, stem in [("VSEARCH", "results_vsearch"), ("B1 forced", "results_forced")]:
        r = load(stem)
        r["role"] = role.loc[list(zip(r.split_id, r.seq_id))].values
        r["truth"] = species.loc[r.seq_id].values
        out = {}
        for k, g in r.groupby("split_id"):
            nov, kn = g[g.role == "test_novel"], g[g.role == "test_known"]
            known_acc = (kn.predicted_species == kn.truth).mean()
            novel_ok = (nov.predicted_species == "novel").mean()
            correct = np.where(g.role == "test_novel", g.predicted_species == "novel",
                               g.predicted_species == g.truth).mean()
            out[k] = dict(balanced_correct=(known_acc + novel_ok) / 2, novel_misnamed=1 - novel_ok,
                          known_acc=known_acc, overall_correct=correct)
        per[name] = pd.DataFrame(out).T
    for o in ["balanced_correct", "novel_misnamed", "known_acc", "overall_correct"]:
        add(rows, 4, o == "balanced_correct", "VSEARCH", "B1 forced", o, "Teleo",
            paired(per["VSEARCH"][o], per["B1 forced"][o], o))


def main():
    clean = pd.read_csv(DATA / "guiana_teleo_clean.csv")
    clean["seq_id"] = clean["Specimen identification code"].astype(str)
    species = clean.drop_duplicates("seq_id").set_index("seq_id").species_name
    rows = []
    claim1(rows)
    claim2(rows, species)
    claim3(rows)
    claim4(rows, species)
    t = pd.DataFrame(rows)
    t["holm_p"] = np.nan
    for (c, prim), g in t.groupby(["claim", "primary"]):
        t.loc[g.index, "holm_p"] = holm(g.wilcoxon_p.values)
    t.to_csv(RES / "confirm_claims.csv", index=False)

    names = {1: "Long 12S beats Teleo", 2: "Calibrated grouping beats fixed thresholds",
             3: "Tree supervision helps k-mer methods", 4: "Open-set beats forced assignment"}
    for c, g in t.groupby("claim"):
        p = g[g.primary]
        ok = len(p) > 0 and bool((p.favours_a & (p.holm_p < 0.05)).all())
        print(f"\nCLAIM {c}: {names[c]} -> {'CONFIRMED' if ok else 'NOT CONFIRMED'}")
        show = g[["primary", "a", "b", "marker", "outcome", "mean_a", "mean_b", "diff", "ci_lo", "ci_hi",
                  "holm_p"]]
        print(show.round(4).to_string(index=False))
    print("\nwrote outputs/results/confirm_claims.csv")


if __name__ == "__main__":
    main()
