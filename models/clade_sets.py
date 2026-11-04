#!/usr/bin/env python3
"""
Clade-shaped conformal prediction sets for NEW species: "this new fish is in one of these families",
with a guaranteed coverage. Does the TREE give smaller sets than taxonomy or plain sequence
similarity, at the same guaranteed coverage?

Every method answers the same question - which FAMILY a truly novel sequence belongs to - with a SET
of families, grown step by step from the same starting point, the B2 / VSEARCH nearest reference
species h (data/pairwise_identity_<marker>.npz, B2 tie rule). Only the way the set grows differs:
  tree    S(t) = families of reference species r with tree distance D_tree(h, r) <= t   (My, patristic,
                 Fish Tree) -> the set is the families inside a clade around h
  taxonomy S(0) = family of h;  S(1) = families of reference species in h's order;  S(2) = all
  top-k   S(k) = families of the k reference species most similar to the query (best identity per
                 species; ties -> h first, then alphabetical)  -> sequence similarity only, no hierarchy
Nonconformity score of a query = the smallest step whose set contains its true family.
Set reported = the step at the conformal quantile of the calibration scores.

RULES, FIXED BEFORE RUNNING
  Queries: truly novel test sequences (role == test_novel), local reference only, both markers.
    Excluded (counted): novel sequences whose FAMILY has no species in the reference (no set of
    reference families can contain it); the same rule for calibration queries.
  Calibration: the pseudo-novel queries of the 5 pseudo-splits B2 already uses (baselines/common.py
    pseudo_split, same seeds), each scored against its pseudo-reference; each species counts once per
    repeat (weights), repeats averaged. No test sequence is used.
  Conformal: Mondrian by divergence, the SAME bins for every method: 3 bins of the query's best
    identity, cut at the weighted tertiles of the calibration identities (per split). Per bin,
    q = smallest score v with sum w[score <= v] >= (1 - alpha) (W_bin + 1); none -> the full set.
    alpha = 0.05, 0.10, 0.20.  PRIMARY alpha = 0.10.
  Outcomes per split (novel sequences): coverage (true family in the set), size (number of families
    in the set; lower = better), single (share of sets with exactly one family), n_ref_families.
  PRIMARY: at alpha = 0.10, size(tree) - size(top-k) and size(tree) - size(taxonomy), paired Wilcoxon
    over the 30 splits + bootstrap 95% CI, Holm over these two (per marker). The tree is better only
    if BOTH CIs are below 0 with Holm p < 0.05 on Teleo; long 12S = replication.
    Validity (each method): mean coverage over splits, not rejected if the 95% CI upper bound >= 1 - alpha.
  Secondary: every alpha; single-family rate; the same comparisons without Mondrian bins (one bin).

Usage (repo root):  python models/clade_sets.py               # Teleo
                    python models/clade_sets.py --marker=full
Output (outputs/results/):
  clade_sets[_full]_per_split.csv   split x method x alpha x binning
  clade_sets[_full]_validity.csv    mean coverage + CI per method x alpha x binning
  clade_sets[_full]_tests.csv       paired size comparisons
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs" / "results"
sys.path.insert(0, str(ROOT / "baselines"))
sys.path.insert(0, str(ROOT / "models"))
from common import PSEUDO_REPEATS, pseudo_rng, pseudo_split  # noqa: E402
from group_cutoffs import Context, holm  # noqa: E402

METHODS = ["tree", "taxonomy", "top-k"]
ALPHAS = [0.05, 0.10, 0.20]
PRIMARY_ALPHA = 0.10
BINNINGS = {"mondrian": 3, "single": 1}


# ------------------------------------------------------------- per-query nested sets
class Ref:
    """Everything about one reference set that the nested sets need."""
    def __init__(self, ctx, ref, fam_of_sp, order_of_sp):
        self.ids = ref.index.values
        self.species = np.array(sorted(ref.species_name.unique()))
        self.fam = fam_of_sp.reindex(self.species).to_numpy(object)
        self.order = order_of_sp.reindex(self.species).to_numpy(object)
        si = ctx.sidx[self.species].values
        self.D = ctx.D_tree[np.ix_(si, si)]
        self.col_sp = pd.Series(np.arange(len(self.species)), index=self.species) \
            .reindex(ref.species_name.values).values           # ref sequence -> species column
        self.ri = ctx.pos[self.ids].values
        self.families = set(self.fam)


def species_identity(ctx, R, q_ids):
    """(n_query, n_species) best identity of each query to each reference species."""
    M = ctx.ID[np.ix_(ctx.pos[q_ids].values, R.ri)]
    out = np.full((len(q_ids), len(R.species)), -1.0)
    np.maximum.at(out.T, R.col_sp, M.T)
    return out


def nested(ctx, R, q_ids, true_fam):
    """Scores (per method) and what is needed to build the sets at any threshold."""
    _, hit_sp = ctx.best_hits(q_ids, R.ids)
    h = pd.Series(np.arange(len(R.species)), index=R.species).reindex(hit_sp).values
    S = species_identity(ctx, R, q_ids)
    best_id = S.max(1)
    n = len(q_ids)
    sc = {m: np.full(n, np.inf) for m in METHODS}
    rank = np.empty_like(S, dtype=int)
    for i in range(n):
        key = np.lexsort((R.species, np.arange(len(R.species)) != h[i], -S[i]))  # identity desc, h first, alpha
        rank[i, key] = np.arange(1, len(R.species) + 1)
        is_f = R.fam == true_fam[i]
        if not is_f.any():
            continue
        sc["tree"][i] = R.D[h[i], is_f].min()
        sc["taxonomy"][i] = 0 if R.fam[h[i]] == true_fam[i] else (
            1 if (R.order[is_f] == R.order[h[i]]).any() else 2)
        sc["top-k"][i] = rank[i, is_f].min()
    return dict(score=sc, h=h, rank=rank, best_id=best_id)


def set_families(R, nest, i, method, q):
    if not np.isfinite(q):
        return R.families
    if method == "tree":
        m = R.D[nest["h"][i]] <= q + 1e-9
    elif method == "taxonomy":
        m = (np.zeros(len(R.species), bool) if q < 0 else
             (R.fam == R.fam[nest["h"][i]]) if q < 1 else
             (R.order == R.order[nest["h"][i]]) if q < 2 else np.ones(len(R.species), bool))
    else:
        m = nest["rank"][i] <= q
    return set(R.fam[m])


# ------------------------------------------------------------- conformal
def wquantile(scores, w, alpha):
    o = np.argsort(scores, kind="stable")
    s, cw = scores[o], np.cumsum(w[o])
    need = (1 - alpha) * (w.sum() + 1)
    k = np.searchsorted(cw, need - 1e-12, side="left")
    return s[k] if k < len(s) else np.inf


def bin_edges(ids, w, nbins):
    if nbins == 1:
        return np.array([])
    o = np.argsort(ids)
    cw = np.cumsum(w[o]) / w.sum()
    return np.array([ids[o][np.searchsorted(cw, b / nbins)] for b in range(1, nbins)])


# ------------------------------------------------------------- one split
def run_split(ctx, ref, test, sid, fam_of_sp, order_of_sp):
    R = Ref(ctx, ref, fam_of_sp, order_of_sp)
    cal_s, cal_w, cal_id, n_cal_excl = {m: [] for m in METHODS}, [], [], 0
    for r in range(PSEUDO_REPEATS):
        pref, pq = pseudo_split(ref, pseudo_rng(sid, r))
        pq = pq[pq.is_novel]
        if len(pq) == 0:
            continue
        PR = Ref(ctx, pref, fam_of_sp, order_of_sp)
        tf = fam_of_sp.reindex(pq.species_name).to_numpy(object)
        nest = nested(ctx, PR, pq.index.values, tf)
        keep = np.isfinite(nest["score"]["tree"])
        n_cal_excl += int((~keep).sum())
        cnt = pq.species_name[keep].map(pq.species_name[keep].value_counts()).to_numpy(float)
        for m in METHODS:
            cal_s[m].append(nest["score"][m][keep])
        cal_w.append(1 / cnt / PSEUDO_REPEATS)
        cal_id.append(nest["best_id"][keep])
    cal_s = {m: np.concatenate(v) for m, v in cal_s.items()}
    cal_w, cal_id = np.concatenate(cal_w), np.concatenate(cal_id)

    nov = test[test.role == "test_novel"]
    tf = fam_of_sp.reindex(nov.species_name).to_numpy(object)
    nest = nested(ctx, R, nov.index.values, tf)
    keep = np.isfinite(nest["score"]["tree"])
    rows = []
    for bname, nb in BINNINGS.items():
        edges = bin_edges(cal_id, cal_w, nb)
        cb, tb = np.searchsorted(edges, cal_id, "right"), np.searchsorted(edges, nest["best_id"], "right")
        for a in ALPHAS:
            for m in METHODS:
                q = {b: wquantile(cal_s[m][cb == b], cal_w[cb == b], a) for b in range(nb)}
                cov, size, single = [], [], []
                for i in np.where(keep)[0]:
                    fams = set_families(R, nest, i, m, q[tb[i]])
                    cov.append(tf[i] in fams)
                    size.append(len(fams))
                    single.append(len(fams) == 1)
                rows.append(dict(split=sid, binning=bname, alpha=a, method=m, n_novel=len(nov),
                                 n_scored=int(keep.sum()), n_excluded_new_family=int((~keep).sum()),
                                 n_calib=len(cal_w), n_calib_excluded=n_cal_excl,
                                 n_ref_families=len(R.families), coverage=np.mean(cov), size=np.mean(size),
                                 single=np.mean(single)))
    return rows


# ------------------------------------------------------------- main
def main():
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    marker = "teleo" if opts.get("marker", "teleo") == "teleo" else "full"
    suffix = "" if marker == "teleo" else "_full"
    ctx = Context(marker)
    info = ctx.seqs.drop_duplicates("species_name").set_index("species_name")
    fam_of_sp, order_of_sp = info.Family, info.Order
    splits = pd.concat([pd.read_csv(DATA / f, dtype={"seq_id": str, "difficulty": str})
                        for f in ["splits.csv", "splits_extra.csv"]])
    rows = []
    for sid in sorted(splits.split_id.unique()):
        s = splits[splits.split_id == sid].set_index("seq_id")
        m = ctx.seqs[["species_name", "Genus", "teleo"]].join(s[["role", "difficulty"]], how="inner")
        ref, test = m[m.role == "reference"].copy(), m[m.role != "reference"].copy()
        r = run_split(ctx, ref, test, sid, fam_of_sp, order_of_sp)
        rows += r
        p = {x["method"]: x for x in r if x["binning"] == "mondrian" and x["alpha"] == PRIMARY_ALPHA}
        print(f"[{marker}] split {sid:2d} | a=0.10 size/coverage: " + " | ".join(
            f"{k} {v['size']:.2f}/{v['coverage']:.2f}" for k, v in p.items())
              + f" | excluded (new family) {p['tree']['n_excluded_new_family']}", flush=True)
    per = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    per.round(5).to_csv(OUT / f"clade_sets{suffix}_per_split.csv", index=False)

    rng = np.random.default_rng(0)
    val = []
    for (b, a, m), g in per.groupby(["binning", "alpha", "method"]):
        x = g.coverage.values
        boots = [rng.choice(x, len(x)).mean() for _ in range(5000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        val.append(dict(binning=b, alpha=a, method=m, mean_coverage=x.mean(), ci_lo=lo, ci_hi=hi,
                        valid=bool(hi >= 1 - a), mean_size=g["size"].mean(), single=g.single.mean()))
    val = pd.DataFrame(val)
    val.to_csv(OUT / f"clade_sets{suffix}_validity.csv", index=False)

    tests = []
    for (b, a), g in per.groupby(["binning", "alpha"]):
        T = g[g.method == "tree"].set_index("split")
        for other in ["top-k", "taxonomy"]:
            O = g[g.method == other].set_index("split")
            for o in ["size", "single", "coverage"]:
                d = (T[o] - O[o]).dropna()
                boots = [rng.choice(d.values, len(d)).mean() for _ in range(5000)]
                p = wilcoxon(d.values).pvalue if (d != 0).sum() > 1 else 1.0
                tests.append(dict(marker=marker, binning=b, alpha=a, a="tree", b=other, outcome=o,
                                  primary=bool(b == "mondrian" and a == PRIMARY_ALPHA and o == "size"),
                                  mean_tree=T[o].mean(), mean_other=O[o].mean(), diff=d.mean(),
                                  ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5),
                                  tree_better=int(((d < 0) if o == "size" else (d > 0)).sum()),
                                  other_better=int(((d > 0) if o == "size" else (d < 0)).sum()),
                                  n=len(d), wilcoxon_p=p))
    tests = pd.DataFrame(tests)
    tests["holm_p"] = np.nan
    pr = tests.primary
    tests.loc[pr, "holm_p"] = holm(tests.loc[pr, "wilcoxon_p"].values)
    tests.to_csv(OUT / f"clade_sets{suffix}_tests.csv", index=False)

    print(f"\n[{marker}] VALIDITY + SIZE (mean over splits; size = number of families in the set)")
    print(val.round(4).to_string(index=False))
    print(f"\n[{marker}] tree vs others (diff = tree - other; size: negative = tree smaller = better)")
    print(tests[["binning", "alpha", "b", "outcome", "mean_tree", "mean_other", "diff", "ci_lo", "ci_hi",
                 "wilcoxon_p", "holm_p"]].round(4).to_string(index=False))
    t = tests[tests.primary]
    vt = val[(val.binning == "mondrian") & (val.alpha == PRIMARY_ALPHA)]
    ok = bool(((t.ci_hi < 0) & (t.holm_p < 0.05)).all()) and bool(vt.valid.all())
    print(f"\n[{marker}] PRIMARY (Mondrian, alpha 0.10): tree vs top-k size {t.iloc[0]['diff']:+.2f} "
          f"(Holm p {t.iloc[0]['holm_p']:.2g}), tree vs taxonomy size {t.iloc[1]['diff']:+.2f} "
          f"(Holm p {t.iloc[1]['holm_p']:.2g}); coverage valid for all: {bool(vt.valid.all())} "
          f"-> {'TREE BETTER' if ok else 'TREE NOT BETTER'}")


if __name__ == "__main__":
    main()
