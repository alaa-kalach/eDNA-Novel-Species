#!/usr/bin/env python3
"""
Per-group novelty cut-offs: does the tree tell us "how different is too different" better
than taxonomy, or better than one global cut-off? (Next Directions doc, option 1; proposal s.4)

Detection signal for every method = the B2 / VSEARCH distance  d_q = 1 - best identity of the
query to the reference, with s* = species of the best hit (same tie rule as B2). Only the
CUT-OFF changes between methods: query q is called novel if  d_q >= cutoff(s*).

Methods (each gives a per-species cut-off scale c_s; score = d_q / c_s* / alpha, novel if >= 1)
  global        one cut-off for everyone                        (= B2, the current pipeline)
  own           c_s = DNA distance from s to its nearest other reference species
                (B4's scaling, but on VSEARCH identity). DNA only, no tree, no taxonomy.
  tax_groups    a learned cut-off per group, hierarchy genus -> family -> order -> global
  tree_groups   same, but the groups are clades of the time tree, cut at the ages that give
                the SAME NUMBER of groups as genus / family / order in that split's reference.
                => same granularity as taxonomy, only the boundaries (divergence time) differ.
  tax_kernel    c_s = exp( weighted mean of log nearest-relative DNA distance b_k over
  tree_kernel   reference species k ),  weights exp(-D(s,k)/tau),  D = taxonomic distance
                (data/distance_matrices.npz D_tax) or patristic distance in My (time tree).
                tau = 0 -> own,  tau = inf -> global.

Learned group cut-offs (tax_groups / tree_groups), fit on the pseudo-novel queries:
  start from the global F1-optimal cut-off; going down the hierarchy, each group keeps its
  parent's cut-off unless its own queries say otherwise, in which case it moves the least
  needed to reach the fewest errors (t_hat); then shrink:
      log t_g = (n_g log t_hat + kappa log t_parent) / (n_g + kappa)
NOTHING is tuned on the test species. Every hyper-parameter (kappa, tau) is chosen per split
by leave-one-pseudo-repeat-out F1 inside the reference (5 pseudo-novel repeats, the same
pseudo splits as the baselines); the final operating point alpha is then tuned on the pooled
pseudo scores (out-of-fold scores for the group methods, so their cut-offs are not graded on
the queries they were fit on).

Outcomes per split (test set; hard = known + hard-novel, as in models/detect_features.py)
  auroc, auroc_hard, f1 (novel class), novel_recall, close_recall (novel sequences within
  3 Teleo differences of a reference sequence - the near-identical sisters that VSEARCH misses;
  the same sequences for both markers), known_acc (known sequences given their right species),
  known_fp (known sequences wrongly called novel).

Needs data/pairwise_identity_<marker>.npz  (python data/pairwise_identity.py [--marker=full])
Usage (repo root):  python models/group_cutoffs.py                 # Teleo, 30 splits
                    python models/group_cutoffs.py --marker=full   # long 12S
Output (outputs/results/):
  group_cutoffs[_full]_per_split.csv   one row per split x method
  group_cutoffs[_full]_tests.csv       paired Wilcoxon over splits + bootstrap 95% CI + Holm
  group_cutoffs[_full]_choices.csv     chosen kappa / tau, matched tree ages, per split
  group_cutoffs[_full]_calls.csv.gz    per test sequence: score, call, nearest species
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs" / "results"
sys.path.insert(0, str(ROOT / "baselines"))
sys.path.insert(0, str(ROOT / "models"))
from common import PSEUDO_REPEATS, pseudo_rng, pseudo_split, tune_threshold  # noqa: E402
from tree_distances import patristic_matrix  # noqa: E402

EPS = 1e-4                 # floor for distances / cut-offs used in log space
CLOSE_DIFFS = 3            # "close" novel = within 3 Teleo differences of a reference sequence
KAPPAS = [np.inf, 100, 30, 10, 3, 1, 0]          # simplest first (ties -> simplest)
TAUS = {"tree": [np.inf, 160, 80, 40, 20, 10, 5, 0],          # My (patristic)
        "tax": [np.inf, 1.6, 0.8, 0.4, 0.2, 0.1, 0.05, 0]}    # D_tax units (0.25 per rank)
METHODS = ["global", "own", "tax_groups", "tree_groups", "tax_kernel", "tree_kernel"]
OUTCOMES = ["auroc", "auroc_hard", "f1", "novel_recall", "close_recall", "known_acc", "known_fp"]
PRIMARY = [("tree_groups", "tax_groups"), ("tree_kernel", "tax_kernel")]
SECONDARY = [(m, "global") for m in METHODS[1:]] + [("tree_kernel", "own"), ("tree_groups", "own")]


# ------------------------------------------------------------------------------ data
class Context:
    def __init__(self, marker):
        d = pd.read_csv(DATA / "guiana_teleo_clean.csv", dtype={"Specimen identification code": str})
        d = d[d["in_tree"]].rename(columns={"Specimen identification code": "seq_id"})
        self.seqs = d.drop_duplicates("seq_id").set_index("seq_id").sort_index()
        self.seqs["teleo"] = self.seqs["teleo"].str.strip().str.upper()
        tag = "teleo" if marker == "teleo" else "full12s"
        z = np.load(DATA / f"pairwise_identity_{tag}.npz", allow_pickle=True)
        zt = np.load(DATA / "pairwise_identity_teleo.npz", allow_pickle=True)
        assert list(z["seq_id"]) == list(self.seqs.index) == list(zt["seq_id"])
        self.ID = z["ID"].astype(np.float64)
        self.n_diff_teleo = (1 - zt["ID"]) * self.seqs["teleo"].str.len().values[:, None]
        self.pos = pd.Series(np.arange(len(self.seqs)), index=self.seqs.index)
        self.sp = self.seqs["species_name"].to_numpy(dtype=object)
        tax = self.seqs.drop_duplicates("species_name").set_index("species_name")[["Genus", "Family", "Order"]]
        self.species = np.array(sorted(tax.index))
        self.tax = tax.loc[self.species]
        dm = np.load(DATA / "distance_matrices.npz", allow_pickle=True)
        o = pd.Series(np.arange(len(dm["species"])), index=dm["species"]).loc[self.species].values
        self.D_tax = dm["D_tax"][np.ix_(o, o)]
        self.D_tree = patristic_matrix(DATA / "fish_tree_full.tre", list(self.species)).astype(np.float64)
        self.sidx = pd.Series(np.arange(len(self.species)), index=self.species)

    # best hit (B2 rule: highest identity; ties -> species with most tied hits, then alphabetical)
    def best_hits(self, q_ids, r_ids):
        qi, ri = self.pos[q_ids].values, self.pos[r_ids].values
        M = self.ID[np.ix_(qi, ri)]
        best = M.max(1)
        rsp = self.sp[ri]
        near = rsp[M.argmax(1)].copy()
        tied = (M >= best[:, None] - 1e-9)
        for i in np.where(tied.sum(1) > 1)[0]:
            c = pd.Series(rsp[tied[i]]).value_counts()
            near[i] = sorted(c.index[c == c.max()])[0]
        return 1 - best, near

    def nearest_relative(self, r_ids):
        """species -> DNA distance (1 - identity) to the nearest sequence of another species."""
        ri = self.pos[r_ids].values
        D = 1 - self.ID[np.ix_(ri, ri)]
        s = self.sp[ri]
        D[s[:, None] == s[None, :]] = np.inf
        return pd.Series(D.min(1), index=s).groupby(level=0).min().clip(lower=EPS)


def batch(ctx, ref_ids, q_ids, y):
    d, near = ctx.best_hits(q_ids, ref_ids)
    return dict(d=d, s=near, y=np.asarray(y, bool), b=ctx.nearest_relative(ref_ids))


# ------------------------------------------------------------------------ hierarchies
def tree_levels(ctx, ref_species):
    """Clades of the time tree with the same number of groups as genus / family / order."""
    idx = ctx.sidx[ref_species].values
    Z = linkage(squareform(ctx.D_tree[np.ix_(idx, idx)], checks=False), "complete")
    levels, ages = [], []
    for rank in ["Genus", "Family", "Order"]:
        n = ctx.tax.loc[ref_species, rank].nunique()
        lab = fcluster(Z, n, "maxclust")
        levels.append(pd.Series([f"{rank[0]}{v}" for v in lab], index=ref_species))
        # crown-age cut used: half the largest within-clade patristic distance
        ages.append(max(ctx.D_tree[np.ix_(idx[lab == v], idx[lab == v])].max() for v in np.unique(lab)) / 2)
    return levels, ages


def tax_levels(ctx, ref_species):
    return [ctx.tax.loc[ref_species, r] for r in ["Genus", "Family", "Order"]]


# ------------------------------------------------------------------- group cut-offs
def move_least(d, y, t_parent):
    """Cut-off with the fewest errors (novel if d >= t), closest (log scale) to t_parent."""
    u = np.unique(d)
    cands = np.r_[u[0] * 0.5 if u[0] > 0 else EPS / 2, (u[1:] + u[:-1]) / 2, u[-1] * 1.0001 + 1e-6, t_parent]
    err = (y[:, None] & (d[:, None] < cands[None])).sum(0) + (~y[:, None] & (d[:, None] >= cands[None])).sum(0)
    if err[-1] == err.min():
        return t_parent
    ok = cands[err == err.min()]
    return ok[np.argmin(np.abs(np.log(np.maximum(ok, EPS)) - np.log(max(t_parent, EPS))))]


def fit_groups(d, s, y, levels, kappa):
    """levels: [fine, ..., coarse] Series species -> group. Returns Series species -> cut-off."""
    t0 = max(tune_threshold(d, y)[0], EPS)
    species = levels[0].index
    t = pd.Series(t0, index=species)
    if kappa == np.inf:
        return t
    for lev in levels[::-1]:                                   # coarse -> fine
        g_of_q = lev.reindex(s).values
        new = t.copy()
        for g, members in lev.groupby(lev).groups.items():
            tp = float(np.exp(np.log(t.loc[members]).mean()))  # parent value (nested -> all equal)
            m = g_of_q == g
            n = int(m.sum())
            if n == 0:
                continue
            th = max(move_least(d[m], y[m], tp), EPS)
            new.loc[members] = np.exp((n * np.log(th) + kappa * np.log(tp)) / (n + kappa))
        t = new
    return t


# ----------------------------------------------------------------- kernel scaling
def kernel_scale(ctx, b, s_query, D, tau):
    """c_s for the queries' nearest species from the batch's nearest-relative distances b."""
    if tau == np.inf:
        return np.ones(len(s_query))
    if tau == 0:
        return b.reindex(s_query).values
    ks = ctx.sidx[b.index].values
    qs = ctx.sidx[s_query].values
    W = np.exp(-D[np.ix_(qs, ks)] / tau)
    return np.exp(W @ np.log(b.values) / W.sum(1))


# ----------------------------------------------------------------- one split
def f1_of(score, y):
    call = score >= 1
    return 2 * (call & y).sum() / max(call.sum() + y.sum(), 1)


def run_split(ctx, ref, test, sid):
    ref_ids = ref.index.values
    ref_species = np.array(sorted(ref.species_name.unique()))
    pseudo = []
    for r in range(PSEUDO_REPEATS):
        pref, pq = pseudo_split(ref, pseudo_rng(sid, r))
        pseudo.append(batch(ctx, pref.index.values, pq.index.values, pq.is_novel.values))
    y_test = (test.role == "test_novel").values
    tb = batch(ctx, ref_ids, test.index.values, y_test)

    lv_tax = tax_levels(ctx, ref_species)
    lv_tree, ages = tree_levels(ctx, ref_species)
    choice = dict(split=sid, tree_age_genus=ages[0], tree_age_family=ages[1], tree_age_order=ages[2])
    scores = {}

    def pool(bs, key):
        return np.concatenate([b_[key] for b_ in bs])

    # -- scale-type methods: global, own, kernels (alpha tuned on the pooled pseudo scores)
    def scale_method(name, D, grid):
        def raw(bs, tau):
            return np.concatenate([b_["d"] / np.maximum(kernel_scale(ctx, b_["b"], b_["s"], D, tau), EPS)
                                   for b_ in bs])
        best, best_f1 = grid[0], -1
        if len(grid) > 1:
            for tau in grid:
                f1s = []
                for h in range(PSEUDO_REPEATS):
                    tr = [p for i, p in enumerate(pseudo) if i != h]
                    a, _ = tune_threshold(raw(tr, tau), pool(tr, "y"))
                    f1s.append(f1_of(raw([pseudo[h]], tau) / a, pseudo[h]["y"]))
                if np.mean(f1s) > best_f1 + 1e-12:
                    best, best_f1 = tau, np.mean(f1s)
            choice[f"{name}_tau"] = best
        a, _ = tune_threshold(raw(pseudo, best), pool(pseudo, "y"))
        scores[name] = raw([tb], best) / a

    scale_method("global", None, [np.inf])
    scale_method("own", None, [0])
    scale_method("tax_kernel", ctx.D_tax, TAUS["tax"])
    scale_method("tree_kernel", ctx.D_tree, TAUS["tree"])

    # -- learned group cut-offs
    def group_method(name, levels):
        def oof(kappa):                       # out-of-fold scores + held-out F1 per repeat
            sc, f1s = [], []
            for h in range(PSEUDO_REPEATS):
                tr = [p for i, p in enumerate(pseudo) if i != h]
                t = fit_groups(pool(tr, "d"), pool(tr, "s"), pool(tr, "y"), levels, kappa)
                ins = pool(tr, "d") / t.reindex(pool(tr, "s")).values
                a, _ = tune_threshold(ins, pool(tr, "y"))
                o = pseudo[h]["d"] / t.reindex(pseudo[h]["s"]).values
                sc.append(o)
                f1s.append(f1_of(o / a, pseudo[h]["y"]))
            return np.concatenate(sc), np.mean(f1s)
        best, best_f1, best_sc = None, -1, None
        for kappa in KAPPAS:
            sc, f1 = oof(kappa)
            if f1 > best_f1 + 1e-12:
                best, best_f1, best_sc = kappa, f1, sc
        choice[f"{name}_kappa"] = best
        t = fit_groups(pool(pseudo, "d"), pool(pseudo, "s"), pool(pseudo, "y"), levels, best)
        a, _ = tune_threshold(best_sc, pool(pseudo, "y"))
        scores[name] = tb["d"] / t.reindex(tb["s"]).values / a

    group_method("tax_groups", lv_tax)
    group_method("tree_groups", lv_tree)

    # -- outcomes
    known = ~y_test
    hard = known | (test.difficulty == "hard").to_numpy(dtype=bool, na_value=False)
    ti = ctx.pos[test.index.values].values
    ri = ctx.pos[ref_ids].values
    close = y_test & (ctx.n_diff_teleo[np.ix_(ti, ri)].min(1) <= CLOSE_DIFFS + 1e-6)
    true_sp = test.species_name.to_numpy(dtype=object)
    rows, calls = [], []
    for m in METHODS:
        sc = scores[m]
        call = sc >= 1
        rows.append(dict(split=sid, method=m, n_test=len(sc), n_novel=int(y_test.sum()), n_close=int(close.sum()),
                         auroc=roc_auc_score(y_test, sc), auroc_hard=roc_auc_score(y_test[hard], sc[hard]),
                         f1=f1_of(sc, y_test), novel_recall=call[y_test].mean(),
                         close_recall=call[close].mean() if close.any() else np.nan,
                         known_acc=(~call[known] & (tb["s"][known] == true_sp[known])).mean(),
                         known_fp=call[known].mean()))
        calls.append(pd.DataFrame({"seq_id": test.index.values, "split_id": sid, "method": m,
                                   "score": np.round(sc, 5), "called_novel": call.astype(int),
                                   "nearest_species": tb["s"], "is_novel": y_test.astype(int),
                                   "close": close.astype(int)}))
    return rows, choice, calls


# ------------------------------------------------------------------- statistics
def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def compare(per, pairs, family):
    rng = np.random.default_rng(0)
    out = []
    for a, b in pairs:
        A, B = per[per.method == a].set_index("split"), per[per.method == b].set_index("split")
        for o in OUTCOMES:
            d = (A[o] - B[o]).dropna()
            boots = [rng.choice(d.values, len(d)).mean() for _ in range(5000)]
            p = wilcoxon(d.values).pvalue if (d != 0).sum() > 1 else 1.0
            out.append(dict(family=family, a=a, b=b, outcome=o, mean_a=A[o].mean(), mean_b=B[o].mean(),
                            diff=d.mean(), ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5),
                            a_better=int((d > 0).sum()), b_better=int((d < 0).sum()), n=len(d), wilcoxon_p=p))
    t = pd.DataFrame(out)
    t["holm_p"] = holm(t.wilcoxon_p.values)
    return t


def main():
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    marker = "teleo" if opts.get("marker", "teleo") == "teleo" else "full"
    suffix = "" if marker == "teleo" else "_full"
    ctx = Context(marker)
    splits = pd.concat([pd.read_csv(DATA / f, dtype={"seq_id": str, "difficulty": str})
                        for f in ["splits.csv", "splits_extra.csv"]])
    rows, choices, calls = [], [], []
    for sid in sorted(splits.split_id.unique()):
        s = splits[splits.split_id == sid].set_index("seq_id")
        m = ctx.seqs[["species_name", "Genus", "teleo"]].join(s[["role", "difficulty"]], how="inner")
        ref, test = m[m.role == "reference"].copy(), m[m.role != "reference"].copy()
        r, c, cl = run_split(ctx, ref, test, sid)
        rows += r
        choices.append(c)
        calls += cl
        line = " | ".join(f"{x['method']} F1 {x['f1']:.3f} close {x['close_recall']:.2f}" for x in r)
        print(f"[{marker}] split {sid:2d} | {line}", flush=True)

    per = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    per.round(5).to_csv(OUT / f"group_cutoffs{suffix}_per_split.csv", index=False)
    pd.DataFrame(choices).round(3).to_csv(OUT / f"group_cutoffs{suffix}_choices.csv", index=False)
    pd.concat(calls).to_csv(OUT / f"group_cutoffs{suffix}_calls.csv.gz", index=False)
    tests = pd.concat([compare(per, PRIMARY, "primary"), compare(per, SECONDARY, "secondary")])
    tests.to_csv(OUT / f"group_cutoffs{suffix}_tests.csv", index=False)

    print(f"\n[{marker}] mean over {per.split.nunique()} splits")
    print(per.groupby("method")[OUTCOMES].mean().loc[METHODS].round(4).to_string())
    print("\nprimary comparisons (Holm within the primary family)")
    t = tests[tests.family == "primary"]
    print(t[["a", "b", "outcome", "diff", "ci_lo", "ci_hi", "wilcoxon_p", "holm_p"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
