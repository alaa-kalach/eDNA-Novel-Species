#!/usr/bin/env python3
"""
Position-weighted mismatches (Next Directions doc, "structure inside the sequence", step 1):
does weighting each difference by how unusual it is at that position catch the near-identical
sister species that a plain identity cut-off misses, WITHOUT more false alarms on known species?

The ALIGNMENT is VSEARCH's own pairwise alignment for every method (data/pair_alignments.py);
only the way differences are counted changes:
    distance(q, r) = sum over differences of w[position] / alignment length
All methods then use the B2 decision rule: distance to the nearest reference sequence, one
cut-off per split tuned for F1 on the pseudo-novel splits inside the reference.

  vsearch   1 - VSEARCH identity from the full matrix (data/pairwise_identity_<marker>.npz)
            = the current pipeline.
  flat      every position weight 1  ->  identical to vsearch by construction (CONTROL: checks
            that the pair store reproduces the pipeline; any difference here is a bug).
  within    w_j = -log p_within(j): a difference at a position that rarely varies inside a
            species counts more (per-position variability profile).
  llr       w_j = log( p_between(j) / p_within(j) ), floored: a difference counts by how much
            more often that position differs between sister species than within a species.
  (weights scaled to mean 1 over positions, so a typical difference still counts ~1)

  p_within(j)  = share of within-species sequence pairs (in the reference) that differ at j,
                 every species with 2+ sequences counted once, shrunk toward the mean.
  p_between(j) = same over each reference sequence and its nearest sequence of another species
                 (the sister-level pairs that get confused), every species counted once.
Weights are re-estimated from each reference set (each pseudo-reference while tuning, the real
reference for the test), never from the test species.

The pair store keeps each sequence's 100 best VSEARCH hits; the nearest reference is searched
among those (a query with none of its 100 best hits in the reference falls back to the vsearch
distance for every method; the count is printed - expect 0).

Outcomes per split (test set), as in group_cutoffs.py plus threshold-free views of the hard cases:
  auroc, auroc_hard, auroc_close (close novel vs known), f1, novel_recall, close_recall,
  known_acc, known_fp, close_at_fp05 / close_at_fp10 (share of close novel sequences caught when
  the cut-off lets exactly 5% / 10% of known sequences be called novel).
  close = novel test sequences within 3 Teleo differences of a reference sequence.

Usage (repo root):  python models/mismatch_weights.py                 # Teleo
                    python models/mismatch_weights.py --marker=full   # long 12S
Needs data/pairwise_identity_<marker>.npz and data/pair_alignments_<marker>.npz.
Output (outputs/results/):
  mismatch_weights[_full]_per_split.csv   one row per split x method
  mismatch_weights[_full]_tests.csv       paired Wilcoxon + bootstrap 95% CI + Holm
  mismatch_weights[_full]_profile.csv     per shared position: centre base, coverage,
                                          p_within, p_between, weights (mean of 30 references)
  mismatch_weights[_full]_calls.csv.gz    per test sequence: distance, call, nearest species
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs" / "results"
sys.path.insert(0, str(ROOT / "baselines"))
from common import PSEUDO_REPEATS, pseudo_rng, pseudo_split, tune_threshold  # noqa: E402

CLOSE_DIFFS = 3
PRIOR = 1.0            # shrinkage of p_within / p_between toward their mean (in "species")
LLR_FLOOR = 0.1        # llr weights below this share of their mean are raised to it
METHODS = ["vsearch", "flat", "within", "llr"]
WEIGHTED = ["flat", "within", "llr"]
OUTCOMES = ["auroc", "auroc_hard", "auroc_close", "f1", "novel_recall", "close_recall",
            "known_acc", "known_fp", "close_at_fp05", "close_at_fp10"]
PRIMARY = [("within", "flat"), ("llr", "flat")]
SECONDARY = [("within", "vsearch"), ("llr", "vsearch"), ("flat", "vsearch"), ("llr", "within")]


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
        a = np.load(DATA / f"pair_alignments_{tag}.npz", allow_pickle=True)
        ids = list(self.seqs.index)
        assert list(z["seq_id"]) == ids == list(zt["seq_id"]) == list(a["seq_id"]), "sequence order differs"
        self.n = len(ids)
        self.ID = z["ID"].astype(np.float64)
        self.n_diff_teleo = (1 - zt["ID"]) * self.seqs["teleo"].str.len().values[:, None]
        self.L = int(a["L"])
        self.centre_seq = str(a["centre_seq"])
        self.cov = a["cov"]
        self.pq, self.pt = a["pair_q"].astype(np.int64), a["pair_t"].astype(np.int64)
        self.alen = a["aln_len"].astype(np.float64)
        ptr, dpos = a["diff_ptr"], a["diff_pos"]
        self.M = csr_matrix((np.ones(len(dpos)), dpos, ptr), shape=(len(self.pq), self.L))
        self.key = pd.Series(np.arange(len(self.pq)), index=self.pq * self.n + self.pt)
        self.key = self.key[~self.key.index.duplicated()]
        self.pos = pd.Series(np.arange(self.n), index=ids)
        self.sp = self.seqs["species_name"].to_numpy(dtype=object)
        self.fallbacks = 0

    def idx(self, ids):
        return self.pos[ids].values

    def pair_rows(self, a, b):
        """store row of each pair (a[i], b[i]) in either direction, -1 if not stored."""
        k1 = self.key.reindex(a * self.n + b).values
        k2 = self.key.reindex(b * self.n + a).values
        r = np.where(np.isnan(k1), k2, k1)
        return np.where(np.isnan(r), -1, r).astype(np.int64)


# ------------------------------------------------------------------- distances
def pick_nearest(q, d, tsp):
    """per query: min distance and its species (B2 tie rule: most tied hits, then alphabetical)."""
    df = pd.DataFrame({"q": q, "d": d, "sp": tsp})
    best = df.groupby("q").d.min()
    tied = df[df.d <= best.reindex(df.q).values + 1e-12]
    cnt = tied.groupby(["q", "sp"]).size().reset_index(name="c")
    cnt = cnt.sort_values(["q", "c", "sp"], ascending=[True, False, True]).drop_duplicates("q")
    return best, cnt.set_index("q").sp


def vsearch_nearest(ctx, qi, ri):
    D = 1 - ctx.ID[np.ix_(qi, ri)]
    rsp = ctx.sp[ri]
    best = D.min(1)
    near = rsp[D.argmin(1)].copy()
    tied = D <= best[:, None] + 1e-12
    for i in np.where(tied.sum(1) > 1)[0]:
        c = pd.Series(rsp[tied[i]]).value_counts()
        near[i] = sorted(c.index[c == c.max()])[0]
    return best, near


def shrink(num, den):
    p0 = num.sum() / max(den.sum(), 1e-12)
    return (num + PRIOR * p0) / (den + PRIOR)


def profile(ctx, rows, a, b, wt):
    """weighted share of the stored pairs that differ at each position (both sequences span it)."""
    num = np.asarray(ctx.M[rows].multiply(wt[:, None]).sum(0)).ravel()
    den = (wt[:, None] * (ctx.cov[a] & ctx.cov[b])).sum(0)
    return shrink(num, den)


def estimate_weights(ctx, ri):
    rsp = ctx.sp[ri]
    inR = np.zeros(ctx.n, bool)
    inR[ri] = True
    # within-species pairs in the store, each species counted once
    m = inR[ctx.pq] & inR[ctx.pt] & (ctx.sp[ctx.pq] == ctx.sp[ctx.pt])
    rows = np.where(m)[0]
    spw = pd.Series(ctx.sp[ctx.pq[rows]])
    wt = 1.0 / spw.map(spw.value_counts()).values
    p_within = profile(ctx, rows, ctx.pq[rows], ctx.pt[rows], wt)
    # each reference sequence and its nearest sequence of another species (VSEARCH identity)
    S = ctx.ID[np.ix_(ri, ri)].copy()
    S[rsp[:, None] == rsp[None, :]] = -np.inf
    nb = ri[S.argmax(1)]
    r2 = ctx.pair_rows(ri, nb)
    ok = r2 >= 0
    per_sp = pd.Series(rsp).value_counts()
    p_between = profile(ctx, r2[ok], ri[ok], nb[ok], 1.0 / per_sp.loc[rsp[ok]].values)

    within = -np.log(p_within)
    llr = np.log(p_between / p_within)
    llr = np.maximum(llr, LLR_FLOOR * np.mean(np.maximum(llr, 0)))
    w = {"flat": np.ones(ctx.L), "within": within / within.mean(), "llr": llr / llr.mean()}
    return w, p_within, p_between


def scores_for(ctx, ref_ids, q_ids):
    ri, qi = ctx.idx(ref_ids), ctx.idx(q_ids)
    out = {"vsearch": vsearch_nearest(ctx, qi, ri)}
    w, pw, pb = estimate_weights(ctx, ri)
    inR, inQ = np.zeros(ctx.n, bool), np.zeros(ctx.n, bool)
    inR[ri], inQ[qi] = True, True
    rows = np.where(inQ[ctx.pq] & inR[ctx.pt])[0]
    q, t = ctx.pq[rows], ctx.pt[rows]
    Msub, alen = ctx.M[rows], ctx.alen[rows]
    missing = ~np.isin(qi, q)
    ctx.fallbacks += int(missing.sum())
    for m in WEIGHTED:
        best, near = pick_nearest(q, (Msub @ w[m]) / alen, ctx.sp[t])
        d = np.array(best.reindex(qi).values, dtype=np.float64)
        s = np.array(near.reindex(qi).to_numpy(dtype=object), dtype=object)
        d[missing], s[missing] = out["vsearch"][0][missing], out["vsearch"][1][missing]
        out[m] = (d, s)
    return out, (w, pw, pb)


# ----------------------------------------------------------------- one split
def f1_of(call, y):
    return 2 * (call & y).sum() / max(call.sum() + y.sum(), 1)


def caught_at_fp(sc, known, close, fp):
    if not close.any():
        return np.nan
    return (sc[close] > np.quantile(sc[known], 1 - fp)).mean()


def run_split(ctx, ref, test, sid):
    sc_ps, y_ps = {m: [] for m in METHODS}, []
    for r in range(PSEUDO_REPEATS):
        pref, pq = pseudo_split(ref, pseudo_rng(sid, r))
        o, _ = scores_for(ctx, pref.index.values, pq.index.values)
        for m in METHODS:
            sc_ps[m].append(o[m][0])
        y_ps.append(pq.is_novel.values.astype(bool))
    y_ps = np.concatenate(y_ps)
    o, (w, pw, pb) = scores_for(ctx, ref.index.values, test.index.values)

    y = (test.role == "test_novel").values
    known = ~y
    hard = known | (test.difficulty == "hard").to_numpy(dtype=bool, na_value=False)
    ti, ri = ctx.idx(test.index.values), ctx.idx(ref.index.values)
    close = y & (ctx.n_diff_teleo[np.ix_(ti, ri)].min(1) <= CLOSE_DIFFS + 1e-6)
    true_sp = test.species_name.to_numpy(dtype=object)
    kc = known | close
    rows, calls = [], []
    for m in METHODS:
        t, _ = tune_threshold(np.concatenate(sc_ps[m]), y_ps)
        sc, near = o[m]
        call = sc >= t
        rows.append(dict(split=sid, method=m, cutoff=t, n_test=len(sc), n_novel=int(y.sum()),
                         n_close=int(close.sum()),
                         auroc=roc_auc_score(y, sc), auroc_hard=roc_auc_score(y[hard], sc[hard]),
                         auroc_close=roc_auc_score(close[kc], sc[kc]) if close.any() else np.nan,
                         f1=f1_of(call, y), novel_recall=call[y].mean(),
                         close_recall=call[close].mean() if close.any() else np.nan,
                         known_acc=(~call[known] & (near[known] == true_sp[known])).mean(),
                         known_fp=call[known].mean(),
                         close_at_fp05=caught_at_fp(sc, known, close, 0.05),
                         close_at_fp10=caught_at_fp(sc, known, close, 0.10)))
        calls.append(pd.DataFrame({"seq_id": test.index.values, "split_id": sid, "method": m,
                                   "distance": np.round(sc, 6), "called_novel": call.astype(int),
                                   "nearest_species": near, "is_novel": y.astype(int),
                                   "close": close.astype(int)}))
    prof = pd.DataFrame({"split": sid, "position": np.arange(ctx.L), "p_within": pw, "p_between": pb,
                         "w_within": w["within"], "w_llr": w["llr"], "coverage": ctx.cov[ri].mean(0)})
    return rows, calls, prof


# ------------------------------------------------------------------- statistics
def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj, run = np.empty_like(p), 0.0
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
            if len(d) < 2:
                continue
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
    print(f"[{marker}] {ctx.n} sequences, {ctx.L} shared positions, {len(ctx.pq)} stored pairs")
    splits = pd.concat([pd.read_csv(DATA / f, dtype={"seq_id": str, "difficulty": str})
                        for f in ["splits.csv", "splits_extra.csv"]])
    rows, calls, profs = [], [], []
    for sid in sorted(splits.split_id.unique()):
        s = splits[splits.split_id == sid].set_index("seq_id")
        m = ctx.seqs[["species_name", "teleo"]].join(s[["role", "difficulty"]], how="inner")
        ref, test = m[m.role == "reference"].copy(), m[m.role != "reference"].copy()
        r, c, p = run_split(ctx, ref, test, sid)
        rows += r
        calls += c
        profs.append(p)
        line = " | ".join(f"{x['method']} F1 {x['f1']:.3f} close {x['close_recall']:.2f} "
                          f"fp {x['known_fp']:.3f}" for x in r)
        print(f"[{marker}] split {sid:2d} | {line}", flush=True)

    per = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    per.round(5).to_csv(OUT / f"mismatch_weights{suffix}_per_split.csv", index=False)
    pd.concat(calls).to_csv(OUT / f"mismatch_weights{suffix}_calls.csv.gz", index=False)
    pr = pd.concat(profs).groupby("position").mean().drop(columns="split")
    pr.insert(0, "centre_base", list(ctx.centre_seq[:ctx.L]))
    pr.round(5).to_csv(OUT / f"mismatch_weights{suffix}_profile.csv")
    tests = pd.concat([compare(per, PRIMARY, "primary"), compare(per, SECONDARY, "secondary")])
    tests.to_csv(OUT / f"mismatch_weights{suffix}_tests.csv", index=False)

    print(f"\n[{marker}] queries with no stored pair in the reference (fell back to vsearch): {ctx.fallbacks}")
    print(f"\n[{marker}] mean over {per.split.nunique()} splits  (CHECK: flat must match vsearch)")
    print(per.groupby("method")[OUTCOMES].mean().loc[METHODS].round(4).to_string())
    print("\nprimary comparisons: weighting vs flat, same alignment (Holm within the primary family)")
    t = tests[tests.family == "primary"]
    print(t[["a", "b", "outcome", "diff", "ci_lo", "ci_hi", "wilcoxon_p", "holm_p"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
