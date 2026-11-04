#!/usr/bin/env python3
"""
A conformal "is it new?" layer for ANY eDNA classifier, and a fair comparison of classifiers at
the same guaranteed false-alarm rate (Teleo, 30 splits).

Each tool gives every query a novelty score (higher = more likely a new species) and a nearest
known species:
  vsearch     1 - identity to the closest reference sequence (B2 score, data/pairwise_identity_teleo.npz)
  sintax      1 - SINTAX bootstrap confidence of the top species (Edgar 2016 algorithm, re-implemented:
              8-mers, 32 query words drawn with replacement, 100 bootstraps, top hit = reference
              sequence sharing most drawn words, ties -> species with most tied sequences, then alphabetical)
  rdp         1 - RDP naive Bayes bootstrap confidence (baselines/b3_naive_bayes.py, global + reference)
  kmer        B4 score: nearest-sequence k-mer distance / that species' nearest-relative distance
  epa         tree placement (models/phylo_placement.py): pendant + path to the nearest reference leaf
  treekmer    B4-style score in the space of the GLOBAL pretrained tree-clade model
              (models/pretrained_treeclade.npz; never trained on French Guiana -> no leakage)

Calibration data (no test sequence is ever used): the 5 pseudo-splits B2/B3/B4 already use
(baselines/common.py pseudo_split, same seeds) built from the split's reference only. Their
pseudo-KNOWN queries (one held-out haplotype of a species that stays in the pseudo-reference) are
the calibration points; their pseudo-NOVEL queries are used only to tune the F1 cut-off.

Rules compared per tool:
  standard    the published default "unknown" rule where one exists: vsearch identity < 97%,
              sintax confidence < 0.8, rdp bootstrap < 0.8
  f1          cut-off maximising F1 on the pseudo-splits (as B2/B3/B4 do)
  conf05, conf10      conformal, alpha = 0.05 / 0.10, weights: each species counts once per
                      pseudo-split, the 5 pseudo-splits averaged (total weight = number of species)
  pool05, pool10      same, pseudo-splits NOT averaged (5x more calibration weight; less cautious;
                      the 5 repeats share species, so validity is checked empirically, not guaranteed)
  mond05, mond10      Mondrian conformal: a separate cut-off for "crowded" queries (another species
                      within 3 Teleo differences of the nearest species, seen at query time) and for
                      the rest; pooled weights as pool* (with averaged weights the groups are too
                      small to call anything at 5%: decided after a 1-split code check, before the run)
  Reported groups (known_fp only): rare = nearest species has 1 distinct sequence in the reference,
                      crowded as above.
  conformal p-value p = (1 + sum w [c >= s]) / (1 + sum w); call new if p <= alpha.

RULES, FIXED BEFORE RUNNING
  Outcomes per split: known_fp (known test sequences called new), novel_recall, close_recall
  (new sequences within 3 Teleo differences of a reference sequence), f1, known_acc, auroc.
  PRIMARY (Holm across the 6 tools within each test):
    (1) validity  conf05: mean known_fp over 30 splits, bootstrap 95% CI lower bound <= 0.05
                  (every tool must pass)
    (2) control   conf05 known_fp < f1 known_fp (paired Wilcoxon, 95% CI of the difference < 0)
  SECONDARY: novel_recall and close_recall at conf05 / conf10 of every tool vs vsearch (Holm);
    standard rules: share of splits with known_fp > 0.05; Mondrian: known_fp in the crowded and
    non-crowded groups, pool vs mond (does the per-group cut-off keep crowded known fish under alpha?);
    pool vs conf: validity and recall gained.

Usage (repo root):  python models/conformal_any.py                  # all 30 splits
                    python models/conformal_any.py --splits=1       # quick check
                    python models/conformal_any.py --skip=epa,rdp   # leave slow tools out
Tools: mafft, raxml-ng, epa-ng on the PATH for epa (as models/phylo_placement.py).
Output (outputs/results/): conformal_any_per_split.csv, conformal_any_groups.csv,
                           conformal_any_validity.csv, conformal_any_tests.csv
"""
import sys
import tempfile
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs" / "results"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines"))
sys.path.insert(0, str(ROOT / "models"))
from common import PSEUDO_REPEATS, pseudo_rng, pseudo_split, tune_threshold  # noqa: E402

TOOLS = ["vsearch", "sintax", "rdp", "kmer", "epa", "treekmer"]
CLOSE = 3
OUTCOMES = ["known_fp", "novel_recall", "close_recall", "f1", "known_acc"]


# ------------------------------------------------------------------ data
def load():
    d = pd.read_csv(DATA / "guiana_teleo_clean.csv", dtype={"Specimen identification code": str})
    d = d[d["in_tree"]].rename(columns={"Specimen identification code": "seq_id"})
    seqs = d.drop_duplicates("seq_id").set_index("seq_id").sort_index()
    seqs["teleo"] = seqs["teleo"].str.strip().str.upper()
    z = np.load(DATA / "pairwise_identity_teleo.npz", allow_pickle=True)
    assert list(z["seq_id"]) == list(seqs.index)
    ID = z["ID"].astype(np.float64)
    ndiff = (1 - ID) * seqs["teleo"].str.len().values[:, None]
    pos = pd.Series(np.arange(len(seqs)), index=seqs.index)
    splits = pd.concat([pd.read_csv(DATA / f, dtype={"seq_id": str, "difficulty": str})
                        for f in ["splits.csv", "splits_extra.csv"]])
    return seqs, ID, ndiff, pos, splits


# ------------------------------------------------------------------ tools
def nearest_by_matrix(S, rsp, larger_is_closer=True):
    """Best reference per query from a similarity (or distance) matrix; ties -> species with most
    tied references, then alphabetical (B2 rule)."""
    best = S.max(1) if larger_is_closer else S.min(1)
    tied = (S >= best[:, None] - 1e-9) if larger_is_closer else (S <= best[:, None] + 1e-9)
    near = np.empty(len(S), dtype=object)
    for i in range(len(S)):
        c = pd.Series(rsp[tied[i]]).value_counts()
        near[i] = sorted(c.index[c == c.max()])[0]
    return best, near


class Vsearch:
    def __init__(self, env):
        self.ID, self.pos = env["ID"], env["pos"]

    def score(self, ref, q):
        rsp = ref.species_name.to_numpy(dtype=object)
        M = self.ID[np.ix_(self.pos[q.index].values, self.pos[ref.index].values)]
        best, near = nearest_by_matrix(M, rsp)
        return 1 - best, near


K8 = 8
_CODE = np.full(256, -1, dtype=np.int64)
for _i, _c in enumerate("ACGT"):
    _CODE[ord(_c)] = _i


def words8(seq):
    a = _CODE[np.frombuffer(seq.upper().encode(), dtype=np.uint8)]
    if len(a) < K8:
        return np.empty(0, dtype=np.int64)
    win = np.lib.stride_tricks.sliding_window_view(a, K8)
    ok = (win >= 0).all(axis=1)
    return np.unique(win[ok] @ (4 ** np.arange(K8 - 1, -1, -1)))


class Sintax:
    """SINTAX (Edgar 2016): 100 bootstraps of 32 query words; top hit = reference with most shared words."""
    def __init__(self, env):
        self.W = pd.Series([words8(s) for s in env["seqs"].teleo], index=env["seqs"].index)

    def score(self, ref, q, seed):
        rng = np.random.default_rng(seed)
        rw = list(self.W[ref.index])
        rows = np.repeat(np.arange(len(rw)), [len(w) for w in rw])
        R = sparse.csr_matrix((np.ones(len(rows)), (rows, np.concatenate(rw))), shape=(len(rw), 4 ** K8)).T.tocsr()
        rsp = ref.species_name.to_numpy(dtype=object)
        qw = list(self.W[q.index])
        votes = [dict() for _ in qw]
        for _ in range(100):
            cols, rr = [], []
            for i, w in enumerate(qw):
                if len(w):
                    cols.append(w[rng.integers(len(w), size=32)]); rr.append(np.full(32, i))
            if not cols:
                break
            Qb = sparse.csr_matrix((np.ones(sum(map(len, cols))), (np.concatenate(rr), np.concatenate(cols))),
                                   shape=(len(qw), 4 ** K8))
            Sh = (Qb @ R).toarray()
            _, top = nearest_by_matrix(Sh, rsp)
            for i, s in enumerate(top):
                if len(qw[i]):
                    votes[i][s] = votes[i].get(s, 0) + 1
        near = np.empty(len(qw), dtype=object)
        conf = np.zeros(len(qw))
        for i, v in enumerate(votes):
            if v:
                m = max(v.values())
                near[i] = sorted(k for k in v if v[k] == m)[0]
                conf[i] = m / 100
            else:
                near[i] = rsp[0]
        return 1 - conf, near


class Rdp:
    def __init__(self, env):
        import b3_naive_bayes as b3
        from models.pretrain import load_global
        self.b3 = b3
        with redirect_stdout(StringIO()):
            g = load_global()
        self.glob = b3.GlobalCounts([b3.words(s) for s in g["teleo"]], g["species_name"].values)
        self.W = pd.Series([b3.words(s) for s in env["seqs"].teleo], index=env["seqs"].index)

    def score(self, ref, q, seed):
        nb = self.b3.NaiveBayes(list(self.W[ref.index]), ref.species_name.values, self.glob)
        return nb.score(list(self.W[q.index]), np.random.default_rng(seed))


class Kmer:
    """B4 rule in a given feature space (raw k-mers, or a model embedding)."""
    def __init__(self, X):
        self.X = X

    def score(self, ref, q):
        Xr, Xq = self.X.loc[ref.index].values, self.X.loc[q.index].values
        rsp = ref.species_name.to_numpy(dtype=object)
        D = cdist(Xr, Xr)
        D[rsp[:, None] == rsp[None, :]] = np.inf
        r = pd.Series(D.min(1), index=rsp).groupby(level=0).min().clip(lower=1e-9)
        Dq = cdist(Xq, Xr)
        best, near = nearest_by_matrix(-Dq, rsp)
        return -best / r.loc[near].values, near


class Epa:
    def __init__(self, env):
        import phylo_placement as pp
        self.pp = pp
        self.tree = pp.load_tree()
        self.clean = env["seqs"][["species_name"]]
        self.lower = env["seqs"].teleo.str.lower()

    def score(self, ref, q):
        with tempfile.TemporaryDirectory() as tmp:
            df = self.pp.place_split(ref.index.values, q.index.values, self.clean, self.lower, self.tree, Path(tmp))
        df = df.set_index("seq_id").loc[q.index]
        return df.tree_dist.values, df.phylo_species.to_numpy(dtype=object)


def make_tools(env, skip):
    tools = {}
    if "vsearch" not in skip:
        tools["vsearch"] = Vsearch(env)
    if "sintax" not in skip:
        tools["sintax"] = Sintax(env)
    if "rdp" not in skip:
        tools["rdp"] = Rdp(env)
    k = np.load(DATA / "kmer_features_k3to6.npz", allow_pickle=True)
    X = pd.DataFrame(k["X"], index=k["seq_id"].astype(str))
    X = X[~X.index.duplicated()].loc[env["seqs"].index]
    if "kmer" not in skip:
        tools["kmer"] = Kmer(X)
    if "treekmer" not in skip:
        from models.pretrain import load_encoder
        enc = load_encoder(ROOT / "models" / "pretrained_treeclade.npz")
        tools["treekmer"] = Kmer(pd.DataFrame(enc.embed(X.values), index=X.index))
    if "epa" not in skip:
        tools["epa"] = Epa(env)
    return tools


def run_tool(name, tool, ref, q, seed):
    return tool.score(ref, q, seed) if name in ("sintax", "rdp") else tool.score(ref, q)


# ------------------------------------------------------------------ conformal
def conformal_p(cal, w, s):
    if len(cal) == 0:
        return np.ones(len(s))
    o = np.argsort(cal, kind="stable")
    c, ww = cal[o], w[o]
    tail = np.r_[np.cumsum(ww[::-1])[::-1], 0.0]
    i = np.searchsorted(c, s - 1e-12, side="left")
    return (1 + tail[i]) / (1 + w.sum())


def groups(near, ref, ndiff, pos):
    """0..3 = rare (nearest species has 1 distinct sequence in ref) x crowded (another species within
    3 Teleo differences of the nearest species in ref)."""
    rsp = ref.species_name.to_numpy(dtype=object)
    n_hap = ref.groupby("species_name").teleo.nunique()
    ri = pos[ref.index].values
    Dn = ndiff[np.ix_(ri, ri)].copy()
    Dn[rsp[:, None] == rsp[None, :]] = np.inf
    crowd = pd.Series(Dn.min(1), index=rsp).groupby(level=0).min() <= CLOSE + 1e-6
    rare = (n_hap.reindex(near).values == 1)
    cr = crowd.reindex(near).fillna(False).values.astype(bool)
    return rare.astype(int) * 2 + cr.astype(int)


def evaluate(call, y, close, near, true_sp, score):
    known = ~y
    return dict(known_fp=call[known].mean(), novel_recall=call[y].mean(),
                close_recall=call[close].mean() if close.any() else np.nan,
                f1=2 * (call & y).sum() / max(call.sum() + y.sum(), 1),
                known_acc=(~call[known] & (near[known] == true_sp[known])).mean(),
                auroc=roc_auc_score(y, score))


# ------------------------------------------------------------------ main
def main():
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    skip = set(opts.get("skip", "").split(",")) - {""}
    seqs, ID, ndiff, pos, splits = load()
    env = dict(seqs=seqs, ID=ID, pos=pos)
    t0 = time.time()
    tools = make_tools(env, skip)
    print(f"tools: {list(tools)} ({time.time() - t0:.0f}s to load)", flush=True)
    sids = sorted(splits.split_id.unique())
    if "splits" in opts:
        sids = [int(x) for x in opts["splits"].split(",")]
    rows, grows = [], []
    for sid in sids:
        t1 = time.time()
        s = splits[splits.split_id == sid].set_index("seq_id")
        m = seqs[["species_name", "teleo"]].join(s[["role", "difficulty"]], how="inner")
        ref, test = m[m.role == "reference"], m[m.role != "reference"]
        y = (test.role == "test_novel").to_numpy(bool)
        true_sp = test.species_name.to_numpy(dtype=object)
        close = y & (ndiff[np.ix_(pos[test.index].values, pos[ref.index].values)].min(1) <= CLOSE + 1e-6)
        pseudo = [pseudo_split(ref, pseudo_rng(sid, r)) for r in range(PSEUDO_REPEATS)]
        for name, tool in tools.items():
            score, near = run_tool(name, tool, ref, test, 900_000 + 100 * sid)
            g_test = groups(near, ref, ndiff, pos)
            cal_s, cal_w, cal_g, cal_wp, f1_s, f1_y = [], [], [], [], [], []
            for r, (pref, pq) in enumerate(pseudo):
                ps, pn = run_tool(name, tool, pref, pq, 900_000 + 100 * sid + r + 1)
                f1_s.append(ps); f1_y.append(pq.is_novel.values)
                kn = ~pq.is_novel.values
                cnt = pq.species_name[kn].map(pq.species_name[kn].value_counts()).to_numpy(float)
                cal_s.append(ps[kn]); cal_w.append(1 / cnt / PSEUDO_REPEATS); cal_wp.append(1 / cnt)
                cal_g.append(groups(pn[kn], pref, ndiff, pos))
            cal_s, cal_w, cal_g, cal_wp = map(np.concatenate, (cal_s, cal_w, cal_g, cal_wp))
            t_f1, _ = tune_threshold(np.concatenate(f1_s), np.concatenate(f1_y))
            rules = {"f1": score >= t_f1}
            if name == "vsearch":
                rules["standard"] = score > 1 - 0.97 + 1e-9
            if name in ("sintax", "rdp"):
                rules["standard"] = score > 1 - 0.8 + 1e-9
            for a in (0.05, 0.10):
                tag = f"{int(a * 100):02d}"
                rules[f"conf{tag}"] = conformal_p(cal_s, cal_w, score) <= a
                rules[f"pool{tag}"] = conformal_p(cal_s, cal_wp, score) <= a
                mond = np.zeros(len(score), bool)
                for c in (0, 1):
                    sel, cs = (g_test % 2) == c, (cal_g % 2) == c
                    if sel.any():
                        mond[sel] = conformal_p(cal_s[cs], cal_wp[cs], score[sel]) <= a
                rules[f"mond{tag}"] = mond
            for rule, call in rules.items():
                rows.append(dict(split=sid, tool=name, rule=rule, n_calib_species=round(cal_w.sum(), 2),
                                 **evaluate(call, y, close, near, true_sp, score)))
                for g in range(4):
                    sel = (g_test == g) & ~y
                    grows.append(dict(split=sid, tool=name, rule=rule, group=g, rare=g >= 2, crowded=g % 2 == 1,
                                      n_known=int(sel.sum()), known_fp=call[sel].mean() if sel.any() else np.nan))
        print(f"split {sid:2d} done ({time.time() - t1:.0f}s)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    per = pd.DataFrame(rows)
    per.round(5).to_csv(OUT / "conformal_any_per_split.csv", index=False)
    grp = pd.DataFrame(grows)
    grp.to_csv(OUT / "conformal_any_groups.csv", index=False)
    if len(sids) < 2:
        print(per.round(3).to_string(index=False))
        return
    rng = np.random.default_rng(0)

    def boot(x):
        x = np.asarray(x, float); x = x[~np.isnan(x)]
        b = [rng.choice(x, len(x)).mean() for _ in range(5000)]
        return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)

    val = []
    for (tool, rule), g in per.groupby(["tool", "rule"]):
        mu, lo, hi = boot(g.known_fp)
        a = 0.10 if rule.endswith("10") else 0.05
        val.append(dict(tool=tool, rule=rule, target=a if rule[-2:] in ("05", "10") else np.nan,
                        known_fp=mu, ci_lo=lo, ci_hi=hi, splits_over_5pct=int((g.known_fp > 0.05).sum()),
                        splits_over_10pct=int((g.known_fp > 0.10).sum()),
                        novel_recall=g.novel_recall.mean(), close_recall=g.close_recall.mean(),
                        f1=g.f1.mean(), known_acc=g.known_acc.mean(), auroc=g.auroc.mean()))
    val = pd.DataFrame(val)
    val.to_csv(OUT / "conformal_any_validity.csv", index=False)

    def holm(p):
        p = np.asarray(p, float); o = np.argsort(p); adj = np.empty(len(p)); run_ = 0.0
        for r, i in enumerate(o):
            run_ = max(run_, (len(p) - r) * p[i]); adj[i] = min(run_, 1.0)
        return adj

    def paired(a_tool, a_rule, b_tool, b_rule, outcome, family):
        A = per[(per.tool == a_tool) & (per.rule == a_rule)].set_index("split")[outcome]
        B = per[(per.tool == b_tool) & (per.rule == b_rule)].set_index("split")[outcome]
        d = (A - B).dropna()
        b = [rng.choice(d.values, len(d)).mean() for _ in range(5000)]
        p = wilcoxon(d.values).pvalue if (d != 0).sum() > 1 else 1.0
        return dict(family=family, a=f"{a_tool}:{a_rule}", b=f"{b_tool}:{b_rule}", outcome=outcome,
                    mean_a=A.mean(), mean_b=B.mean(), diff=d.mean(), ci_lo=np.percentile(b, 2.5),
                    ci_hi=np.percentile(b, 97.5), n=len(d), wilcoxon_p=p)

    tests = []
    for t in tools:
        tests.append(paired(t, "conf05", t, "f1", "known_fp", "primary: control"))
    for t in tools:
        if t != "vsearch":
            for rule in ("conf05", "conf10"):
                for o in ("novel_recall", "close_recall"):
                    tests.append(paired(t, rule, "vsearch", rule, o, f"secondary: {rule} recall vs vsearch"))
    for t in tools:
        for a in ("05", "10"):
            for o in ("known_fp", "novel_recall"):
                tests.append(paired(t, f"pool{a}", t, f"conf{a}", o, "secondary: pooled vs averaged"))
    tests = pd.DataFrame(tests)
    tests["holm_p"] = np.nan
    for fam, g in tests.groupby("family"):
        tests.loc[g.index, "holm_p"] = holm(g.wilcoxon_p.values)
    tests.to_csv(OUT / "conformal_any_tests.csv", index=False)

    gsum = grp.groupby(["tool", "rule", "rare", "crowded"]).agg(known_fp=("known_fp", "mean"),
                                                               n_known=("n_known", "sum")).reset_index()
    pd.set_option("display.width", 200)
    print("\nPER TOOL AND RULE (mean over splits; known_fp = known fish wrongly called new)")
    show = val[["tool", "rule", "known_fp", "ci_lo", "ci_hi", "splits_over_5pct", "novel_recall", "close_recall",
                "f1", "known_acc", "auroc"]]
    print(show.round(3).to_string(index=False))
    print("\nPRIMARY 1 (validity, conf05: CI lower bound <= 0.05 for every tool)")
    v = val[val.rule == "conf05"]
    for _, r in v.iterrows():
        print(f"  {r.tool:9s} known_fp {r.known_fp:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}] -> "
              f"{'held' if r.ci_lo <= 0.05 else 'REJECTED'}")
    print("\nPRIMARY 2 (control: conf05 known_fp < f1 known_fp, Holm across tools)")
    print(tests[tests.family == "primary: control"][["a", "b", "mean_a", "mean_b", "diff", "ci_lo", "ci_hi",
                                                     "holm_p"]].round(4).to_string(index=False))
    print("\nSECONDARY tests")
    print(tests[tests.family != "primary: control"][["family", "a", "b", "outcome", "mean_a", "mean_b", "diff",
                                                     "holm_p"]].round(4).to_string(index=False))
    print("\nPER GROUP known_fp (rare = nearest species has 1 known variant; crowded = close sister in library)")
    sel = gsum[gsum.rule.isin(["f1", "conf05", "pool05", "mond05", "conf10", "pool10", "mond10"])]
    print(sel.pivot_table(index=["tool", "rare", "crowded"], columns="rule", values="known_fp").round(3).to_string())
    print("\nwrote outputs/results/conformal_any_per_split.csv, _groups.csv, _validity.csv, _tests.csv")


if __name__ == "__main__":
    main()
