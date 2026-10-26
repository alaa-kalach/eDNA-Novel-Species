"""
Detection with tree features (the proposal's fallback idea), with a taxonomy control.

Idea from the proposal: a query whose closest known species has many close relatives
should look "novel" even at a small distance; one with no close relatives needs a big
distance. So the closest known species' neighbourhood is added to the novelty score.

For each query, find its closest reference sequence (raw k-mers, same as B4) and its
species s*. Features:
  base      : log distance to the closest reference sequence (raw k-mers)
  + tree    : log tree distance from s* to its nearest reference relative,
              number of reference species within 30 My and within 60 My of s*
              (30 / 60 My ~ the median genus / family split ages in this data)
  + taxonomy: number of reference species in s*'s genus, in s*'s family,
              and whether s* has a congener in the reference
A logistic regression (standardised features) gives P(novel). It is trained on
pseudo-novel splits built only from the reference (5 repeats, same rules as B2/B4),
so the true novel species are never seen. Same for the "novel" cut-off (max F1).

Outcomes per split: AUROC, AUROC on the hard cases only (known + novel with a congener),
novel F1 at the tuned cut-off. Paired Wilcoxon over the 30 splits.

Usage (repo root):  python models/detect_features.py [--marker=full]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baselines"))
from common import PSEUDO_REPEATS, pseudo_rng, pseudo_split, tune_threshold  # noqa: E402

DATA = ROOT / "data"
MARKER = "full" if "--marker=full" in sys.argv[1:] else "teleo"
SETS = {"base": ["base"],
        "base+tree": ["base", "tree_nn", "tree_n30", "tree_n60"],
        "base+taxonomy": ["base", "tax_genus", "tax_family", "tax_congener"]}


def dist(A, B):
    return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T, 0))


class Context:
    def __init__(self):
        clean = pd.read_csv(DATA / "guiana_teleo_clean.csv")
        clean["seq_id"] = clean["Specimen identification code"].astype(str)
        self.clean = clean[clean.in_tree].drop_duplicates("seq_id").set_index("seq_id")
        self.info = self.clean.drop_duplicates("species_name").set_index("species_name")
        k = np.load(DATA / ("kmer_features_full12S.npz" if MARKER == "full" else "kmer_features_k3to6.npz"),
                    allow_pickle=True)
        self.X = pd.DataFrame(k["X"], index=k["seq_id"].astype(str))
        tz = np.load(DATA / "global_tree_distances.npz")
        self.row = {s: i for i, s in enumerate(tz["species"])}
        self.age = tz["D"] * float(tz["scale"]) / 2

    def features(self, ref_ids, query_ids):
        ref_sp = self.clean.loc[ref_ids, "species_name"].values
        D = dist(self.X.loc[query_ids].values, self.X.loc[ref_ids].values)
        j = D.argmin(1)
        near = ref_sp[j]
        sps = np.unique(ref_sp)
        idx = [self.row[s] for s in sps]
        A = self.age[np.ix_(idx, idx)].copy()
        np.fill_diagonal(A, np.inf)
        pos = {s: i for i, s in enumerate(sps)}
        gen = np.asarray(self.info.loc[sps, "Genus"].astype(str), dtype=object)
        fam = np.asarray(self.info.loc[sps, "Family"].astype(str), dtype=object)
        p = np.array([pos[s] for s in near])
        f = pd.DataFrame({
            "base": np.log(D[np.arange(len(j)), j] + 1e-6),
            "tree_nn": np.log(A[p].min(1) + 1e-3),
            "tree_n30": (A[p] < 30).sum(1),
            "tree_n60": (A[p] < 60).sum(1),
            "tax_genus": (gen[p][:, None] == gen[None, :]).sum(1),
            "tax_family": (fam[p][:, None] == fam[None, :]).sum(1),
        })
        f["tax_congener"] = (f.tax_genus > 1).astype(float)
        return f


def main():
    ctx = Context()
    splits = pd.concat([pd.read_csv(DATA / "splits.csv"), pd.read_csv(DATA / "splits_extra.csv")])
    splits["seq_id"] = splits.seq_id.astype(str)
    rows = []
    for sid in sorted(splits.split_id.unique()):
        s = splits[splits.split_id == sid]
        ref = s[s.role == "reference"].copy()
        ref = ref.assign(species_name=ctx.clean.loc[ref.seq_id, "species_name"].values,
                         teleo=ctx.clean.loc[ref.seq_id, "teleo"].values).set_index("seq_id")
        test = s[s.role != "reference"]
        # training data: pseudo-novel splits inside the reference
        F, Y = [], []
        for r in range(PSEUDO_REPEATS):
            pref, pq = pseudo_split(ref, pseudo_rng(sid, r))
            F.append(ctx.features(pref.index.values, pq.index.values))
            Y.append(pq["is_novel"].values)
        F, Y = pd.concat(F, ignore_index=True), np.concatenate(Y)
        Ft = ctx.features(ref.index.values, test.seq_id.values)
        y = (test.role == "test_novel").values
        hard = (test.role == "test_known").values | (test.difficulty == "hard").values
        out = {"split": sid}
        for name, cols in SETS.items():
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(F[cols], Y)
            t, _ = tune_threshold(clf.predict_proba(F[cols])[:, 1], Y)
            p = clf.predict_proba(Ft[cols])[:, 1]
            call = p >= t
            out[f"{name}|auroc"] = roc_auc_score(y, p)
            out[f"{name}|auroc_hard"] = roc_auc_score(y[hard], p[hard])
            out[f"{name}|f1"] = 2 * (call & y).sum() / (call.sum() + y.sum())
        rows.append(out)
        print(f"split {sid:2d} | " + " | ".join(f"{n} AUROC {out[n + '|auroc']:.3f}" for n in SETS), flush=True)
    df = pd.DataFrame(rows).set_index("split")
    df.to_csv(ROOT / "outputs" / "results" / f"detect_features{'_full' if MARKER == 'full' else ''}.csv")

    def compare(a, b):
        print(f"\n{a}  vs  {b}")
        for o in ["auroc", "auroc_hard", "f1"]:
            x, z = df[f"{a}|{o}"], df[f"{b}|{o}"]
            d = x - z
            p = wilcoxon(x, z).pvalue if (d != 0).sum() > 1 else 1.0
            print(f"  {o:10s} {x.mean():.3f} vs {z.mean():.3f} | diff {d.mean():+.4f} "
                  f"| better in {(d > 0).sum()}/{len(d)} | p = {p:.3f}")

    print(f"\nmarker: {MARKER} | {len(df)} splits")
    compare("base+tree", "base")
    compare("base+taxonomy", "base")
    compare("base+tree", "base+taxonomy")


if __name__ == "__main__":
    main()
