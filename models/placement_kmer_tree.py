#!/usr/bin/env python3
"""
Raw k-mers + encoder (combined distance) for PLACING new species.

Question: does adding a fine-tuned encoder's distances on top of raw k-mer distances
improve the genus/family guess for novel sequences, especially those with no
same-genus relative in the reference? Detection ("novel or not") is NOT changed:
it stays with raw k-mers / B4. This script only scores placement.

Method (per split, per training seed)
  centre of a reference species = mean of its reference sequences, in each space
  D_raw   : Euclidean distance test sequence -> species centre, raw k-mer space
            (data/kmer_features_k3to6.npz, same vectors as B4)
  D_model : same, in the fine-tuned embedding space
            (outputs/embeddings/embeddings_ft_<target>[_s<N>]_split<k>.npz)
  scale   : each distance is divided by the median distance between reference
            species centres in its own space (reference only, never test data)
  D = D_raw / median_raw + w * D_model / median_model,     w = 1 (fixed in advance)
  placement = nearest reference species under D -> its genus and family.

Methods compared (all with the same code and the same centres):
  raw                        w = 0 (raw k-mer centres alone)
  raw+<target>               w = 1, target = tree2g_w10m0.4 (original tree), labels (B5),
                             taxonomy (B6), treecap120 (capped tree) and treeclade (tree clades)
  <target> alone             model distance only (as in models/placement_test.py)
  B4 nearest sequence        nearest_species from outputs/results_kmer[_extra].csv
                             (nearest reference SEQUENCE, not centre; reference row)
Sensitivity (reported, never used to choose): w = 0.25, 0.5, 2, 4.

Outcomes, truly novel sequences only (role == test_novel):
  family       family correct, all novel                         (all novel species)
  family_easy  family correct, no same-genus species in reference  <- KEY outcome
  genus_hard   genus correct, a same-genus species is in reference
  known_sp     species correct for test_known sequences (side check: does the
               combination hurt the placement of known species?)
  tree_error       split age (million years) between the guessed and the true species on
                   the fish tree, all novel (lower = better; no taxonomic ranks involved)
  tree_error_easy  same, novel species with no same-genus species in the reference
Each split's score is averaged over the seeds first; then paired Wilcoxon
signed-rank tests over the 30 splits + bootstrap 95% CI of the mean difference.

Usage (from the repo root):
    python models/placement_kmer_tree.py                      # seeds 0-4, all 30 splits
    python models/placement_kmer_tree.py --seeds=0            # one seed (quick check)
    python models/placement_kmer_tree.py --marker=full        # long 12S region (no B4 row)
Needs : data/kmer_features_k3to6.npz   (python models/kmers.py)
        data/global_tree_distances.npz (python models/tree_distances.py) for the tree error
        for --marker=full: data/kmer_features_full12S.npz (data/make_full12s.py) and the
        *_full embeddings
        outputs/embeddings/embeddings_ft_<target>[_s<N>]_split<k>.npz for every
        target, seed and split (written by models/finetune_eval.py)
        outputs/results_kmer.csv + results_kmer_extra.csv (B4)
Output: outputs/placement_kmer_tree[_full]_per_split.csv  (split, method, w, outcomes; seed-averaged)
        outputs/placement_kmer_tree[_full]_per_seed.csv   (same, before averaging over seeds)
        outputs/placement_kmer_tree[_full]_tests.csv      (paired comparisons)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT, EMB = ROOT / "data", ROOT / "outputs", ROOT / "outputs" / "embeddings"
TARGETS = {"tree": "tree2g_w10m0.4", "labels": "labels", "taxonomy": "taxonomy",
           "treecap": "treecap120", "treeclade": "treeclade"}
W_MAIN = 1.0
W_SENS = (0.25, 0.5, 2.0, 4.0)
OUTCOMES = ["family", "family_easy", "genus_hard", "known_sp", "tree_error", "tree_error_easy"]
LOWER_BETTER = {"tree_error", "tree_error_easy"}
MARKER = "full" if "--marker=full" in sys.argv[1:] else "teleo"
MK = "_full" if MARKER == "full" else ""          # long 12S files carry _full in their names


def seed_suffix(seed):
    return "" if int(seed) == 0 else f"_s{int(seed)}"


def load():
    clean = pd.read_csv(DATA / "guiana_teleo_clean.csv")
    clean["seq_id"] = clean["Specimen identification code"].astype(str)
    clean = clean[clean.in_tree].drop_duplicates("seq_id").set_index("seq_id")
    splits = pd.concat([pd.read_csv(DATA / "splits.csv"), pd.read_csv(DATA / "splits_extra.csv")])
    splits["seq_id"] = splits["seq_id"].astype(str)
    kf = DATA / ("kmer_features_full12S.npz" if MARKER == "full" else "kmer_features_k3to6.npz")
    if not kf.exists():
        sys.exit(f"missing {kf} - run: python models/kmers.py (Teleo) or data/make_full12s.py (long 12S)")
    k = np.load(kf, allow_pickle=True)
    X = pd.DataFrame(k["X"], index=k["seq_id"].astype(str))
    return clean, splits, X[~X.index.duplicated()]


def emb_path(target, seed, split_id):
    return EMB / f"embeddings_ft_{target}{MK}{seed_suffix(seed)}_split{split_id}.npz"


_TREE = {}


def split_age(pred, true):
    """Age (million years) of the split between each guessed and true species on the tree
    = half the tree distance. data/global_tree_distances.npz stores D scaled to 0-1 and the
    scale (largest distance) separately."""
    if not _TREE:
        tf = DATA / "global_tree_distances.npz"
        if not tf.exists():
            sys.exit(f"missing {tf} - run: python models/tree_distances.py")
        z = np.load(tf)
        _TREE.update(row={s: i for i, s in enumerate(z["species"])}, D=z["D"], scale=float(z["scale"]))
    i = [_TREE["row"][s] for s in pred]
    j = [_TREE["row"][s] for s in true]
    return _TREE["D"][i, j] * _TREE["scale"] / 2


def check_inputs(split_ids, seeds):
    missing = [emb_path(t, s, k) for t in TARGETS.values() for s in seeds for k in split_ids
               if not emb_path(t, s, k).exists()]
    if missing:
        sys.exit(f"{len(missing)} embedding files missing, e.g.\n  " +
                 "\n  ".join(str(p.relative_to(ROOT)) for p in missing[:5]) +
                 "\nThey are written by models/finetune_eval.py (outputs/embeddings/ is not in git).")


def centre_distances(V_ref, ref_species, V_test):
    """(species, D test->centre, median distance between reference centres)."""
    sp = np.unique(ref_species)
    C = np.stack([V_ref[ref_species == s].mean(0) for s in sp])

    def dist(A, B):
        d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
        return np.sqrt(np.maximum(d2, 0))

    CC = dist(C, C)
    med = np.median(CC[np.triu_indices(len(sp), k=1)])
    return sp, dist(V_test, C), med


def outcomes(pred, test, clean, info):
    """Per-split outcome dict for predicted species `pred` of the test rows."""
    true = clean.loc[test.seq_id, "species_name"].values
    nov = (test.role == "test_novel").values
    easy = nov & (test.difficulty == "easy").values
    hard = nov & (test.difficulty == "hard").values
    fam = info.loc[pred, "Family"].values == info.loc[true, "Family"].values
    gen = info.loc[pred, "Genus"].values == info.loc[true, "Genus"].values
    age = split_age(pred, true)
    mean = lambda m: m.mean() if len(m) else np.nan  # noqa: E731
    return dict(family=mean(fam[nov]), family_easy=mean(fam[easy]),
                genus_hard=mean(gen[hard]), known_sp=mean((pred == true)[~nov]),
                tree_error=mean(age[nov]), tree_error_easy=mean(age[easy]))


def run(clean, splits, X, seeds):
    info = clean.drop_duplicates("species_name").set_index("species_name")
    b4 = None
    if MARKER == "teleo":                          # B4 was run on Teleo only
        b4 = pd.concat([pd.read_csv(OUT / "results_kmer.csv", dtype={"seq_id": str}),
                        pd.read_csv(OUT / "results_kmer_extra.csv", dtype={"seq_id": str})])
        if "nearest_species" not in b4:
            sys.exit("results_kmer*.csv has no nearest_species column - rerun baselines/b4_kmer.py")
        b4 = b4.set_index(["split_id", "seq_id"])["nearest_species"]
    rows = []
    for k in sorted(splits.split_id.unique()):
        s = splits[splits.split_id == k]
        ref, test = s[s.role == "reference"], s[s.role != "reference"].reset_index(drop=True)
        rsp = clean.loc[ref.seq_id, "species_name"].values
        sp, D_raw, m_raw = centre_distances(X.loc[ref.seq_id].values, rsp, X.loc[test.seq_id].values)
        Rn = D_raw / m_raw
        base = dict(split=k, seed=-1)
        rows.append({**base, "method": "raw", "w": 0.0, **outcomes(sp[Rn.argmin(1)], test, clean, info)})
        if b4 is not None:
            pred_b4 = b4.loc[[(k, q) for q in test.seq_id]].values
            rows.append({**base, "method": "B4 nearest sequence", "w": 0.0,
                         **outcomes(pred_b4, test, clean, info)})
        for name, target in TARGETS.items():
            for seed in seeds:
                z = np.load(emb_path(target, seed, k), allow_pickle=True)
                V = pd.DataFrame(z["emb"], index=z["seq_id"].astype(str))
                sp_m, D_m, m_m = centre_distances(V.loc[ref.seq_id].values, rsp, V.loc[test.seq_id].values)
                assert (sp_m == sp).all()
                Mn = D_m / m_m
                b = dict(split=k, seed=seed)
                rows.append({**b, "method": f"{name} alone", "w": np.inf,
                             **outcomes(sp[Mn.argmin(1)], test, clean, info)})
                for w in (W_MAIN, *W_SENS):
                    rows.append({**b, "method": f"raw+{name}", "w": w,
                                 **outcomes(sp[(Rn + w * Mn).argmin(1)], test, clean, info)})
        print(f"split {k:2d} done", flush=True)
    return pd.DataFrame(rows)


def compare(per_split, a, b, w=W_MAIN, rng=None):
    """Paired comparison of method a vs b (rows of per_split) over splits."""
    def pick(m):
        ww = 0.0 if m in ("raw", "B4 nearest sequence") else (np.inf if m.endswith("alone") else w)
        return per_split[(per_split.method == m) & (per_split.w == ww)].set_index("split")
    A, B = pick(a), pick(b)
    out = []
    for c in OUTCOMES:
        d = (A[c] - B[c]).dropna()
        boots = [rng.choice(d.values, len(d)).mean() for _ in range(5000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p = wilcoxon(d.values).pvalue if (d != 0).any() else 1.0
        a_wins, b_wins = ((d < 0), (d > 0)) if c in LOWER_BETTER else ((d > 0), (d < 0))
        out.append(dict(a=a, b=b, w=w, outcome=c, mean_a=A[c].mean(), mean_b=B[c].mean(),
                        diff=d.mean(), ci_lo=lo, ci_hi=hi, a_better=int(a_wins.sum()),
                        b_better=int(b_wins.sum()), n=len(d), wilcoxon_p=p))
    return out


def main():
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    seeds = [int(x) for x in opts.get("seeds", "0,1,2,3,4").split(",")]
    clean, splits, X = load()
    check_inputs(sorted(splits.split_id.unique()), seeds)
    per_seed = run(clean, splits, X, seeds)
    per_split = per_seed.groupby(["split", "method", "w"], as_index=False)[OUTCOMES].mean()
    OUT.mkdir(exist_ok=True)
    kw = dict(index=False, lineterminator="\n")
    per_seed.to_csv(OUT / f"placement_kmer_tree{MK}_per_seed.csv", **kw)
    per_split.to_csv(OUT / f"placement_kmer_tree{MK}_per_split.csv", **kw)

    n = per_split.split.nunique()
    print(f"\nmarker: {MARKER} | {n} splits | seeds {seeds} | mean over splits (each split averaged over seeds)")
    print("tree_error = split age in million years between guessed and true species (lower = better)\n")
    tab = per_split.groupby(["method", "w"])[OUTCOMES].mean()
    order = ["raw"] + (["B4 nearest sequence"] if MARKER == "teleo" else []) \
        + [f"raw+{t}" for t in TARGETS] + [f"{t} alone" for t in TARGETS]
    main_tab = pd.concat([tab.loc[[(m, w) for (m, w) in tab.index if m == mm and w in (0.0, W_MAIN, np.inf)]]
                          for mm in order])
    print(main_tab.round(3).to_string())

    rng = np.random.default_rng(0)
    tests = []
    pairs = [("raw+tree", "raw"), ("raw+tree", "raw+labels"), ("raw+tree", "raw+taxonomy"),
             ("raw+labels", "raw"), ("raw+taxonomy", "raw"), ("raw+tree", "tree alone"),
             # fixed tree models (chosen on global data only, models/select_treeloss.py)
             ("raw+treecap", "raw+taxonomy"), ("raw+treecap", "raw"), ("raw+treecap", "raw+tree"),
             ("raw+treecap", "raw+labels"), ("raw+treeclade", "raw+taxonomy"), ("raw+treeclade", "raw"),
             ("raw+treeclade", "raw+tree"), ("treecap alone", "taxonomy alone"),
             ("treeclade alone", "taxonomy alone")]
    if MARKER == "teleo":
        pairs += [("raw+tree", "B4 nearest sequence"), ("raw+treecap", "B4 nearest sequence")]
    for a, b in pairs:
        tests += compare(per_split, a, b, rng=rng)
    print(f"\nPaired comparisons at w = {W_MAIN:g} (Wilcoxon over {n} splits; family_easy and "
          "tree_error = key outcomes; for tree_error 'better' means lower)")
    for t in tests:
        tag = "  <- key" if t["outcome"] in ("family_easy", "tree_error") else ""
        print(f"{t['a']:>12s} vs {t['b']:<20s} {t['outcome']:11s} {t['mean_a']:.3f} vs {t['mean_b']:.3f} "
              f"diff {t['diff']:+.3f} [{t['ci_lo']:+.3f}, {t['ci_hi']:+.3f}] "
              f"better/worse {t['a_better']:2d}/{t['b_better']:2d} p = {t['wilcoxon_p']:.3g}{tag}")

    print("\nSensitivity to the weight w (reported only, w = 1 was fixed in advance):")
    sens = tab.loc[[(m, w) for (m, w) in tab.index if m.startswith("raw+")]]
    print(sens.round(3).to_string())
    for w in W_SENS:
        for a, b in [("raw+tree", "raw"), ("raw+tree", "raw+labels"),
                     ("raw+treecap", "raw+taxonomy"), ("raw+treeclade", "raw+taxonomy")]:
            tests += [t for t in compare(per_split, a, b, w=w, rng=rng)
                      if t["outcome"] in ("family_easy", "tree_error")]
    pd.DataFrame(tests).to_csv(OUT / f"placement_kmer_tree{MK}_tests.csv", **kw)

    if len(seeds) > 1:
        ps = per_seed[(per_seed.w == W_MAIN)]
        for a, b in [("raw+tree", "raw+labels"), ("raw+treecap", "raw+taxonomy")]:
            print(f"\nPer seed, mean difference {a} minus {b} (w = 1):")
            for c in OUTCOMES:
                d = [(ps[(ps.method == a) & (ps.seed == s)].set_index("split")[c] -
                      ps[(ps.method == b) & (ps.seed == s)].set_index("split")[c]).mean() for s in seeds]
                print(f"{c:15s} " + "  ".join(f"s{s}: {x:+.3f}" for s, x in zip(seeds, d)))
    print(f"\nwrote placement_kmer_tree{MK}_per_split.csv, placement_kmer_tree{MK}_per_seed.csv, "
          f"placement_kmer_tree{MK}_tests.csv in {OUT}")


if __name__ == "__main__":
    main()
