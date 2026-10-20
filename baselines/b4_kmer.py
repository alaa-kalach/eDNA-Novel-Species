#!/usr/bin/env python3
"""
B4 - raw k-mer space (no training).

Embedding: concatenated k-mer frequency profiles for k = 3..6 (64+256+1024+4096 = 5440
dims). Each k block sums to 1, so every k carries equal weight. k-mers containing a
non-ACGT letter (IUPAC ambiguity codes) are skipped. Distance = Euclidean.

Novelty score (same definition as the main method, proposal section 5.2):
    d_q  = distance from query q to its nearest known species s*
           (= nearest reference sequence; s* = that sequence's species)
    r_s* = distance from s* to its own nearest relative in the reference
           (min distance between a sequence of s* and a sequence of any other species)
    novelty_score = d_q / r_s*
The cut-off is tuned per split on pseudo-novel splits inside the reference
(r_s recomputed inside each pseudo-reference). Novel if score >= cut-off,
otherwise the query takes the label of s*. nearest_species = s* is always written,
also for sequences called novel (needed to score genus/family placement).

Outputs: outputs/embeddings_kmer.npy      float32, one row per sequence (1200 x 5440)
         outputs/embeddings_kmer_ids.csv  seq_id of each row, same order
         outputs/results_kmer[<tag>].csv
         outputs/thresholds_kmer[<tag>].csv      (split_id, score_cutoff, pseudo_f1)
The embedding does not depend on the split, so the .npy files are the same for
every splits file (rewritten on each run, identical content).
HDBSCAN clustering of embeddings_kmer.npy is done by Person A (clustering/).
Usage  : python baselines/b4_kmer.py                                          # splits 1-10
         python baselines/b4_kmer.py --splits=splits_extra.csv --tag=_extra   # splits 11-30
"""
from itertools import product

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from common import (OUT, PSEUDO_REPEATS, load_data, parse_opts, pseudo_rng,
                    pseudo_split, quick_summary, results_frame, split_frames,
                    tune_threshold)

KS = (3, 4, 5, 6)
EPS = 1e-9


def kmer_profiles(sequences, ks=KS):
    blocks = []
    for k in ks:
        index = {"".join(p): i for i, p in enumerate(product("ACGT", repeat=k))}
        M = np.zeros((len(sequences), 4 ** k), dtype=np.float64)
        for r, s in enumerate(sequences):
            for j in range(len(s) - k + 1):
                i = index.get(s[j:j + k])
                if i is not None:
                    M[r, i] += 1
        M /= np.maximum(M.sum(axis=1, keepdims=True), 1)
        blocks.append(M)
    return np.hstack(blocks)


class KmerNovelty:
    """Nearest-species distance scaled by that species' nearest-relative distance."""

    def __init__(self, X_ref, species_ref):
        self.X = X_ref
        self.sp = np.asarray(species_ref)
        D = cdist(X_ref, X_ref)
        D[self.sp[:, None] == self.sp[None, :]] = np.inf      # ignore own species
        per_seq = D.min(axis=1)
        self.r = pd.Series(per_seq).groupby(self.sp).min()   # species -> nearest relative

    def score(self, X_q):
        D = cdist(X_q, self.X)
        j = D.argmin(axis=1)
        nearest = self.sp[j]
        d = D[np.arange(len(j)), j]
        return d / np.maximum(self.r.loc[nearest].values, EPS), nearest


def main():
    splits_file, tag = parse_opts()
    seqs, splits = load_data(splits_file)
    X = kmer_profiles(seqs["teleo"].tolist())
    row = pd.Series(np.arange(len(seqs)), index=seqs.index)

    OUT.mkdir(exist_ok=True)
    np.save(OUT / "embeddings_kmer.npy", X.astype(np.float32))
    pd.DataFrame({"seq_id": seqs.index}).to_csv(OUT / "embeddings_kmer_ids.csv",
                                                index=False, lineterminator="\n")
    print(f"embeddings_kmer.npy: {X.shape} (k = {KS})")

    res_all, thr_all = [], []
    for k in sorted(splits["split_id"].unique()):
        ref, test = split_frames(seqs, splits, k)

        sc, lab = [], []
        for r in range(PSEUDO_REPEATS):
            pref, pq = pseudo_split(ref, pseudo_rng(k, r))
            m = KmerNovelty(X[row[pref.index]], pref["species_name"])
            s, _ = m.score(X[row[pq.index]])
            sc.append(s)
            lab.append(pq["is_novel"].values)
        t, f1 = tune_threshold(np.concatenate(sc), np.concatenate(lab))

        m = KmerNovelty(X[row[ref.index]], ref["species_name"])
        score, nearest = m.score(X[row[test.index]])
        res = results_frame(test, k, score, nearest, score >= t)
        print(f"split {k:2d} | cut-off {t:.3f} (pseudo F1 {f1:.3f}) | {quick_summary(res, test)}")
        res_all.append(res)
        thr_all.append({"split_id": k, "score_cutoff": round(t, 6), "pseudo_f1": round(f1, 4)})

    kw = dict(index=False, lineterminator="\n")
    pd.concat(res_all, ignore_index=True).to_csv(OUT / f"results_kmer{tag}.csv", **kw)
    pd.DataFrame(thr_all).to_csv(OUT / f"thresholds_kmer{tag}.csv", **kw)
    print(f"wrote embeddings_kmer.npy, embeddings_kmer_ids.csv, results_kmer{tag}.csv, "
          f"thresholds_kmer{tag}.csv in {OUT}")


if __name__ == "__main__":
    main()
