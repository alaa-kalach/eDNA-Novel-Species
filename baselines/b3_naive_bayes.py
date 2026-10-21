#!/usr/bin/env python3
"""
B3 - naive Bayes classifier (RDP classifier, Wang et al. 2007), the standard
k-mer classifier for eDNA taxonomic assignment.

Training data = the same sequences the encoder baselines B5/B6 see:
  1. "pretraining": every GLOBAL fish sequence, loaded with models/pretrain.py's
     load_global() (MitoFish, no French Guiana species and no sequence identical to a
     French Guiana Teleo sequence -> no leakage of the hidden species);
  2. "fine-tuning": the split's French Guiana REFERENCE sequences are added.
  B5/B6 fine-tune on global + reference with the encoder; naive Bayes is a counting
  model, so the same step is simply adding the reference counts to the global counts.
  The candidate classes are all global species + all reference species.

Model (RDP):
  words      = the distinct overlapping 8-mers of a sequence (only A/C/G/T)
  word prior P(w)       = (n(w) + 0.5) / (N + 1)        n(w): training sequences with w
  P(w | species c)      = (m(w,c) + P(w)) / (M_c + 1)  m(w,c): sequences of c with w,
                                                       M_c: sequences of c
  log P(query | c)      = sum over the query's words of log P(w | c)
Bootstrap confidence: 100 times, draw 1/8 of the query's words (with replacement) and
see which class wins (ties share the vote).

Output per test sequence:
  nearest_species = reference species with the highest full-sequence likelihood
                    (always filled - used for genus/family placement)
  novelty_score   = 1 - bootstrap confidence of nearest_species
                    (low confidence, e.g. because global relatives or several regional
                    species compete, -> more likely novel)
  predicted_species = "novel" if novelty_score >= cut-off, else nearest_species
The cut-off is tuned per split on pseudo-novel splits built only from the reference
(the same pseudo splits as B2/B4; global data always stays in the training set).
B3 is deterministic for a given split (fixed bootstrap seeds), so no training seeds.

Outputs: outputs/results_nb[<tag>].csv
         outputs/thresholds_nb[<tag>].csv   (split_id, score_cutoff, pseudo_f1)
Grouping of B3's novel calls can use Person A's clustering, e.g.
    python clustering/cluster.py --emb=kmer --novel-from=outputs/results_nb.csv
Usage  : python baselines/b3_naive_bayes.py                                          # splits 1-10
         python baselines/b3_naive_bayes.py --splits=splits_extra.csv --tag=_extra   # splits 11-30
         add --no-global to train on the reference only (ablation -> results_nb_noglobal*.csv)
"""
import sys
import time
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import pandas as pd
from scipy import sparse

from common import (OUT, PSEUDO_REPEATS, ROOT, load_data, parse_opts, pseudo_rng,
                    pseudo_split, quick_summary, results_frame, split_frames,
                    tune_threshold)

sys.path.insert(0, str(ROOT))
from models.pretrain import load_global  # noqa: E402  (same global data as B5/B6)

K = 8                 # word size (RDP default)
N_WORDS = 4 ** K
BOOTSTRAPS = 100      # RDP default
BOOT_FRAC = 1 / 8     # share of the query's words drawn per bootstrap (RDP default)
_CODE = np.full(256, -1, dtype=np.int64)
for _i, _c in enumerate("ACGT"):
    _CODE[ord(_c)] = _i


def words(seq, k=K):
    """Sorted distinct k-mer codes of a sequence; k-mers with non-ACGT letters skipped."""
    a = _CODE[np.frombuffer(seq.upper().encode(), dtype=np.uint8)]
    if len(a) < k:
        return np.empty(0, dtype=np.int64)
    win = np.lib.stride_tricks.sliding_window_view(a, k)
    ok = (win >= 0).all(axis=1)
    codes = win[ok] @ (4 ** np.arange(k - 1, -1, -1))
    return np.unique(codes)


def presence(word_lists):
    """Sparse 0/1 matrix (sequences x words)."""
    rows = np.repeat(np.arange(len(word_lists)), [len(w) for w in word_lists])
    cols = np.concatenate(word_lists) if len(word_lists) else np.empty(0, int)
    return sparse.csr_matrix((np.ones(len(cols)), (rows, cols)),
                             shape=(len(word_lists), N_WORDS))


def counts_by_species(word_lists, species):
    """(classes, m, M): m[c, w] = sequences of class c containing w, M[c] = sequences of c."""
    species = np.asarray(species)
    classes, inv = np.unique(species, return_inverse=True)
    G = sparse.csr_matrix((np.ones(len(species)), (inv, np.arange(len(species)))),
                          shape=(len(classes), len(species)))
    return classes, (G @ presence(word_lists)).tocsr(), np.bincount(inv).astype(float)


class GlobalCounts:
    """'Pretrained' part: word counts of the global sequences, computed once."""

    def __init__(self, word_lists, species):
        self.classes, self.m, self.M = counts_by_species(word_lists, species)
        self.n_w = np.asarray(self.m.sum(axis=0)).ravel()   # sequences containing w
        self.N = float(self.M.sum())


class NaiveBayes:
    """RDP naive Bayes over global (optional) + reference species."""

    def __init__(self, ref_words, ref_species, glob=None):
        r_cls, r_m, r_M = counts_by_species(ref_words, ref_species)
        self.ref_classes = r_cls
        if glob is not None:
            assert not set(r_cls) & set(glob.classes), "reference species found in global data"
            m = sparse.vstack([r_m, glob.m]).tocsr()
            M = np.concatenate([r_M, glob.M])
        else:
            m, M = r_m, r_M
        n_w = np.asarray(m.sum(axis=0)).ravel()
        N = M.sum()
        self.n_ref = len(r_cls)                       # reference classes come first
        self.logPw = np.log((n_w + 0.5) / (N + 1))
        # log P(w|c) = log(m + P(w)) - log(M_c + 1). Only m > 0 differs from the
        # m = 0 value, so keep A[c, w] = log(m + P(w)) - log(P(w)) sparse.
        m = m.tocoo()
        Pw = np.exp(self.logPw)
        A = np.log(m.data + Pw[m.col]) - self.logPw[m.col]
        self.A_T = sparse.csr_matrix((A, (m.col, m.row)), shape=(N_WORDS, len(M)))
        self.logM1 = np.log(M + 1)

    def loglik(self, Q):
        """Q: sparse (queries x words) word counts. Returns dense (queries x classes),
        log P(query | c) up to a per-query constant (the same for every class)."""
        n = np.asarray(Q.sum(axis=1))
        return (Q @ self.A_T).toarray() - n * self.logM1[None, :]

    def score(self, query_words, rng):
        """Returns (novelty_score, nearest reference species)."""
        nq = len(query_words)
        L = self.loglik(presence(query_words))
        best = L[:, :self.n_ref].argmax(axis=1)       # nearest reference species
        conf = np.zeros(nq)
        n_draw = np.array([max(1, int(len(w) * BOOT_FRAC)) for w in query_words])
        has = np.array([len(w) > 0 for w in query_words])
        rows = np.repeat(np.arange(nq), n_draw)
        for _ in range(BOOTSTRAPS):
            cols = np.concatenate([w[rng.integers(len(w), size=d)] if len(w) else np.empty(0, int)
                                   for w, d in zip(query_words, n_draw)])
            Qb = sparse.csr_matrix((np.ones(len(cols)), (rows[np.repeat(has, n_draw)], cols)),
                                   shape=(nq, N_WORDS))
            Lb = self.loglik(Qb)
            top = Lb.max(axis=1, keepdims=True)
            tied = np.isclose(Lb, top, rtol=0, atol=1e-9)
            win = tied[np.arange(nq), best]
            conf += win / tied.sum(axis=1)
        conf /= BOOTSTRAPS
        conf[~has] = 0.0                               # no usable word -> no evidence
        return 1.0 - conf, self.ref_classes[best]


def boot_rng(split_id, r):
    return np.random.default_rng(300_000 + 100 * split_id + r)


def main():
    splits_file, tag = parse_opts()
    use_global = "--no-global" not in sys.argv[1:]
    name = "nb" if use_global else "nb_noglobal"
    t0 = time.time()
    seqs, splits = load_data(splits_file)
    W = pd.Series([words(s) for s in seqs["teleo"]], index=seqs.index)

    glob = None
    if use_global:
        with redirect_stdout(StringIO()) as log:
            g = load_global()
        gw = [words(s) for s in g["teleo"]]
        glob = GlobalCounts(gw, g["species_name"].values)
        print(f"{log.getvalue().strip()} -> {len(glob.classes)} global classes "
              f"({time.time() - t0:.0f}s)")
    else:
        print("ablation: reference only, no global data")

    res_all, thr_all = [], []
    for k in sorted(splits["split_id"].unique()):
        ref, test = split_frames(seqs, splits, k)

        # 1. tune the cut-off on pseudo-novel splits inside the reference
        sc, lab = [], []
        for r in range(PSEUDO_REPEATS):
            pref, pq = pseudo_split(ref, pseudo_rng(k, r))
            nb = NaiveBayes(list(W[pref.index]), pref["species_name"].values, glob)
            s, _ = nb.score(list(W[pq.index]), boot_rng(k, r))
            sc.append(s)
            lab.append(pq["is_novel"].values)
        t, f1 = tune_threshold(np.concatenate(sc), np.concatenate(lab))

        # 2. classify the real test set (global + full reference)
        nb = NaiveBayes(list(W[ref.index]), ref["species_name"].values, glob)
        score, nearest = nb.score(list(W[test.index]), boot_rng(k, 99))
        res = results_frame(test, k, score, nearest, score >= t)
        print(f"split {k:2d} | cut-off {t:.3f} (pseudo F1 {f1:.3f}) | "
              f"{quick_summary(res, test)} | {time.time() - t0:.0f}s", flush=True)
        res_all.append(res)
        thr_all.append({"split_id": k, "score_cutoff": round(t, 6), "pseudo_f1": round(f1, 4)})

    OUT.mkdir(exist_ok=True)
    kw = dict(index=False, lineterminator="\n")
    pd.concat(res_all, ignore_index=True).to_csv(OUT / f"results_{name}{tag}.csv", **kw)
    pd.DataFrame(thr_all).to_csv(OUT / f"thresholds_{name}{tag}.csv", **kw)
    print(f"wrote results_{name}{tag}.csv, thresholds_{name}{tag}.csv in {OUT}")


if __name__ == "__main__":
    main()
