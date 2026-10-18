#!/usr/bin/env python3
"""
B2 - VSEARCH identity threshold (standard eDNA pipeline).

1. Tune an identity cut-off per split on pseudo-novel splits built only from that
   split's reference (PSEUDO_REPEATS splits pooled, F1 of the novel class maximised).
   The true novel species are never used for tuning.
2. Test sequence -> best VSEARCH identity to the reference.
   identity < cut-off  -> "novel";  otherwise -> species of the best hit.
   novelty_score = 1 - best identity.
3. Sequences called novel are clustered with vsearch --cluster_fast at the same cut-off.

Outputs: outputs/results_vsearch.csv
         outputs/clusters_vsearch.csv   (seq_id, split_id, cluster_id)
         outputs/thresholds_vsearch.csv (split_id, identity_cutoff, pseudo_f1)
Usage  : python baselines/b2_vsearch.py
"""
import numpy as np
import pandas as pd

from common import (NOVEL_LABEL, OUT, PSEUDO_REPEATS, find_vsearch, load_data,
                    pseudo_rng, pseudo_split, quick_summary, results_frame,
                    split_frames, tune_threshold, vsearch_best_hits, vsearch_cluster)


def main():
    vs = find_vsearch()
    seqs, splits = load_data()
    res_all, clu_all, thr_all = [], [], []
    for k in sorted(splits["split_id"].unique()):
        ref, test = split_frames(seqs, splits, k)

        # 1. tune on pseudo-novel splits inside the reference
        sc, lab = [], []
        for r in range(PSEUDO_REPEATS):
            pref, pq = pseudo_split(ref, pseudo_rng(k, r))
            h = vsearch_best_hits(pq, pref, vs)
            sc.append(1 - h["best_id"].values)
            lab.append(pq["is_novel"].values)
        t, f1 = tune_threshold(np.concatenate(sc), np.concatenate(lab))
        cutoff = 1 - t                                   # identity cut-off

        # 2. classify the real test set
        h = vsearch_best_hits(test, ref, vs)
        score = 1 - h["best_id"].values
        pred = np.where(score >= t, NOVEL_LABEL, h["best_species"].values)
        res = results_frame(test, k, score, pred)

        # 3. cluster the predicted-novel sequences at the same cut-off
        novel = test.loc[res.loc[res["predicted_species"] == NOVEL_LABEL, "seq_id"]]
        cl = vsearch_cluster(novel, cutoff, vs)
        clu_all.append(pd.DataFrame({"seq_id": cl.index, "split_id": k,
                                     "cluster_id": cl.values}))

        print(f"split {k:2d} | cut-off {cutoff:.4f} (pseudo F1 {f1:.3f}) | "
              f"{quick_summary(res, test)} | {cl.nunique()} novel clusters "
              f"(true {test.loc[test.role == 'test_novel', 'species_name'].nunique()})")
        res_all.append(res)
        thr_all.append({"split_id": k, "identity_cutoff": round(cutoff, 6),
                        "pseudo_f1": round(f1, 4)})

    OUT.mkdir(exist_ok=True)
    kw = dict(index=False, lineterminator="\n")
    pd.concat(res_all, ignore_index=True).to_csv(OUT / "results_vsearch.csv", **kw)
    pd.concat(clu_all, ignore_index=True).to_csv(OUT / "clusters_vsearch.csv", **kw)
    pd.DataFrame(thr_all).to_csv(OUT / "thresholds_vsearch.csv", **kw)
    print(f"wrote results_vsearch.csv, clusters_vsearch.csv, thresholds_vsearch.csv in {OUT}")


if __name__ == "__main__":
    main()
