#!/usr/bin/env python3
"""
B1 - forced assignment.

Every test sequence gets the species of its best VSEARCH match in the reference
(global alignment, highest identity). There is no "unknown" option, so the baseline
shows how often novel fish are assigned to a known species.

novelty_score is constant 0 (the method never ranks anything as novel -> AUROC 0.5).
nearest_species = predicted_species = species of the best hit (nothing is called novel).

Output: outputs/results_forced[<tag>].csv
Usage : python baselines/b1_forced.py                                          # splits 1-10
        python baselines/b1_forced.py --splits=splits_extra.csv --tag=_extra   # splits 11-30
"""
import pandas as pd

from common import (OUT, find_vsearch, load_data, parse_opts, quick_summary,
                    results_frame, split_frames, vsearch_best_hits)


def main():
    splits_file, tag = parse_opts()
    vs = find_vsearch()
    seqs, splits = load_data(splits_file)
    rows = []
    for k in sorted(splits["split_id"].unique()):
        ref, test = split_frames(seqs, splits, k)
        hits = vsearch_best_hits(test, ref, vs)
        res = results_frame(test, k, 0.0, hits["best_species"].values, False)
        print(f"split {k:2d} | {len(test)} test seqs | {quick_summary(res, test)}")
        rows.append(res)
    OUT.mkdir(exist_ok=True)
    out = pd.concat(rows, ignore_index=True)
    name = f"results_forced{tag}.csv"
    out.to_csv(OUT / name, index=False, lineterminator="\n")
    print(f"wrote {OUT / name} ({len(out)} rows)")


if __name__ == "__main__":
    main()
