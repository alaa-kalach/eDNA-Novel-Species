#!/bin/bash
# Clustering of the sequences called novel, for every method, on all 30 splits.
# Novel calls come from B4 (raw k-mers), as in the final pipeline. Seed-0 embeddings.
# Needs Alaa's outputs/results_kmer.csv and outputs/results_kmer_extra.csv.
# Usage (repo root, venv active):  bash run_clustering.sh
set -e
for emb in kmer ft_tree2g_w10m0.4 ft_labels ft_taxonomy ft_treecap120 ft_treeclade; do
  echo "=========== $emb ==========="
  python -W ignore clustering/cluster.py --emb=$emb --novel-from=outputs/results_kmer.csv
  python -W ignore clustering/cluster.py --emb=$emb --novel-from=outputs/results_kmer_extra.csv \
         --splits=splits_extra.csv --tag=_extra
done
