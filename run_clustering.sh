#!/bin/bash
# Clustering of the sequences called novel, for every method, on all 30 splits.
# Novel calls come from B4 (raw k-mers), as in the final pipeline. Seed-0 embeddings.
# Needs Alaa's outputs/results_kmer.csv and outputs/results_kmer_extra.csv.
# Usage (repo root, venv active):  bash run_clustering.sh            # everything
set -e
run() {   # $1 = embedding, $2 = extra flag (optional)
  echo "=========== $1 $2 ==========="
  python -W ignore clustering/cluster.py --emb=$1 $2 --novel-from=outputs/results_kmer.csv
  python -W ignore clustering/cluster.py --emb=$1 $2 --novel-from=outputs/results_kmer_extra.csv \
         --splits=splits_extra.csv --tag=_extra
}
# HDBSCAN on raw k-mers (supplement); model-embedding runs were cut
run kmer
python -W ignore clustering/score_clusters.py kmer
