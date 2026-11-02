#!/bin/bash
# Clustering of the sequences called novel, for every method, on all 30 splits.
# Novel calls come from B4 (raw k-mers), as in the final pipeline. Seed-0 embeddings.
# Needs Alaa's outputs/results_kmer.csv and outputs/results_kmer_extra.csv.
# Usage (repo root, venv active):  bash run_clustering.sh            # everything
#                                  bash run_clustering.sh new        # only the new runs (cap 90 + raw + model)
set -e
run() {   # $1 = embedding, $2 = extra flag (optional)
  echo "=========== $1 $2 ==========="
  python -W ignore clustering/cluster.py --emb=$1 $2 --novel-from=outputs/results_kmer.csv
  python -W ignore clustering/cluster.py --emb=$1 $2 --novel-from=outputs/results_kmer_extra.csv \
         --splits=splits_extra.csv --tag=_extra
}
if [ "$1" != "new" ]; then
  # model space alone (grouping + naming with the model embedding only)
  for emb in kmer ft_tree2g_w10m0.4 ft_labels ft_taxonomy ft_treecap120 ft_treeclade; do run $emb; done
fi
run ft_treecap90
# raw + model: the same combined distance as the final placement method
for emb in ft_taxonomy ft_treecap90 ft_treecap120 ft_treeclade ft_labels; do run $emb --with-raw; done

# scores (ARI, AMI, B-cubed F1, family / genus naming) for every method vs kmer and vs taxonomy
python -W ignore clustering/score_clusters.py kmer ft_taxonomy_withraw ft_taxonomy ft_labels_withraw \
       ft_treeclade_withraw ft_treecap120_withraw ft_treecap90_withraw ft_treeclade ft_treecap120 ft_treecap90 \
       ft_labels ft_tree2g_w10m0.4
