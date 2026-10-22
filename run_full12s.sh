#!/bin/bash
# Long-marker experiment: does the tree beat taxonomy when the marker carries more tree signal?
# Same species, splits, tree, taxonomy and training recipe as the Teleo runs; only the DNA region changes.
# Usage (repo root, venv active):  bash run_full12s.sh
set -e
# data first, checked by hand:  python -W ignore data/make_full12s.py
for s in 0 1 2 3 4; do
  for t in tree2g_w10m0.4 taxonomy labels; do
    echo "=========== full 12S | $t | seed $s ==========="
    python -W ignore models/pretrain.py $t --seed=$s --marker=full
    python -W ignore models/finetune_eval.py $t --seed=$s --marker=full
    python -W ignore models/finetune_eval.py $t --seed=$s --marker=full --splits=splits_extra.csv --tag=_extra
  done
done
python -W ignore models/placement_test.py --a=ft_tree2g_w10m0.4_full --b=ft_taxonomy_full --marker=full --seeds=0,1,2,3,4
python -W ignore rawplus_preview.py --marker=full
