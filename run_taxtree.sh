#!/bin/bash
# Taxonomy + tree: train the chosen weight with 5 seeds, evaluate, compare with B6 (taxonomy only).
# Usage (repo root, venv active):  bash run_taxtree.sh 0.25      <- the a chosen by select_taxtree.py
set -e
A=$1
for s in 0 1 2 3 4; do
  echo "=========== taxtree$A seed $s ==========="
  python -W ignore models/pretrain.py taxtree$A --seed=$s
  python -W ignore models/finetune_eval.py taxtree$A --seed=$s
  python -W ignore models/finetune_eval.py taxtree$A --seed=$s --splits=splits_extra.csv --tag=_extra
done
python -W ignore models/placement_test.py --a=ft_taxtree$A --b=ft_taxonomy --seeds=0,1,2,3,4
