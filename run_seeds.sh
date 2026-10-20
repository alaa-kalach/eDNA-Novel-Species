#!/bin/bash
# Train and evaluate the tree (two-goal) and labels models with extra random seeds.
# Seed 0 is the run you already have. Usage (repo root, venv active):
#     bash run_seeds.sh 1 2 3 4
set -e
for s in "$@"; do
  echo "=========== seed $s ==========="
  python -W ignore models/select_twogoal.py w10m0.4 --seed=$s
  python -W ignore models/pretrain.py labels --seed=$s
  python -W ignore models/finetune_eval.py tree2g_w10m0.4 labels --seed=$s
  python -W ignore models/finetune_eval.py tree2g_w10m0.4 labels --seed=$s --splits=splits_extra.csv --tag=_extra
done
python -W ignore models/placement_test.py --seeds=0,$(echo "$@" | tr ' ' ',')
