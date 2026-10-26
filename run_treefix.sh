#!/bin/bash
# Train one new target with 5 seeds on all 30 splits, then compare it with taxonomy and the
# current tree model (model alone and raw + model), plus the tree-fit diagnostic.
# Usage (repo root, venv active):  bash run_treefix.sh treecap60          # Teleo
#                                  bash run_treefix.sh treecap60 full     # long 12S
set -e
T=$1
M=""; MARK=""; MK=""
if [ "$2" == "full" ]; then M="--marker=full"; MARK="--marker=full"; MK="_full"; fi
for s in 0 1 2 3 4; do
  echo "=========== $T $2 seed $s ==========="
  python -W ignore models/pretrain.py $T --seed=$s $M
  python -W ignore models/finetune_eval.py $T --seed=$s $M
  python -W ignore models/finetune_eval.py $T --seed=$s $M --splits=splits_extra.csv --tag=_extra
done
python -W ignore models/placement_test.py --a=ft_$T$MK --b=ft_taxonomy$MK $MARK --seeds=0,1,2,3,4
python -W ignore models/placement_test.py --a=ft_$T$MK --b=ft_tree2g_w10m0.4$MK $MARK --seeds=0,1,2,3,4
python -W ignore rawplus_preview.py $MARK --add=$T
python -W ignore models/tree_fit.py $MARK --add=ft_$T
