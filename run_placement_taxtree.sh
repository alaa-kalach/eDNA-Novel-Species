#!/bin/bash
# Taxonomy + tree for placement, no new training: raw k-mers + taxonomy model + tree-clade model,
# combined distance with weights fixed in advance. Teleo and long 12S, 5 seeds x 30 splits.
# Usage (repo root, venv active):  bash run_placement_taxtree.sh
set -e
python -W ignore models/placement_taxtree.py
python -W ignore models/placement_taxtree.py --marker=full
