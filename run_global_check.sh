#!/bin/bash
# Confirm the taxonomy + tree combination on GLOBAL held-out species (no French Guiana data),
# with the same held-out check that chose the tree-clade model. Teleo and long 12S, seeds 0-4.
# Usage (repo root, venv active):  bash run_global_check.sh
set -e
# global taxonomy table (not in git): download MitoFish ONLY (check_mitofish.py would also
# rewrite data/global_teleo.csv), then build data/global_taxonomy.csv
if [ ! -f data/global_taxonomy.csv ]; then
  [ -f data/mitofish.12S.Oct2023.tsv ] || python -c "import urllib.request; urllib.request.urlretrieve('https://zenodo.org/records/8428816/files/mitofish.12S.Oct2023.tsv?download=1', 'data/mitofish.12S.Oct2023.tsv')"
  python -W ignore data/add_global_taxonomy.py
fi
[ -f data/global_tree_distances.npz ] || python -W ignore models/tree_distances.py
python -W ignore models/global_check_combined.py
python -W ignore models/global_check_combined.py --marker=full
