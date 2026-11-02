#!/bin/bash
# Per-group novelty cut-offs (Next Directions option 1): global vs taxonomy groups vs tree clades,
# on Teleo and long 12S. Hyper-parameters are chosen inside each split's reference only.
# Usage (repo root, venv active):  bash run_group_cutoffs.sh
set -e
# all-pairs VSEARCH identity, once per marker (skip if the .npz files are already in data/)
[ -f data/pairwise_identity_teleo.npz ]   || python -W ignore data/pairwise_identity.py
[ -f data/pairwise_identity_full12s.npz ] || python -W ignore data/pairwise_identity.py --marker=full
python -W ignore models/group_cutoffs.py                  # ~4 min
python -W ignore models/group_cutoffs.py --marker=full    # ~4 min
