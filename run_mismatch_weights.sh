#!/bin/bash
# Position-weighted mismatches (Next Directions "structure inside the sequence", step 1):
# VSEARCH vs the same alignment counted flat / weighted by within-species variability /
# weighted by log-likelihood ratio, on Teleo and long 12S. Weights come from each reference only.
# Usage (repo root, venv active):  bash run_mismatch_weights.sh
set -e
[ -f data/pairwise_identity_teleo.npz ]   || python -W ignore data/pairwise_identity.py
[ -f data/pairwise_identity_full12s.npz ] || python -W ignore data/pairwise_identity.py --marker=full
python -W ignore data/pair_alignments.py                  # needs VSEARCH (tools/vsearch.exe)
python -W ignore data/pair_alignments.py --marker=full
python -W ignore models/mismatch_weights.py
python -W ignore models/mismatch_weights.py --marker=full
