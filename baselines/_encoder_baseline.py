"""Shared runner for the encoder baselines B5 (labels) and B6 (taxonomy).

They use exactly the same pipeline as the main method (models/), only the
training target changes:
  1. pretrain on global fish        -> models/pretrained_<target>[_s<N>].npz
  2. fine-tune on each split's reference and score the test sequences
     on splits 1-10 (splits.csv) and 11-30 (splits_extra.csv)
     -> outputs/results/results_ft_<target>[_s<N>][_extra].csv
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script, *args):
    cmd = [sys.executable, "-W", "ignore", str(ROOT / "models" / script), *args]
    print(">>", " ".join(cmd[3:]), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main(target):
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
    seeds = [int(s) for s in opts.get("seeds", "0").split(",")]
    for s in seeds:
        run("pretrain.py", target, f"--seed={s}")
        run("finetune_eval.py", target, f"--seed={s}")
        run("finetune_eval.py", target, f"--seed={s}", "--splits=splits_extra.csv", "--tag=_extra")
