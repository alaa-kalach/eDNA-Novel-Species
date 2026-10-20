"""
Evaluate the pretrained encoders on the French Guiana splits, with and without
fine-tuning on each split's reference.

Variants (per split, test species never used for training or choices):
    pre_<target>   pretrained on global data only, applied as is
    ft_<target>    pretrained, then trained further on global data + this
                   split's French Guiana REFERENCE (half of each batch from each)

Usage (from the repo root):
    python models/finetune_eval.py                 # tree + labels, splits 1-10
    python models/finetune_eval.py tree 1 2 3
    python models/finetune_eval.py tree2g_w10m0.4       # two-goal model from select_twogoal.py
    python models/finetune_eval.py tree2g_w10m0.4 labels --splits=splits_extra.csv --tag=_extra
    python models/finetune_eval.py tree2g_w10m0.4 labels --seed=1   # uses pretrained_*_s1.npz
Output: outputs/results/results_<variant>.csv           (proposal's scaled novelty score)
        outputs/results/results_<variant>_unscaled.csv  (plain distance to nearest species)
        seed N > 0 adds _s<N> before the tag, e.g. results_ft_labels_s1_extra.csv
        outputs/embeddings/embeddings_ft_<target>[_s<N>]_split<k>.npz  (seq_id, emb; reference + test,
                                                                         used by clustering/cluster.py)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.encoder import novelty_scores  # noqa: E402
from models.pretrain import global_target, load_encoder, load_global, parse_args, seed_suffix  # noqa: E402
from models.kmers import kmer_profile  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "outputs" / "results"
EMB = ROOT / "outputs" / "embeddings"
FT = dict(steps=800, batch=512, lr=3e-4)


def finetune(enc, Xg, rg, Xf, rf, T, seed):
    """Continue training: each batch = half global, half French Guiana reference."""
    rng = np.random.default_rng(seed)
    enc.lr = FT["lr"]
    Xg_s, Xf_s = enc._scale(Xg), enc._scale(Xf)
    h = FT["batch"] // 2
    for _ in range(FT["steps"]):
        bg = rng.choice(len(Xg), size=h, replace=False)
        bf = rng.choice(len(Xf), size=min(h, len(Xf)), replace=False)
        Xb = np.vstack([Xg_s[bg], Xf_s[bf]])
        sb = np.concatenate([rg[bg], rf[bf]])
        _, grads = enc._loss_and_grads(Xb, T[np.ix_(sb, sb)])
        enc._adam(grads)
    return enc


def main():
    args, opts = parse_args(sys.argv[1:])
    seed = int(opts.get("seed", 0))
    sfx = seed_suffix(seed)
    splits_file = opts.get("splits", "splits.csv")     # e.g. --splits=splits_extra.csv
    tag = opts.get("tag", "")                          # e.g. --tag=_extra (keeps old results)
    targets = [a for a in args if not a.isdigit()] or ["tree", "labels"]
    split_ids = [int(a) for a in args if a.isdigit()] or None
    OUT.mkdir(parents=True, exist_ok=True)
    EMB.mkdir(parents=True, exist_ok=True)

    tz = np.load(DATA / "global_tree_distances.npz")
    row = {s: i for i, s in enumerate(tz["species"])}
    g = load_global()
    Xg = np.array([kmer_profile(s) for s in g.teleo])
    rg = g.species_name.map(row).values

    clean = pd.read_csv(DATA / "guiana_teleo_clean.csv")
    clean["seq_id"] = clean["Specimen identification code"].astype(str)
    clean = clean[clean.in_tree].set_index("seq_id")
    k = np.load(DATA / "kmer_features_k3to6.npz", allow_pickle=True)
    X_all = pd.DataFrame(k["X"], index=k["seq_id"].astype(str))
    splits = pd.read_csv(DATA / splits_file)
    splits["seq_id"] = splits["seq_id"].astype(str)

    if split_ids is None:
        split_ids = sorted(splits.split_id.unique())
    for target in targets:
        # labels* -> same/different species, taxonomy* -> taxonomy, anything else -> tree
        T = global_target(target, tz)
        res = {f"pre_{target}": [], f"ft_{target}": []}
        for sid in split_ids:
            t0 = time.time()
            s = splits[splits.split_id == sid]
            ref, test = s[s.role == "reference"], s[s.role != "reference"]
            Xr, Xt = X_all.loc[ref.seq_id].values, X_all.loc[test.seq_id].values
            sp_r = clean.loc[ref.seq_id, "species_name"].values
            for variant in [f"pre_{target}", f"ft_{target}"]:
                enc = load_encoder(ROOT / "models" / f"pretrained_{target}{sfx}.npz")
                if variant.startswith("ft_"):
                    rf = np.array([row[x] for x in sp_r])
                    enc = finetune(enc, Xg, rg, Xr, rf, T, seed=sid + 1000 * seed)
                Zr, Zt = enc.embed(Xr), enc.embed(Xt)
                if variant.startswith("ft_"):          # keep fine-tuned embeddings for clustering
                    np.savez(EMB / f"embeddings_{variant}{sfx}_split{sid}.npz",
                             seq_id=np.concatenate([ref.seq_id.values, test.seq_id.values]),
                             emb=np.vstack([Zr, Zt]).astype(np.float32))
                for scaled, score_tag in [(True, ""), (False, "_unscaled")]:
                    score, pred = novelty_scores(Zr, sp_r, Zt, scaled=scaled)
                    res.setdefault(variant + score_tag, []).append(pd.DataFrame(
                        {"seq_id": test.seq_id.values, "split_id": sid,
                         "novelty_score": score, "predicted_species": pred}))
            print(f"{target:6s} seed {seed} split {sid:2d} done ({time.time()-t0:.0f}s)", flush=True)
        for variant, parts in res.items():
            pd.concat(parts).to_csv(OUT / f"results_{variant}{sfx}{tag}.csv", index=False)
            print(f"wrote results_{variant}{sfx}{tag}.csv", flush=True)


if __name__ == "__main__":
    main()
