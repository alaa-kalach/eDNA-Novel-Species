"""
Ablations of the final placement pipeline (raw k-mers + capped-tree model).

Each ablation removes one part and keeps everything else the same:
    full          raw k-mers + fine-tuned model             (outputs/embeddings/embeddings_ft_<target>...)
    no raw        fine-tuned model alone (no raw k-mer distance)
    no fine-tune  raw k-mers + global-pretrained model, not fine-tuned on the split's reference
    no pretrain   raw k-mers + model trained ONLY on the split's French Guiana reference
                  (random start, no global data; same network, target and number of steps
                  as the global pretraining)
    raw only      raw k-mers alone (B4-style nearest species centre)

Step 1 (training, only 'no fine-tune' and 'no pretrain' need new embeddings):
    python models/ablation.py train --target=treecap120 --seeds=0,1,2,3,4
Step 2 (scoring, no training):
    python models/ablation.py score --target=treecap120 --seeds=0,1,2,3,4
Add --marker=full for the long 12S region.

Outcomes for true novel sequences (same as rawplus_preview.py): family, family with no
congener in the reference, genus when a congener is present, tree error (split age in My
between guessed and true species, lower = better). Each split is averaged over the seeds,
then each ablation is compared with 'full' (paired Wilcoxon over the 30 splits).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.encoder import DistanceEncoder  # noqa: E402
from models.pretrain import (HP, TRAIN, encoder_settings, global_target, load_encoder,  # noqa: E402
                             marker_suffix, parse_args, seed_suffix)

DATA, EMB, OUT = ROOT / "data", ROOT / "outputs" / "embeddings", ROOT / "outputs" / "results"


def load_common(marker):
    clean = pd.read_csv(DATA / "guiana_teleo_clean.csv")
    clean["seq_id"] = clean["Specimen identification code"].astype(str)
    clean = clean[clean.in_tree].drop_duplicates("seq_id").set_index("seq_id")
    k = np.load(DATA / ("kmer_features_full12S.npz" if marker == "full" else "kmer_features_k3to6.npz"),
                allow_pickle=True)
    X = pd.DataFrame(k["X"], index=k["seq_id"].astype(str))
    splits = pd.concat([pd.read_csv(DATA / "splits.csv"), pd.read_csv(DATA / "splits_extra.csv")])
    splits["seq_id"] = splits.seq_id.astype(str)
    tz = np.load(DATA / "global_tree_distances.npz")
    return clean, X, splits, tz


def emb_path(kind, target, mk, seed, sid):
    return EMB / f"embeddings_{kind}_{target}{mk}{seed_suffix(seed)}_split{sid}.npz"


def train(target, seeds, marker):
    mk = marker_suffix(marker)
    clean, X, splits, tz = load_common(marker)
    row = {s: i for i, s in enumerate(tz["species"])}
    T = global_target(target, tz)
    EMB.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        pre = load_encoder(ROOT / "models" / f"pretrained_{target}{mk}{seed_suffix(seed)}.npz")
        for sid in sorted(splits.split_id.unique()):
            t0 = time.time()
            s = splits[splits.split_id == sid]
            ref, test = s[s.role == "reference"].seq_id.values, s[s.role != "reference"].seq_id.values
            ids = np.concatenate([ref, test])
            # no fine-tune: the global-pretrained model as it is
            np.savez(emb_path("ablpre", target, mk, seed, sid), seq_id=ids,
                     emb=pre.embed(X.loc[ids].values).astype(np.float32))
            # no pretrain: same model and steps, trained only on this split's reference
            rf = np.array([row[x] for x in clean.loc[ref, "species_name"]])
            enc = DistanceEncoder(X.shape[1], seed=seed, **encoder_settings(target, tz), **HP)
            enc.fit_minibatch(X.loc[ref].values, rf, T, seed=seed + 1000 * sid, verbose=False, **TRAIN)
            np.savez(emb_path("ablnopre", target, mk, seed, sid), seq_id=ids,
                     emb=enc.embed(X.loc[ids].values).astype(np.float32))
            print(f"{target}{mk} seed {seed} split {sid:2d} done ({time.time() - t0:.0f}s)", flush=True)


def dist(A, B):
    return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T, 0))


def score(target, seeds, marker):
    mk = marker_suffix(marker)
    clean, X, splits, tz = load_common(marker)
    info = clean.drop_duplicates("species_name").set_index("species_name")
    trow = {s: i for i, s in enumerate(tz["species"])}
    tscale = float(tz["scale"])

    def centre_dist(V, ref, rsp, test):
        sps = np.unique(rsp)
        C = np.array([V.loc[ref[rsp == x]].values.mean(0) for x in sps])
        return sps, dist(V.loc[test].values, C) / np.median(dist(C, C)[np.triu_indices(len(C), 1)])

    def outcomes(pred, true, diff):
        fam = info.loc[pred, "Family"].values == info.loc[true, "Family"].values
        gen = info.loc[pred, "Genus"].values == info.loc[true, "Genus"].values
        age = tz["D"][[trow[p] for p in pred], [trow[t] for t in true]] * tscale / 2
        return dict(family=fam.mean(), family_easy=fam[diff == "easy"].mean(),
                    genus_hard=gen[diff == "hard"].mean(),
                    tree_error=age.mean(), tree_error_easy=age[diff == "easy"].mean())

    files = {"full": "ft", "no fine-tune": "ablpre", "no pretrain": "ablnopre"}
    rows = []
    for sid in sorted(splits.split_id.unique()):
        s = splits[splits.split_id == sid]
        ref = s[s.role == "reference"].seq_id.values
        nov = s[s.role == "test_novel"]
        test, diff = nov.seq_id.values, nov.difficulty.values
        rsp, true = clean.loc[ref, "species_name"].values, clean.loc[test, "species_name"].values
        sps, Draw = centre_dist(X, ref, rsp, test)
        r = {"split": sid, **{f"raw only|{o}": v for o, v in outcomes(sps[Draw.argmin(1)], true, diff).items()}}
        per = {}
        for seed in seeds:
            D = {}
            for name, kind in files.items():
                f = emb_path(kind, target, mk, seed, sid)
                if not f.exists():
                    break
                z = np.load(f, allow_pickle=True)
                D[name] = centre_dist(pd.DataFrame(z["emb"], index=z["seq_id"].astype(str)), ref, rsp, test)[1]
            if len(D) < len(files):
                continue                              # use only seeds that have every variant
            for name in files:
                per.setdefault(name, []).append(outcomes(sps[(Draw + D[name]).argmin(1)], true, diff))
            per.setdefault("no raw", []).append(outcomes(sps[D["full"].argmin(1)], true, diff))
        if not per:
            raise SystemExit(f"split {sid}: no seed has all embeddings - run the train step first")
        for name, lst in per.items():
            for o in lst[0]:
                r[f"{name}|{o}"] = np.mean([x[o] for x in lst])
        r["n_seeds"] = len(per["full"])
        rows.append(r)
    df = pd.DataFrame(rows).set_index("split")
    df.round(4).to_csv(OUT / f"ablation_{target}{mk}.csv")

    outs = ["family", "family_easy", "genus_hard", "tree_error", "tree_error_easy"]
    names = ["full", "no raw", "no fine-tune", "no pretrain", "raw only"]
    print(f"target {target}{mk} | {len(df)} splits | seeds per split: {int(df.n_seeds.min())}\n")
    print("mean over splits (tree error in My, lower = better)")
    print(pd.DataFrame({n: [df[f"{n}|{o}"].mean() for o in outs] for n in names}, index=outs).round(3).to_string())
    print("\neach ablation vs full: difference (ablation - full), splits where the ablation is worse, Wilcoxon p")
    for n in names[1:]:
        cells = []
        for o in outs:
            d = df[f"{n}|{o}"] - df[f"full|{o}"]
            worse = (d > 0) if o.startswith("tree_error") else (d < 0)
            p = wilcoxon(df[f"{n}|{o}"], df[f"full|{o}"]).pvalue if (d != 0).sum() > 1 else 1.0
            cells.append(f"{o} {d.mean():+.3f} ({worse.sum()}/{len(d)}, p={p:.3f})")
        print(f"  {n:13s} " + " | ".join(cells))
    print(f"\nsaved outputs/results/ablation_{target}{mk}.csv")


def main():
    args, opts = parse_args(sys.argv[1:])
    target = opts.get("target", "treecap120")
    seeds = [int(x) for x in opts.get("seeds", "0,1,2,3,4").split(",")]
    marker = opts.get("marker", "teleo")
    if args[:1] == ["train"]:
        train(target, seeds, marker)
    elif args[:1] == ["score"]:
        score(target, seeds, marker)
    else:
        raise SystemExit("usage: python models/ablation.py train|score --target=treecap120 --seeds=0,1,2,3,4")


if __name__ == "__main__":
    main()
