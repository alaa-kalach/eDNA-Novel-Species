"""
Two placement-time improvements for raw k-mers + capped-tree model (no new training).

1. Gate:    if the query is very close to a known species in raw k-mer space (scaled
            nearest raw distance < theta), trust raw k-mers alone (better at the exact
            genus); otherwise use raw + model. Goal: win back genus accuracy.
2. Medoid:  take the k nearest reference species under raw + model and pick the one
            with the smallest weighted tree distance to the others (weights 1 / distance).
            Goal: smaller tree error (a "minimum-risk" guess on the tree).

HONEST TUNING: theta and k are chosen on GLOBAL held-out fish only - the 10% of global
species that every pretrained model was trained without (same species as in
models/pretrain.py). Reference = all other global species, model = the global-pretrained
model (it never saw the held-out species), raw = k-mer profiles. No French Guiana data.
Rules fixed in advance:
    gate   -> theta with the highest balanced accuracy = (family + genus_congener) / 2
    medoid -> k with the lowest tree error, among the k that lose at most 1 point of
              family accuracy compared with plain raw + model (k = 1)
Then both are applied ONCE to French Guiana (fine-tuned models, 30 splits x 5 seeds) and
compared with plain raw + model and with raw + taxonomy.

Usage (repo root):
    python models/placement_extras.py tune --target=treecap120     # global, ~5 min
    python models/placement_extras.py test --target=treecap120 --theta=<t> --k=<k>
Add --marker=full for the long 12S region (tune again for it).
Needs: data/global_tree_distances.npz, data/global_taxonomy.csv, data/kmer_features_k3to6.npz,
       models/pretrained_<target>[_s<N>].npz, outputs/embeddings/embeddings_ft_<target>...
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.kmers import kmer_profile  # noqa: E402
from models.pretrain import load_encoder, load_global, marker_suffix, parse_args, seed_suffix  # noqa: E402

DATA, EMB, OUT = ROOT / "data", ROOT / "outputs" / "embeddings", ROOT / "outputs" / "results"
SEEDS = [0, 1, 2, 3, 4]
KS = [1, 3, 5, 10]                        # k = 1 is the plain nearest species


def dist(A, B):
    return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T, 0))


def centres(V, species):
    sps = np.unique(species)
    return sps, np.stack([V[species == s].mean(0) for s in sps])


def scaled_dist(Vq, C):
    """query -> centre distances divided by the median distance between centres."""
    return dist(Vq, C) / np.median(dist(C, C)[np.triu_indices(len(C), 1)])


class Tree:
    def __init__(self):
        z = np.load(DATA / "global_tree_distances.npz")
        self.row = {s: i for i, s in enumerate(z["species"])}
        self.age = z["D"] * float(z["scale"]) / 2          # split age in My

    def ages(self, a, b):
        return self.age[[self.row[x] for x in a], [self.row[x] for x in b]]

    def sub(self, sps):
        i = [self.row[x] for x in sps]
        return self.age[np.ix_(i, i)]


def place(sps, Draw, Dmod, tree_sub, theta=None, k=1):
    """Guessed species per query. theta: gate (raw alone if nearest raw < theta).
    k > 1: tree medoid of the k nearest species under raw + model."""
    D = Draw + Dmod
    if k == 1:
        pick = D.argmin(1)
    else:
        nn = np.argsort(D, axis=1)[:, :k]
        pick = np.empty(len(D), int)
        for q in range(len(D)):
            w = 1 / np.maximum(D[q, nn[q]], 1e-9)
            cost = (tree_sub[np.ix_(nn[q], nn[q])] * w[None, :]).sum(1)
            pick[q] = nn[q][cost.argmin()]
    if theta is not None:
        close = Draw.min(1) < theta
        pick = np.where(close, Draw.argmin(1), pick)
    return sps[pick]


def scores(pred, true, family, genus, tree, easy):
    fam = np.array([family.get(p) for p in pred], dtype=object) == np.array([family.get(t) for t in true],
                                                                            dtype=object)
    fam &= np.array([family.get(t) is not None for t in true])
    gen = np.array([genus[p] for p in pred]) == np.array([genus[t] for t in true])
    age = tree.ages(pred, true)
    return dict(family=fam.mean(), family_easy=fam[easy].mean(), genus_congener=gen[~easy].mean(),
                tree_error=age.mean(), tree_error_easy=age[easy].mean())


# ------------------------------------------------------------------ tuning on global data
def tune(target, marker):
    mk = marker_suffix(marker)
    g = load_global(marker)
    X = np.array([kmer_profile(s) for s in g.teleo])
    tree = Tree()
    tax = pd.read_csv(DATA / "global_taxonomy.csv").set_index("species_name")
    family = {s: f for s, f in tax["Family"].items() if pd.notna(f)}
    genus = {s: s.split()[0] for s in g.species_name.unique()}

    rng = np.random.default_rng(0)                   # same held-out species as models/pretrain.py
    species = np.array(sorted(g.species_name.unique()))
    val_sp = set(rng.choice(species, size=len(species) // 10, replace=False))
    is_val = g.species_name.isin(val_sp).values
    ref_sp = g.species_name.values[~is_val]
    q_sp = g.species_name.values[is_val]
    ref_genera = {genus[s] for s in ref_sp}
    easy = np.array([genus[s] not in ref_genera for s in q_sp])
    print(f"tuning set: {is_val.sum()} held-out sequences ({len(val_sp)} species, {easy.mean():.0%} without "
          f"a congener) | reference: {(~is_val).sum()} sequences")

    sps, Craw = centres(X[~is_val], ref_sp)
    Draw = scaled_dist(X[is_val], Craw)
    tsub = tree.sub(sps)
    thetas = [None] + list(np.round(np.quantile(Draw.min(1), [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]), 4))

    rows = []
    for seed in SEEDS:
        f = ROOT / "models" / f"pretrained_{target}{mk}{seed_suffix(seed)}.npz"
        if not f.exists():
            continue
        enc = load_encoder(f)
        _, Cm = centres(enc.embed(X[~is_val]), ref_sp)
        Dm = scaled_dist(enc.embed(X[is_val]), Cm)
        for th in thetas:
            rows.append(dict(seed=seed, rule="gate", value=th if th is not None else -1,
                             **scores(place(sps, Draw, Dm, tsub, theta=th), q_sp, family, genus, tree, easy)))
        for k in KS:
            rows.append(dict(seed=seed, rule="medoid", value=k,
                             **scores(place(sps, Draw, Dm, tsub, k=k), q_sp, family, genus, tree, easy)))
        print(f"seed {seed} done", flush=True)
    rows.append(dict(seed=-1, rule="raw only", value=0,
                     **scores(sps[Draw.argmin(1)], q_sp, family, genus, tree, easy)))
    df = pd.DataFrame(rows)
    t = df.groupby(["rule", "value"]).mean(numeric_only=True).drop(columns="seed")
    t["balanced"] = (t.family + t.genus_congener) / 2
    t.round(4).to_csv(OUT / f"placement_extras_tune_{target}{mk}.csv")
    print("\nglobal held-out species (mean over seeds); gate value -1 = no gate (plain raw + model)")
    print(t.round(3).to_string())
    gate = t.loc["gate"]
    best_theta = gate["balanced"].idxmax()
    med = t.loc["medoid"]
    ok = med[med.family >= med.loc[1, "family"] - 0.01]          # k = 1 always qualifies
    best_k = ok["tree_error"].idxmin()
    print(f"\nchosen gate theta = {best_theta}  (highest balanced accuracy; -1 means no gate)")
    print(f"chosen medoid k   = {best_k}  (lowest tree error without losing > 1 point of family; "
          "1 means plain nearest species)")
    print(f"next: python models/placement_extras.py test --target={target} --theta={best_theta} --k={best_k}"
          + (" --marker=full" if marker == "full" else ""))


# ------------------------------------------------------------------ one test on French Guiana
def test(target, marker, theta, k):
    mk = marker_suffix(marker)
    clean = pd.read_csv(DATA / "guiana_teleo_clean.csv")
    clean["seq_id"] = clean["Specimen identification code"].astype(str)
    clean = clean[clean.in_tree].drop_duplicates("seq_id").set_index("seq_id")
    info = clean.drop_duplicates("species_name").set_index("species_name")
    family, genus = info["Family"].to_dict(), info["Genus"].to_dict()
    kf = np.load(DATA / ("kmer_features_full12S.npz" if marker == "full" else "kmer_features_k3to6.npz"),
                 allow_pickle=True)
    X = pd.DataFrame(kf["X"], index=kf["seq_id"].astype(str))
    splits = pd.concat([pd.read_csv(DATA / "splits.csv"), pd.read_csv(DATA / "splits_extra.csv")])
    splits["seq_id"] = splits.seq_id.astype(str)
    tree = Tree()
    th = None if theta < 0 else theta
    methods = {"raw+model": dict(theta=None, k=1), "gate": dict(theta=th, k=1),
               "medoid": dict(theta=None, k=k), "gate+medoid": dict(theta=th, k=k)}

    rows = []
    for sid in sorted(splits.split_id.unique()):
        s = splits[splits.split_id == sid]
        ref = s[s.role == "reference"].seq_id.values
        nov = s[s.role == "test_novel"]
        test_ids = nov.seq_id.values
        easy = (nov.difficulty == "easy").values
        rsp, true = clean.loc[ref, "species_name"].values, clean.loc[test_ids, "species_name"].values
        sps, Craw = centres(X.loc[ref].values, rsp)
        Draw = scaled_dist(X.loc[test_ids].values, Craw)
        tsub = tree.sub(sps)
        per = {}
        for seed in SEEDS:
            V = {}
            for name, model in [("model", target), ("taxonomy", "taxonomy")]:
                f = EMB / f"embeddings_ft_{model}{mk}{seed_suffix(seed)}_split{sid}.npz"
                if f.exists():
                    z = np.load(f, allow_pickle=True)
                    V[name] = pd.DataFrame(z["emb"], index=z["seq_id"].astype(str))
            if len(V) < 2:
                continue
            _, Cm = centres(V["model"].loc[ref].values, rsp)
            Dm = scaled_dist(V["model"].loc[test_ids].values, Cm)
            _, Ct = centres(V["taxonomy"].loc[ref].values, rsp)
            Dt = scaled_dist(V["taxonomy"].loc[test_ids].values, Ct)
            for name, kw in methods.items():
                per.setdefault(name, []).append(
                    scores(place(sps, Draw, Dm, tsub, **kw), true, family, genus, tree, easy))
            per.setdefault("raw+taxonomy", []).append(
                scores(place(sps, Draw, Dt, tsub), true, family, genus, tree, easy))
        r = {"split": sid, "n_seeds": len(per["raw+model"])}
        for name, lst in per.items():
            for o in lst[0]:
                r[f"{name}|{o}"] = np.mean([x[o] for x in lst])
        rows.append(r)
        print(f"split {sid:2d} done", flush=True)
    df = pd.DataFrame(rows).set_index("split")
    df.round(4).to_csv(OUT / f"placement_extras_test_{target}{mk}.csv")

    outs = ["family", "family_easy", "genus_congener", "tree_error", "tree_error_easy"]
    names = ["raw+model", "gate", "medoid", "gate+medoid", "raw+taxonomy"]
    print(f"\n{target}{mk} | theta {theta} | k {k} | {len(df)} splits | seeds {int(df.n_seeds.min())}")
    print("mean over splits (tree error in My, lower = better)")
    print(pd.DataFrame({n: [df[f"{n}|{o}"].mean() for o in outs] for n in names}, index=outs).round(3).to_string())
    for a, b in [("gate", "raw+model"), ("medoid", "raw+model"), ("gate+medoid", "raw+model"),
                 ("gate", "raw+taxonomy"), ("medoid", "raw+taxonomy"), ("gate+medoid", "raw+taxonomy")]:
        print(f"\n{a} vs {b}")
        for o in outs:
            x, y = df[f"{a}|{o}"], df[f"{b}|{o}"]
            d = x - y
            better = (d < 0) if o.startswith("tree_error") else (d > 0)
            p = wilcoxon(x, y).pvalue if (d != 0).sum() > 1 else 1.0
            print(f"  {o:15s} {x.mean():.3f} vs {y.mean():.3f} | diff {d.mean():+.3f} "
                  f"| better in {better.sum()}/{len(d)} | p = {p:.3f}")
    print(f"\nsaved outputs/results/placement_extras_test_{target}{mk}.csv")


def main():
    args, opts = parse_args(sys.argv[1:])
    target, marker = opts.get("target", "treecap120"), opts.get("marker", "teleo")
    OUT.mkdir(parents=True, exist_ok=True)
    if args[:1] == ["tune"]:
        tune(target, marker)
    elif args[:1] == ["test"]:
        test(target, marker, float(opts["theta"]), int(float(opts["k"])))
    else:
        raise SystemExit("usage: python models/placement_extras.py tune|test --target=treecap120 [...]")


if __name__ == "__main__":
    main()
