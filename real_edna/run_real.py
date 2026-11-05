#!/usr/bin/env python3
"""
Run our open-set pipeline on REAL water samples (Cilleros et al. 2019, French Guiana, Teleo),
after real_edna/process_reads.py turned the raw reads into DNA sequence variants ("ZOTUs").

Reference library = OUR French Guiana library (data/guiana_teleo_clean.csv, in-tree sequences;
the same sequences as data/pairwise_identity_teleo.npz). The study's own reference
(data/real_edna/Reference_sequence.fasta) is NOT used by the pipeline; it only gives a proxy truth.

RULES, FIXED BEFORE RUNNING
  Score of a ZOTU = 1 - VSEARCH identity to its closest library sequence (B2 score; global
  alignment, --iddef 2, exhaustive search; ties -> species with most tied hits, then alphabetical).
  Detection rules (all set on our library only, never on the water samples):
    vsearch97   new if identity < 97%                                   (common practice)
    b2_f1       cut-off with the best F1 on 5 pseudo-splits of the library (baselines/common.py,
                seeds pseudo_rng(0, r)), as B2
    conf05/10   conformal, alpha 0.05 / 0.10: calibration = pseudo-known queries of the same
                5 pseudo-splits, each species once per split, splits averaged (models/conformal_any.py)
  Placement of a ZOTU called new: closest sequence in our library + the global MitoFish library
    (same filter as models/pretrain.load_global: no French Guiana species); a global hit is used only if strictly
    closer than the best local one (as models/global_library.py) -> genus and family.
  Grouping of ZOTUs called new (conf10): calibrated barcode gap (clustering/gap_grouping.py rule) on
    VSEARCH identity: linkage (single / average) and cut-off chosen on our library (ARI vs species).
  Proxy truth (study's reference, independent of our library): a ZOTU whose best hit in the study's
    reference has identity >= 99% gets that species X, if X is a plain binomial (Genus species; names that
    are only a genus/family or carry aff./cf./sp are left unlabelled, because they cannot be matched
    to our library's names; this rule was added after seeing that such names exist, before the final run). It is "known" if X is in our library,
    "new for us" if not. ZOTUs without a >= 99% hit have no proxy label (reported, not scored).
  Net-fishing check: for each site, the species caught by nets (data/real_edna/traditional_result.txt)
    that are NOT in our library. A ZOTU called new at a site "fits the nets" if its placed family
    (or genus) is the family (genus) of one of those net-caught missing species at that site.
  Outcomes: on proxy-labelled ZOTUs, share of known ZOTUs called new (false alarm) and of new ZOTUs
    called new (recall), per rule; agreement of our species lists with the study's per site; share of
    new calls that fit the nets; number of groups vs number of proxy-new species.

POST-HOC SENSITIVITY ANALYSIS (added AFTER seeing the results, reported beside, not instead of, the main one)
  Many ZOTUs called new were < 80% identical to every fish reference, so they may not be fish (other
  DNA, artifacts, or very divergent fish). Filter: keep ZOTUs with identity >= 0.80 to at least one of
  our library, the global MitoFish library, or the study's reference; detection counts, grouping (redone
  on the kept ZOTUs) and the net-fishing check are reported again on the kept ZOTUs.

Usage (repo root):  python real_edna/run_real.py          # uses every data/real_edna/processed/Run*/
Needs vsearch on the PATH (add it at the END: export PATH="$PATH:$HOME/phylo-tools/bin").
Output: data/real_edna/results/zotu_calls.csv, site_check.csv, summary.txt
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
DATA, REAL = ROOT / "data", ROOT / "data" / "real_edna"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines"))
from common import PSEUDO_REPEATS, pseudo_rng, pseudo_split, tune_threshold  # noqa: E402

ALPHAS = (0.05, 0.10)
FISH_MIN = 0.80   # post-hoc sensitivity filter (see header)
FISH_SWEEP = (0.70, 0.75, 0.80, 0.85, 0.90)   # same filter at other cut-offs: result must not hinge on 0.80


def vsearch():
    v = os.environ.get("VSEARCH") or shutil.which("vsearch")
    if not v:
        sys.exit('vsearch not found: export PATH="$PATH:$HOME/phylo-tools/bin"')
    return v


def norm(name):
    return " ".join(str(name).replace("_", " ").lower().split())


def write_fasta(path, names, seqs):
    with open(path, "w") as fh:
        for n, s in zip(names, seqs):
            fh.write(f">{n}\n{s.upper()}\n")


def best_hits(V, q_names, q_seqs, r_names, r_seqs, r_species, tmp):
    """Exhaustive VSEARCH best hit (B2 rule). Returns DataFrame q -> identity (0-1), species."""
    write_fasta(tmp / "q.fa", q_names, q_seqs)
    write_fasta(tmp / "r.fa", r_names, r_seqs)
    subprocess.run([V, "--usearch_global", tmp / "q.fa", "--db", tmp / "r.fa", "--id", "0.0",
                    "--maxaccepts", "0", "--maxrejects", "0", "--top_hits_only", "--minwordmatches", "0",
                    "--strand", "plus", "--minseqlength", "1", "--qmask", "none", "--dbmask", "none", "--userout", tmp / "h.tsv",
                    "--userfields", "query+target+id", "--output_no_hits", "--quiet", "--threads", "0"],
                   check=True)
    h = pd.read_csv(tmp / "h.tsv", sep="\t", header=None, names=["q", "t", "id"], dtype=str)
    h = h[h.t != "*"].copy()
    h["id"] = h.id.astype(float) / 100
    sp = dict(zip(r_names, r_species))
    h["sp"] = h.t.map(sp)
    out = {}
    for q, g in h.groupby("q"):
        c = g.sp.value_counts()
        out[q] = (g.id.max(), sorted(c.index[c == c.max()])[0])
    return pd.DataFrame([(q, *out.get(q, (0.0, None))) for q in q_names], columns=["q", "identity", "species"])


def conformal_p(cal, w, s):
    o = np.argsort(cal, kind="stable")
    c, ww = cal[o], w[o]
    tail = np.r_[np.cumsum(ww[::-1])[::-1], 0.0]
    i = np.searchsorted(c, s - 1e-12, side="left")
    return (1 + tail[i]) / (1 + w.sum())


def main():
    V = vsearch()
    runs = sorted((REAL / "processed").glob("Run[0-9]*"))
    runs = [r for r in runs if not r.name.endswith("_test")]
    if not runs:
        sys.exit("no data/real_edna/processed/Run*/ - run real_edna/process_reads.py first")
    # ---- ZOTUs of all runs (identical sequences across runs merged)
    zs, occ = {}, []
    for r in runs:
        name = None
        for line in open(r / "zotus.fasta"):
            if line.startswith(">"):
                name = line[1:].split(";")[0]
            else:
                zs[(r.name, name)] = line.strip().upper()
        s = pd.read_csv(r / "zotu_samples.csv")
        s["key"] = list(zip([r.name] * len(s), s.zotu))
        occ.append(s)
    occ = pd.concat(occ)
    seq_of = {k: v for k, v in zs.items()}
    uniq = sorted(set(seq_of.values()))
    sid = {s: f"Z{i + 1}" for i, s in enumerate(uniq)}
    occ["zid"] = occ.key.map(lambda k: sid[seq_of[k]])
    occ["code"] = occ["sample"].astype(str).str.replace(r"^Tlse_(\d+)$", r"Tlse\1", regex=True)
    site = pd.read_csv(REAL / "site_data.txt", sep="\t", encoding="utf-8-sig")
    occ["site"] = occ.code.map(dict(zip(site["Sample code"], site["site"])))
    occ = occ[occ.site.notna()]
    zids = sorted(occ.zid.unique(), key=lambda z: int(z[1:]))
    zseq = {v: k for k, v in sid.items()}
    print(f"{len(runs)} run(s): {len(zids)} distinct ZOTUs in {occ.site.nunique()} sites", flush=True)

    # ---- our library
    d = pd.read_csv(DATA / "guiana_teleo_clean.csv", dtype={"Specimen identification code": str})
    d = d[d["in_tree"]].rename(columns={"Specimen identification code": "seq_id"})
    lib = d.drop_duplicates("seq_id").set_index("seq_id").sort_index()
    lib["teleo"] = lib.teleo.str.strip().str.upper()
    tax = lib.drop_duplicates("species_name").set_index("species_name")[["Genus", "Family"]]
    our_species = {norm(s) for s in tax.index}
    z = np.load(DATA / "pairwise_identity_teleo.npz", allow_pickle=True)
    assert list(z["seq_id"]) == list(lib.index)
    ID = z["ID"].astype(float)
    pos = pd.Series(np.arange(len(lib)), index=lib.index)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # ---- detection score on our library
        hit = best_hits(V, zids, [zseq[x] for x in zids], list(lib.index), list(lib.teleo),
                        list(lib.species_name), tmp).set_index("q")
        score = 1 - hit.identity.values

        # ---- calibration on the library (pseudo-splits)
        ref = lib[["species_name", "teleo"]].copy()
        cal_s, cal_w, f1_s, f1_y = [], [], [], []
        for r in range(PSEUDO_REPEATS):
            pref, pq = pseudo_split(ref, pseudo_rng(0, r))
            M = ID[np.ix_(pos[pq.index].values, pos[pref.index].values)]
            s = 1 - M.max(1)
            f1_s.append(s); f1_y.append(pq.is_novel.values)
            kn = ~pq.is_novel.values
            cnt = pq.species_name[kn].map(pq.species_name[kn].value_counts()).to_numpy(float)
            cal_s.append(s[kn]); cal_w.append(1 / cnt / PSEUDO_REPEATS)
        cal_s, cal_w = np.concatenate(cal_s), np.concatenate(cal_w)
        t_f1, _ = tune_threshold(np.concatenate(f1_s), np.concatenate(f1_y))
        calls = {"vsearch97": hit.identity.values < 0.97 - 1e-9, "b2_f1": score >= t_f1}
        for a in ALPHAS:
            calls[f"conf{int(a * 100):02d}"] = conformal_p(cal_s, cal_w, score) <= a
        cutoffs = {"b2_f1 identity cut-off": 1 - t_f1}
        for a in ALPHAS:
            c = calls[f"conf{int(a * 100):02d}"]
            cutoffs[f"conf{int(a * 100):02d} identity cut-off"] = (1 - score[c].min()) if c.any() else np.nan

        # ---- placement: local + global
        # global MitoFish library, same filter as models/pretrain.load_global (no French Guiana species,
        # no sequence identical to a French Guiana one); not imported to avoid loading torch
        g = pd.read_csv(DATA / "global_teleo.csv")
        fg = set(pd.read_csv(DATA / "guiana_teleo_clean.csv")["teleo"].str.strip().str.lower())
        g = g[g.in_tree & ~g.in_guiana & ~g.teleo.isin(fg)].reset_index(drop=True)
        gt = pd.read_csv(DATA / "global_taxonomy.csv").drop_duplicates("species_name").set_index("species_name")
        gnames = [f"g{i}" for i in range(len(g))]
        ghit = best_hits(V, zids, [zseq[x] for x in zids], gnames, list(g.teleo.str.upper()),
                         list(g.species_name), tmp).set_index("q")
        use_g = ghit.identity.values > hit.identity.values + 1e-9
        place_sp = np.where(use_g, ghit.species.values, hit.species.values)
        fam_of = {**gt["Family"].to_dict(), **tax["Family"].to_dict()}
        gen_of = {s: str(s).split()[0] for s in place_sp}
        # ---- proxy truth from the study's reference
        names, seqs, cur = [], [], None
        for line in open(REAL / "Reference_sequence.fasta"):
            if line.startswith(">"):
                cur = line[1:].split()[0]
                names.append(cur); seqs.append("")
            else:
                seqs[-1] += line.strip()
        st_sp = [norm(n.rsplit("_", 1)[0]) for n in names]
        phit = best_hits(V, zids, [zseq[x] for x in zids], names, seqs, st_sp, tmp).set_index("q")
        # ---- pairwise identity among ZOTUs (for grouping)
        write_fasta(tmp / "z.fa", zids, [zseq[x] for x in zids])
        subprocess.run([V, "--allpairs_global", tmp / "z.fa", "--acceptall", "--minseqlength", "1", "--userout", tmp / "ap.tsv",
                        "--userfields", "query+target+id", "--quiet", "--threads", "0"], check=True)
        ap = pd.read_csv(tmp / "ap.tsv", sep="\t", header=None, names=["a", "b", "id"])

    tab = pd.DataFrame({"zotu": zids, "seq": [zseq[x] for x in zids], "identity": hit.identity.values,
                        "nearest_species": hit.species.values, "placed_species": place_sp,
                        "placed_on_global": use_g,
                        "placed_family": [fam_of.get(s) for s in place_sp],
                        "placed_genus": [gen_of[s] for s in place_sp],
                        "study_identity": phit.identity.values, "study_species": phit.species.values})
    for k, c in calls.items():
        tab[k] = c
    binomial = tab.study_species.fillna("").map(
        lambda x: len(x.split()) == 2 and not ({"aff.", "cf.", "sp", "sp."} & set(x.split())))
    lab = np.where((tab.study_identity >= 0.99) & binomial,
                   np.where(tab.study_species.isin(our_species), "known", "new"), "none")
    tab["proxy"] = lab
    tab["reads"] = tab.zotu.map(occ.groupby("zid").reads.sum())
    tab["n_sites"] = tab.zotu.map(occ.groupby("zid").site.nunique())
    tab["best_fish_identity"] = np.maximum.reduce([hit.identity.values, ghit.identity.values, phit.identity.values])
    tab["fish_like"] = tab.best_fish_identity >= FISH_MIN

    # ---- grouping of conf10 new calls
    nov = tab[tab.conf10].zotu.tolist()
    lib_sp = lib.species_name.to_numpy(dtype=object)
    best = None
    for meth in ["single", "average"]:
        Zl = linkage(squareform(1 - ID, checks=False), meth)
        for t in np.round(np.arange(0, 0.1501, 0.0025), 4):
            ari = adjusted_rand_score(lib_sp, fcluster(Zl, t, "distance"))
            if best is None or ari > best[2] + 1e-12:
                best = (meth, t, ari)
    def group(nov):
        if len(nov) < 2:
            return np.ones(len(nov), int)
        ix = {z_: i for i, z_ in enumerate(nov)}
        Dn = np.ones((len(nov), len(nov)))
        apn = ap[ap.a.isin(ix) & ap.b.isin(ix)]
        for a_, b_, i_ in zip(apn.a, apn.b, apn.id):
            Dn[ix[a_], ix[b_]] = Dn[ix[b_], ix[a_]] = 1 - i_ / 100
        np.fill_diagonal(Dn, 0)
        return fcluster(linkage(squareform(Dn, checks=False), best[0]), best[1], "distance")

    grp = group(nov)
    tab["group"] = np.nan
    tab.loc[tab.zotu.isin(nov), "group"] = grp
    nov_f = tab[tab.conf10 & tab.fish_like].zotu.tolist()       # sensitivity: regrouped after the filter
    grp_f = group(nov_f)
    tab["group_filtered"] = np.nan
    tab.loc[tab.zotu.isin(nov_f), "group_filtered"] = grp_f

    # ---- net-fishing check per site
    net = pd.read_csv(REAL / "traditional_result.txt", sep="\t").set_index("Species")
    asg = pd.read_csv(REAL / "assignment_result.txt", sep="\t").set_index("Taxonomic_unit_assigned")
    asg = asg.apply(lambda c: pd.to_numeric(c.astype(str).str.replace(" ", ""), errors="coerce")).fillna(0)
    gen2fam = {**gt.assign(G=gt.index.str.split().str[0]).drop_duplicates("G").set_index("G")["Family"].to_dict(),
               **lib.drop_duplicates("Genus").set_index("Genus")["Family"].to_dict()}
    fam_any = lambda sp: fam_of.get(sp) or gen2fam.get(str(sp).split()[0])
    tab["placed_family"] = [fam_any(x) for x in tab.placed_species]
    def site_rows(tb):
      rows = []
      for s_, g_ in occ[occ.zid.isin(tb.zotu)].groupby("site"):
        zt = tb.set_index("zotu").loc[g_.zid.unique()]
        caught = [x for x in net.index if s_ in net and net.loc[x, s_] > 0]
        missing = [x for x in caught if norm(x) not in our_species]
        miss_fam = {fam_any(x) for x in missing} - {None}
        miss_gen = {x.split()[0] for x in missing}
        ours = {norm(x) for x in zt[~zt.conf10].nearest_species}
        sp_level = [x for x in asg.index if "(" not in x and len(x.split()) >= 2]
        theirs = {norm(x) for x in sp_level if s_ in asg and asg.loc[x, s_] > 0}
        for rule in calls:
            nc, kc = zt[zt[rule]], zt[~zt[rule]]
            rows.append(dict(site=s_, rule=rule, zotus=len(zt), called_new=len(nc),
                             new_fit_family=int(nc.placed_family.isin(miss_fam).sum()),
                             new_fit_genus=int(nc.placed_genus.isin(miss_gen).sum()),
                             known_fit_family=int(kc.placed_family.isin(miss_fam).sum()),
                             known_fit_genus=int(kc.placed_genus.isin(miss_gen).sum()),
                             net_caught=len(caught), net_missing_from_library=len(missing),
                             our_known_species=len(ours) if rule == "conf10" else np.nan,
                             study_species=len(theirs) if rule == "conf10" else np.nan,
                             species_overlap=len(ours & theirs) if rule == "conf10" else np.nan))
      return pd.DataFrame(rows)

    sc = site_rows(tab)
    sc_f = site_rows(tab[tab.fish_like])

    out = REAL / "results"
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "zotu_calls.csv", index=False)
    sc.to_csv(out / "site_check.csv", index=False)

    lines = [f"{len(runs)} run(s), {occ.site.nunique()} sites, {len(tab)} distinct ZOTUs",
             "identity cut-offs: " + ", ".join(f"{k} {v:.4f}" for k, v in cutoffs.items()),
             f"proxy labels (study reference >= 99%): known {int((lab == 'known').sum())}, "
             f"new for us {int((lab == 'new').sum())}, no label {int((lab == 'none').sum())}", "",
             "DETECTION on proxy-labelled ZOTUs (false alarm = known ZOTUs called new; recall = new ZOTUs called new)"]
    for rule in calls:
        k_, n_ = tab[tab.proxy == "known"], tab[tab.proxy == "new"]
        lines.append(f"  {rule:9s} called new {int(tab[rule].sum()):4d} of {len(tab)} | false alarm "
                     f"{k_[rule].mean():.3f} ({int(k_[rule].sum())}/{len(k_)}) | recall {n_[rule].mean():.3f} "
                     f"({int(n_[rule].sum())}/{len(n_)}) | unlabelled called new {int(tab[tab.proxy == 'none'][rule].sum())}")
    lines += ["", "GROUPING CHECK on proxy-new ZOTUs called new (conf10): "
              f"{len(tab[tab.conf10 & (tab.proxy == 'new')])} ZOTUs, "
              f"{tab[tab.conf10 & (tab.proxy == 'new')].study_species.nunique()} species, "
              f"{tab[tab.conf10 & (tab.proxy == 'new')].group.nunique()} groups"]
    lines += ["", "NET-FISHING CHECK (share of ZOTU x site occurrences whose placed family / genus matches a "
              "net-caught species missing from our library at that site; new calls should fit more than known calls)"]
    for rule in calls:
        g_ = sc[sc.rule == rule]
        tot = g_.called_new.sum()
        kt = (g_.zotus - g_.called_new).sum()
        lines.append(f"  {rule:9s} {tot:4d} new occurrences: fit family {g_.new_fit_family.sum() / max(tot, 1):.2f}, "
                     f"genus {g_.new_fit_genus.sum() / max(tot, 1):.2f}  | control, {kt} known-called occurrences: "
                     f"family {g_.known_fit_family.sum() / max(kt, 1):.2f}, genus {g_.known_fit_genus.sum() / max(kt, 1):.2f}")
    g_ = sc[sc.rule == "conf10"]
    lines += ["", f"SPECIES LISTS (known calls, conf10) vs the study's own assignments: "
              f"{int(g_.species_overlap.sum())} shared of {int(g_.our_known_species.sum())} ours / "
              f"{int(g_.study_species.sum())} theirs (summed over sites)",
              f"GROUPING (conf10 new calls): {len(nov)} ZOTUs -> {len(set(grp)) if len(nov) else 0} groups "
              f"(linkage {best[0]}, cut-off {best[1]}); proxy-new species among them: "
              f"{tab[tab.conf10 & (tab.proxy == 'new')].study_species.nunique()}",
              f"placed with a global hit: {int(tab[tab.conf10].placed_on_global.sum())} of {len(nov)} new calls"]
    # ---- post-hoc sensitivity analysis: fish-reference filter
    f_ = tab[tab.fish_like]
    lines += ["", f"SENSITIVITY (post hoc, added after seeing the results): keep only ZOTUs with identity >= "
              f"{FISH_MIN:.2f} to at least one fish reference (our library, global MitoFish, study reference)",
              f"  kept {len(f_)} of {len(tab)} ZOTUs, {f_.reads.sum() / tab.reads.sum():.3f} of reads; "
              f"removed {len(tab) - len(f_)} (removed ones called new by conf10: {int(tab[~tab.fish_like].conf10.sum())})",
              "  proxy-labelled ZOTUs removed by the filter: "
              f"{int((~tab.fish_like & (tab.proxy != 'none')).sum())} (detection table above is unchanged if 0)"]
    for rule in calls:
        lines.append(f"  {rule:9s} called new {int(f_[rule].sum()):4d} of {len(f_)} "
                     f"(unlabelled {int(f_[f_.proxy == 'none'][rule].sum())})")
    lines.append(f"  GROUPING redone after the filter (conf10): {len(nov_f)} ZOTUs -> "
                 f"{len(set(grp_f)) if len(nov_f) else 0} groups; proxy-new: "
                 f"{f_[f_.conf10 & (f_.proxy == 'new')].study_species.nunique()} species in "
                 f"{f_[f_.conf10 & (f_.proxy == 'new')].group_filtered.nunique()} groups")
    for rule in calls:
        g_ = sc_f[sc_f.rule == rule]
        tot, kt = g_.called_new.sum(), (g_.zotus - g_.called_new).sum()
        lines.append(f"  net check {rule:9s} new fit genus {g_.new_fit_genus.sum() / max(tot, 1):.2f} ({tot} occ.) "
                     f"vs known {g_.known_fit_genus.sum() / max(kt, 1):.2f} ({kt} occ.)")
    sc_f.to_csv(out / "site_check_filtered.csv", index=False)
    # ---- the same filter at several cut-offs (80% is our own choice, not a published standard)
    sw = []
    for c in FISH_SWEEP:
        k_ = tab[tab.best_fish_identity >= c]
        nv = k_[k_.conf10].zotu.tolist()
        gf = pd.Series(group(nv), index=nv)
        pn = k_[k_.conf10 & (k_.proxy == "new")]
        sf = site_rows(k_)
        row = dict(cutoff=c, zotus_kept=len(k_), read_share=round(k_.reads.sum() / tab.reads.sum(), 3),
                   labelled_removed=int(((tab.best_fish_identity < c) & (tab.proxy != "none")).sum()),
                   conf10_new=len(nv), conf10_groups=gf.nunique(),
                   checkable_new_species=pn.study_species.nunique(), checkable_new_groups=gf[pn.zotu].nunique())
        for rule in ("conf10", "vsearch97"):
            g_ = sf[sf.rule == rule]
            tot, kt = g_.called_new.sum(), (g_.zotus - g_.called_new).sum()
            row[f"{rule}_net_genus_new"] = round(g_.new_fit_genus.sum() / max(tot, 1), 3)
            row[f"{rule}_net_genus_known"] = round(g_.known_fit_genus.sum() / max(kt, 1), 3)
        sw.append(row)
    sw = pd.DataFrame(sw)
    sw.to_csv(out / "fish_filter_sweep.csv", index=False)
    lines += ["", "FILTER CUT-OFF SWEEP (does the result depend on choosing 80%?)",
              "  cut-off | ZOTUs kept | reads | labelled removed | conf10 new -> groups | checkable new species/groups"
              " | net genus new vs known (conf10; 97%)"]
    for r_ in sw.itertuples():
        lines.append(f"  {r_.cutoff:.2f}    | {r_.zotus_kept:4d}       | {r_.read_share:.3f} | {r_.labelled_removed:3d}"
                     f"              | {r_.conf10_new:3d} -> {r_.conf10_groups:3d}            | "
                     f"{r_.checkable_new_species}/{r_.checkable_new_groups}"
                     f"                         | {r_.conf10_net_genus_new:.2f} vs {r_.conf10_net_genus_known:.2f}; "
                     f"{r_.vsearch97_net_genus_new:.2f} vs {r_.vsearch97_net_genus_known:.2f}")
    tab.to_csv(out / "zotu_calls.csv", index=False)
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}/zotu_calls.csv, site_check.csv, summary.txt")


if __name__ == "__main__":
    main()
