"""Step 7: analysis tables, figures and a generated report (Mac only).

Reads the join outputs (analysis/version_codes.csv, quality.json,
judge_reliability.csv, agreement.csv), the coded notes records, the highest
codebook revisions with their families, and codebook/literature.yml if present.
Writes CSV tables, PNG figures under analysis/figures/, analysis/summary.json
(every headline number, so a results page can be built from it) and
analysis/REPORT.md generated from the summary.

pandas and matplotlib are imported lazily: the rest of the package runs inside
the inference container without them.
"""

import json
import math
from collections import Counter

from .common import (
    atomic_write_text,
    codebook_families,
    latest_codebook,
    log,
    read_json,
    read_yaml,
    require,
    write_json,
)
from .corpus import load_versions
from .steps import idea_of, load_records

STATES = [
    "introduced",
    "kept",
    "dropped",
    "tested_and_refused",
    "refused_untested",
    "considered",
    "absent",
]
# Okabe-Ito, colour-blind safe; absent is a light grey
STATE_COLOURS = {
    "introduced": "#009E73",
    "kept": "#0072B2",
    "dropped": "#D55E00",
    "tested_and_refused": "#CC79A7",
    "refused_untested": "#E69F00",
    "considered": "#56B4E9",
    "absent": "#EEEEEE",
}
LINE_COLOURS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
]
NOTES_NONE = ("no_mechanism", "no_record", "", None)


# --- loading -----------------------------------------------------------------


class Context:
    """Everything the tables need, loaded once."""

    def __init__(self, run, cfg):
        import pandas as pd

        self.run, self.cfg = run, cfg
        adir = run.dir / "analysis"
        vc_path = adir / "version_codes.csv"
        if not vc_path.exists():
            raise SystemExit(f"no {vc_path}; run join first")
        self.vc = pd.read_csv(vc_path, keep_default_na=False, na_values=[""])
        self.quality = read_json(adir / "quality.json")
        self.versions = {v["stem"]: v for v in load_versions(run)}
        self.records = load_records(run, coded=True)
        self.families = {}
        self.family_defs = {}
        self.code_info = {}
        for kind in ("what", "why", "basis"):
            rev, path, codes = latest_codebook(run, kind)
            defs = codebook_families(path) or codebook_families(
                run.dir / "codebook" / f"{kind}.v1.yml"
            )
            fams = {c["name"]: c.get("family") or "none" for c in codes}
            if kind == "what":
                npath = run.dir / "codebook" / "never_implemented.yml"
                nrevs = [
                    p
                    for p in sorted(
                        (run.dir / "codebook").glob("never_implemented.v*.yml")
                    )
                    if p.stem.split(".v")[-1].isdigit()
                ]
                if nrevs:
                    npath = max(nrevs, key=lambda p: int(p.stem.split(".v")[-1]))
                if npath.exists():
                    never = read_yaml(npath).get("codes") or []
                    fams.update({c["name"]: c.get("family") or "none" for c in never})
                    defs = {**codebook_families(npath), **defs}
                    codes = codes + never
            self.families[kind] = fams
            self.family_defs[kind] = defs
            self.code_info[kind] = {c["name"]: c for c in codes}
        lit = run.dir / "codebook" / "literature.yml"
        self.literature = _literature(read_yaml(lit)) if lit.exists() else None
        self.min_agreement = cfg.get("judge", {}).get("min_agreement", 0.8)
        self.reliability = []
        rel = adir / "judge_reliability.csv"
        if rel.exists():
            self.reliability = pd.read_csv(rel).to_dict("records")
        jr = (self.quality.get("judge_reliability") or {}).get("unreliable_codes")
        if jr:
            self.unreliable = set(jr["codes"])
        else:
            self.unreliable = {
                r["code"]
                for r in self.reliability
                if r["code"] != "__overall__"
                and isinstance(r.get("agreement"), float)
                and r["agreement"] < self.min_agreement
            }


def _literature(doc):
    """{code: [publication]} from literature.yml (a mapping, or `codes:` list)."""
    if not doc:
        return {}
    if isinstance(doc, dict) and "codes" in doc and isinstance(doc["codes"], list):
        return {c["code"]: c.get("publications") or [] for c in doc["codes"]}
    if isinstance(doc, dict):
        return {k: (v or []) for k, v in doc.items() if isinstance(v, list)}
    return {}


def _write_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, df.to_csv(index=False))


def _num(x):
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _r(x, k=4):
    x = _num(x)
    return None if x is None else round(x, k)


# --- 1. idea events ----------------------------------------------------------


def idea_state(in_rule, event, notes_status, tested):
    """One label per version and code from the rule codes and the notes."""
    if in_rule:
        return "introduced" if event == "introduced" else "kept"
    if event == "dropped":
        return "dropped"
    if notes_status == "rejected":
        return "tested_and_refused" if tested else "refused_untested"
    if notes_status == "considered":
        return "considered"
    return "absent"


def idea_events(ctx):
    vc = ctx.vc.copy()
    z = vc.drop_duplicates("version").set_index("version")["score_z"].to_dict()
    vc["notes_status_label"] = vc["notes_status"].where(
        ~vc["notes_status"].isin(NOTES_NONE[:2]), "none"
    )
    vc["notes_status_label"] = vc["notes_status_label"].fillna("none")
    in_rule = vc["in_rule"].fillna(0).astype(float) == 1
    tested = vc["notes_tested"].fillna(0).astype(float) == 1
    vc["family"] = vc["what_code"].map(ctx.families["what"]).fillna("none")
    vc["score_z_parent"] = vc["parent"].map(z)
    vc["score_z_change"] = vc["score_z"] - vc["score_z_parent"]
    vc["code_unreliable"] = vc["what_code"].isin(ctx.unreliable)
    vc["idea_state"] = [
        idea_state(r, e, n, t)
        for r, e, n, t in zip(in_rule, vc["event"], vc["notes_status_label"], tested)
    ]
    has_record = vc["notes_status"] != "no_record"
    vc["said_not_done"] = (vc["notes_status_label"] == "adopted") & ~in_rule
    vc["done_not_said"] = in_rule & (vc["notes_status_label"] == "none") & has_record
    cols = [
        "campaign",
        "lineage",
        "version",
        "generation",
        "parent",
        "score",
        "score_status",
        "score_z",
        "score_z_change",
        "what_code",
        "family",
        "code_source",
        "in_rule",
        "event",
        "notes_status_label",
        "notes_tested",
        "notes_relation",
        "notes_source_versions",
        "code_unreliable",
        "idea_state",
        "said_not_done",
        "done_not_said",
    ]
    out = vc[cols].rename(
        columns={
            "notes_status_label": "notes_status",
            "notes_tested": "tested",
            "notes_relation": "relation_to_previous",
            "notes_source_versions": "source_version",
        }
    )
    counts = Counter(out["idea_state"])
    return out, {
        "rows": len(out),
        "versions": int(out["version"].nunique()),
        "codes": int(out["what_code"].nunique()),
        "idea_states": {s: counts.get(s, 0) for s in STATES},
        "said_not_done": int(out["said_not_done"].sum()),
        "done_not_said": int(out["done_not_said"].sum()),
    }


# --- 2. idea flow by generation ----------------------------------------------


def idea_flow(ev):
    import pandas as pd

    rows = []
    for (camp, gen, code), g in ev.groupby(["campaign", "generation", "what_code"]):
        n = g["version"].nunique()
        k = int((g["in_rule"].fillna(0) == 1).sum())
        st = Counter(g["idea_state"])
        rows.append(
            dict(
                level="code",
                campaign=camp,
                generation=gen,
                name=code,
                family=g["family"].iloc[0],
                n_versions=n,
                n_in_rule=k,
                share_in_rule=round(k / n, 4) if n else None,
                **{s: st.get(s, 0) for s in STATES},
            )
        )
    for (camp, gen, fam), g in ev.groupby(["campaign", "generation", "family"]):
        per_version = g.groupby("version")["in_rule"].apply(
            lambda s: bool((s.fillna(0) == 1).any())
        )
        n, k = len(per_version), int(per_version.sum())
        st = Counter(g["idea_state"])
        rows.append(
            dict(
                level="family",
                campaign=camp,
                generation=gen,
                name=fam,
                family=fam,
                n_versions=n,
                n_in_rule=k,
                share_in_rule=round(k / n, 4) if n else None,
                **{s: st.get(s, 0) for s in STATES},
            )
        )
    df = pd.DataFrame(rows)
    gens = sorted(ev["generation"].unique().tolist())
    return df, {"generations": [int(g) for g in gens], "rows": len(df)}


# --- 3. first seen and convergence -------------------------------------------


def first_seen(ev):
    import pandas as pd

    rows = []
    for (camp, code), g in ev.groupby(["campaign", "what_code"]):
        notes = g[g["notes_status"] != "none"]
        tested = g[g["tested"].fillna(0).astype(float) == 1]
        rule = g[g["in_rule"].fillna(0) == 1]
        any_ = g[(g["notes_status"] != "none") | (g["in_rule"].fillna(0) == 1)]
        re_refused = 0
        for _, lg in g.sort_values("generation").groupby("lineage"):
            rej = (lg["notes_status"] == "rejected").tolist()
            if any(rej):
                re_refused += sum(rej[rej.index(True) + 1 :])
        rows.append(
            dict(
                campaign=camp,
                what_code=code,
                family=g["family"].iloc[0],
                code_source=g["code_source"].iloc[0],
                first_in_notes=_min(notes["generation"]),
                first_tested=_min(tested["generation"]),
                first_in_rule=_min(rule["generation"]),
                lineages_in_campaign=g["lineage"].nunique(),
                lineages_with_rule=rule["lineage"].nunique(),
                lineages_with_any=any_["lineage"].nunique(),
                re_refusals=re_refused,
            )
        )
    df = pd.DataFrame(rows)
    conv = df[df["lineages_with_rule"] >= 2]
    return df, {
        "codes_in_rules_of_2plus_lineages": int(len(conv)),
        "max_independent_lineages": (
            conv.sort_values("lineages_with_rule", ascending=False)
            .head(5)[["campaign", "what_code", "lineages_with_rule"]]
            .to_dict("records")
        ),
        "re_refusals_total": int(df["re_refusals"].sum()),
    }


def _min(s):
    return int(s.min()) if len(s) else None


# --- 4. idea value -----------------------------------------------------------


def _stats(xs):
    xs = [float(x) for x in xs]
    n = len(xs)
    if not n:
        return dict(n=0, mean=None, se=None, share_positive=None)
    mean = sum(xs) / n
    se = None
    if n > 1:
        var = sum((x - mean) ** 2 for x in xs) / (n - 1)
        se = math.sqrt(var / n)
    return dict(
        n=n,
        mean=round(mean, 4),
        se=None if se is None else round(se, 4),
        share_positive=round(sum(x > 0 for x in xs) / n, 4),
    )


def idea_value(ev):
    import pandas as pd

    usable = ev[
        ev["event"].isin(["introduced", "dropped"])
        & ev["parent"].notna()
        & ev["score_z_change"].notna()
    ]
    rows = []
    for code, g in ev.groupby("what_code"):
        u = usable[usable["what_code"] == code]
        row = dict(
            what_code=code,
            family=g["family"].iloc[0],
            code_unreliable=bool(g["code_unreliable"].iloc[0]),
        )
        for event in ("introduced", "dropped"):
            allx = u[u["event"] == event]["score_z_change"].tolist()
            rel = [] if row["code_unreliable"] else allx
            s_all, s_rel = _stats(allx), _stats(rel)
            for k, v in s_rel.items():
                row[f"{event}_{k}"] = v
            row[f"{event}_small_n"] = s_rel["n"] < 5
            for k, v in s_all.items():
                row[f"{event}_{k}_unfiltered"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    ok = df[(df["introduced_n"] >= 5)].sort_values("introduced_mean", ascending=False)
    return df, {
        "events_used": int(len(usable)),
        "codes_with_5plus_introductions": int(len(ok)),
        "top_introductions": ok.head(3)[
            ["what_code", "introduced_n", "introduced_mean", "introduced_se"]
        ].to_dict("records"),
        "bottom_introductions": ok.tail(3)[
            ["what_code", "introduced_n", "introduced_mean", "introduced_se"]
        ].to_dict("records"),
    }


# --- 5. reasons by mechanism -------------------------------------------------


def why_by_what(ctx):
    import pandas as pd

    rows = []
    for rec in ctx.records.values():
        for m in rec["mechanisms"]:
            if not m.get("what_code"):
                continue
            for r in m["reasons"]:
                if not r.get("why_code"):
                    continue
                rows.append(
                    dict(
                        what_code=m["what_code"],
                        what_family=ctx.families["what"].get(m["what_code"], "none"),
                        why_code=r["why_code"],
                        why_family=ctx.families["why"].get(r["why_code"], "none"),
                        stance=r["stance"],
                    )
                )
    raw = pd.DataFrame(
        rows, columns=["what_code", "what_family", "why_code", "why_family", "stance"]
    )
    code = (
        raw.groupby(["what_code", "why_code", "stance"])
        .size()
        .reset_index(name="n")
        .assign(level="code")
    )
    fam = (
        raw.groupby(["what_family", "why_family", "stance"])
        .size()
        .reset_index(name="n")
        .assign(level="family")
        .rename(columns={"what_family": "what_code", "why_family": "why_code"})
    )
    df = pd.concat([fam, code], ignore_index=True)[
        ["level", "what_code", "why_code", "stance", "n"]
    ].rename(columns={"what_code": "what", "why_code": "why"})
    top = fam.groupby(["what_code", "why_code"])["n"].sum().sort_values()
    return (
        df,
        raw,
        {
            "reasons_coded": int(len(raw)),
            "for": int((raw["stance"] == "for").sum()),
            "against": int((raw["stance"] == "against").sum()),
            "top_family_pairs": [
                {"what_family": a, "why_family": b, "n": int(n)}
                for (a, b), n in top.tail(3)[::-1].items()
            ],
        },
    )


# --- 6. basis by generation --------------------------------------------------


def ref_kind(ref):
    if ref is None or ref == "":
        return "not_stated"
    r = str(ref).lower()
    if r == "none":
        return "none"
    if "pf-" in r:
        return "earlier_version"
    if "explor" in r or "finding" in r or "entry" in r:
        return "own_exploration"
    return "other"


def basis_by_generation(ctx):
    import pandas as pd

    rows = []
    for stem, rec in ctx.records.items():
        v = ctx.versions[stem]
        for m in rec["mechanisms"]:
            for r in m["reasons"]:
                bc = r.get("basis_code")
                rows.append(
                    dict(
                        campaign=v["campaign"],
                        generation=v["version"],
                        basis_family=(
                            ctx.families["basis"].get(bc, "none") if bc else "uncoded"
                        ),
                        evidence_ref=ref_kind(r.get("evidence_ref")),
                    )
                )
    raw = pd.DataFrame(
        rows, columns=["campaign", "generation", "basis_family", "evidence_ref"]
    )
    out = []
    for dim in ("basis_family", "evidence_ref"):
        g = raw.groupby(["campaign", "generation", dim]).size().reset_index(name="n")
        tot = g.groupby(["campaign", "generation"])["n"].transform("sum")
        g["share"] = (g["n"] / tot).round(4)
        g = g.rename(columns={dim: "value"}).assign(dimension=dim)
        out.append(g)
    df = pd.concat(out, ignore_index=True)[
        ["dimension", "campaign", "generation", "value", "n", "share"]
    ]
    return (
        df,
        raw,
        {
            "reasons": int(len(raw)),
            "basis_family_shares": (
                {
                    k: round(v / len(raw), 4)
                    for k, v in Counter(raw["basis_family"]).items()
                }
                if len(raw)
                else {}
            ),
            "evidence_ref_shares": (
                {
                    k: round(v / len(raw), 4)
                    for k, v in Counter(raw["evidence_ref"]).items()
                }
                if len(raw)
                else {}
            ),
        },
    )


# --- 7. theory ---------------------------------------------------------------


def _named(t):
    return t.get("name_ok", True)


def theory(ctx):
    import pandas as pd

    per_version = []
    for stem, v in ctx.versions.items():
        rec = ctx.records.get(stem)
        names = [t["name"] for t in (rec or {}).get("theory_mentions", []) if _named(t)]
        per_version.append(
            dict(
                campaign=v["campaign"],
                lineage=v["lineage"],
                version=stem,
                generation=v["version"],
                has_record=rec is not None,
                n_mentions=len(names),
                mentions=";".join(names),
            )
        )
    pv = pd.DataFrame(per_version)
    by_gen = (
        pv[pv["has_record"]]
        .groupby(["campaign", "generation"])
        .agg(versions=("version", "count"), with_mention=("n_mentions", "size"))
        .reset_index()
    )
    by_gen["with_mention"] = (
        pv[pv["has_record"]]
        .assign(m=lambda d: d["n_mentions"] > 0)
        .groupby(["campaign", "generation"])["m"]
        .sum()
        .values
    )
    by_gen["share"] = (by_gen["with_mention"] / by_gen["versions"]).round(4)
    codes = []
    for kind in ("what", "why"):
        for name, c in ctx.code_info[kind].items():
            cands = c.get("theory_candidates") or (
                [c["theory_link"]] if c.get("theory_link") else []
            )
            pubs = (ctx.literature or {}).get(name, [])
            codes.append(
                dict(
                    kind=kind,
                    code=name,
                    family=ctx.families[kind].get(name),
                    theory_candidates=";".join(map(str, cands)),
                    n_publications=len(pubs),
                    publications=" | ".join(
                        str(p.get("reference", p)) if isinstance(p, dict) else str(p)
                        for p in pubs
                    ),
                    verified=sum(
                        bool(p.get("verified")) for p in pubs if isinstance(p, dict)
                    ),
                )
            )
    cd = pd.DataFrame(codes)
    linked = {
        r["code"]
        for r in codes
        if r["kind"] == "what" and (r["theory_candidates"] or r["n_publications"])
    }
    mech = []
    for stem, rec in ctx.records.items():
        explicit = any(_named(t) for t in rec.get("theory_mentions", []))
        for m in rec["mechanisms"]:
            if m["status"] != "adopted":
                continue
            implicit = m.get("what_code") in linked
            mech.append(
                dict(
                    version=stem,
                    what_code=m.get("what_code"),
                    idea=idea_of(m),
                    explicit=explicit,
                    implicit=implicit,
                    link=(
                        "explicit" if explicit else ("implicit" if implicit else "none")
                    ),
                )
            )
    md = pd.DataFrame(mech)
    names = Counter(n for s in pv["mentions"] if s for n in s.split(";"))
    return (
        pv,
        by_gen,
        cd,
        md,
        {
            "versions_with_mention": int((pv["n_mentions"] > 0).sum()),
            "versions_with_record": int(pv["has_record"].sum()),
            "top_mentions": names.most_common(8),
            "adopted_mechanisms": int(len(md)),
            "adopted_explicit": int((md["link"] == "explicit").sum()) if len(md) else 0,
            "adopted_implicit_only": (
                int((md["link"] == "implicit").sum()) if len(md) else 0
            ),
            "literature_present": ctx.literature is not None,
            "codes_with_theory_link": len(linked),
        },
    )


# --- 8. quality --------------------------------------------------------------


def quality_table(ctx):
    import pandas as pd

    q = ctx.quality
    rows = []

    def add(group, metric, value):
        rows.append(dict(group=group, metric=metric, value=value))

    overall = next((r for r in ctx.reliability if r["code"] == "__overall__"), None)
    if overall:
        add("judge", "agreement_overall", _r(overall["agreement"]))
        add("judge", "kappa_overall", _r(overall["kappa"]))
    for r in ctx.reliability:
        if r["code"] != "__overall__":
            add("judge_agreement", r["code"], _r(r["agreement"]))
    add("judge", "unreliable_codes", ";".join(sorted(ctx.unreliable)))
    for code, share in (q.get("code_prevalence") or {}).items():
        add("prevalence", code, share)
    cov = (q.get("coverage_what_on_notes_mechanisms") or {}).get("all") or {}
    add("coverage", "what_on_notes_mechanisms", cov.get("rate"))
    unassigned = q.get("unassigned_by_codebook") or _unassigned_from_members(ctx)
    for kind, revs in unassigned.items():
        for rev, d in revs.items():
            add("unassigned", f"{kind}:{rev}", d.get("share"))
    for k, d in (q.get("quote_check") or {}).items():
        add("quote_check", k, d.get("rate"))
    for k, d in (q.get("say_do_gap") or {}).items():
        add("say_do_gap", k, d.get("rate"))
    for d in q.get("agreement_overall") or []:
        add("expert_check", f"{d['rater']} vs {d['judge']}", _r(d.get("agreement")))
    tot = ((q.get("calls") or {}).get("total")) or {}
    add("failures", "failed_items", tot.get("failed_items", 0))
    add("failures", "length_finishes", tot.get("length_finishes", 0))
    df = pd.DataFrame(rows)
    return df, {
        "judge_agreement": overall and _r(overall["agreement"]),
        "judge_kappa": overall and _r(overall["kappa"]),
        "unreliable_codes": sorted(ctx.unreliable),
        "coverage_what": cov.get("rate"),
        "unassigned": {
            k: {r: d.get("share") for r, d in v.items()} for k, v in unassigned.items()
        },
        "say_do_gap": {
            k: d.get("rate") for k, d in (q.get("say_do_gap") or {}).items()
        },
        "failed_items": tot.get("failed_items", 0),
        "expert_check": [
            {"rater": d["rater"], "agreement": _r(d.get("agreement"))}
            for d in q.get("agreement_overall") or []
        ],
    }


def _unassigned_from_members(ctx):
    out = {}
    for kind in ("what", "why", "basis"):
        p = ctx.run.dir / "codebook" / f"{kind}.v1.members.yml"
        if p.exists():
            mem = read_yaml(p) or {}
            total = sum(len(v) for v in mem.values())
            n = len(mem.get("none", []))
            out[kind] = {"v1": {"share": round(n / total, 4) if total else None}}
    return out


# --- figures -----------------------------------------------------------------


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "savefig.dpi": 200, "figure.dpi": 100})
    return plt


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def _code_order(ev, codes=None):
    """What codes grouped by family, within family by first appearance in a rule."""
    first = ev[ev["in_rule"].fillna(0) == 1].groupby("what_code")["generation"].min()
    fam = ev.drop_duplicates("what_code").set_index("what_code")["family"]
    codes = list(codes if codes is not None else fam.index)
    return sorted(codes, key=lambda c: (str(fam.get(c, "")), first.get(c, 99), str(c)))


def fig_prevalence(ev, flow, figdir):
    plt = _plt()
    import numpy as np

    out = []
    fl = flow[flow["level"] == "code"]
    for camp, g in fl.groupby("campaign"):
        gens = sorted(g["generation"].unique())
        codes = _code_order(ev[ev["campaign"] == camp])
        mat = np.full((len(codes), len(gens)), np.nan)
        for _, r in g.iterrows():
            mat[codes.index(r["name"]), gens.index(r["generation"])] = r[
                "share_in_rule"
            ]
        fam = ev.drop_duplicates("what_code").set_index("what_code")["family"]
        fig, ax = plt.subplots(
            figsize=(1.2 + 0.45 * len(gens), 0.6 + 0.22 * len(codes))
        )
        im = ax.imshow(mat, aspect="auto", cmap="cividis", vmin=0, vmax=1)
        ax.set_xticks(range(len(gens)))
        n_per_gen = g.groupby("generation")["n_versions"].max()
        ax.set_xticklabels([f"v{x}\nn={int(n_per_gen.get(x, 0))}" for x in gens])
        ax.set_yticks(range(len(codes)))
        ax.set_yticklabels([f"{c}  [{fam.get(c, '')}]" for c in codes], fontsize=7)
        prev = None
        for i, c in enumerate(codes):
            if prev is not None and fam.get(c) != prev:
                ax.axhline(i - 0.5, color="white", lw=1.5)
            prev = fam.get(c)
        ax.set_xlabel("version index")
        fig.colorbar(im, ax=ax, fraction=0.03, label="share of rules with the code")
        path = figdir / f"prevalence_{camp}.png"
        _save(fig, path)
        out.append(
            dict(
                file=f"figures/{path.name}",
                caption=f"Share of rules in campaign {camp} that implement each what "
                "code, by version index; codes grouped by family.",
            )
        )
    return out


def fig_lineages(ev, figdir, min_versions=4):
    plt = _plt()
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    out = []
    cmap = ListedColormap([STATE_COLOURS[s] for s in STATES])
    for lin, g in ev.groupby("lineage"):
        stems = (
            g.drop_duplicates("version").sort_values("generation")["version"].tolist()
        )
        if len(stems) < min_versions:
            continue
        active = g[g["idea_state"] != "absent"]["what_code"].unique().tolist()
        codes = _code_order(ev, active)
        if not codes:
            continue
        idx = {s: i for i, s in enumerate(STATES)}
        piv = g.pivot_table(
            index="what_code", columns="version", values="idea_state", aggfunc="first"
        )
        mat = [[idx[piv.loc[c, s]] for s in stems] for c in codes]
        unrel = set(g[g["code_unreliable"]]["what_code"])
        z = g.drop_duplicates("version").set_index("version")["score_z"]
        fig, (ax0, ax1) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(1.5 + 0.55 * len(stems), 1.6 + 0.22 * len(codes)),
            gridspec_kw={"height_ratios": [1, max(2, 0.22 * len(codes))]},
        )
        ys = [_num(z.get(s)) for s in stems]
        ax0.plot(range(len(stems)), ys, marker="o", color="#000000", lw=1)
        ax0.set_ylabel("score (z)")
        ax0.grid(axis="y", color="#DDDDDD", lw=0.5)
        ax1.imshow(mat, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(STATES) - 0.5)
        ax1.set_yticks(range(len(codes)))
        ax1.set_yticklabels(
            [c + (" *" if c in unrel else "") for c in codes], fontsize=7
        )
        ax1.set_xticks(range(len(stems)))
        ax1.set_xticklabels(
            [s.replace(lin, "").lstrip("-") or "v1" for s in stems], fontsize=8
        )
        ax1.set_xticks([x - 0.5 for x in range(1, len(stems))], minor=True)
        ax1.set_yticks([y - 0.5 for y in range(1, len(codes))], minor=True)
        ax1.grid(which="minor", color="white", lw=0.8)
        ax1.tick_params(which="minor", length=0)
        ax1.set_xlabel("version")
        # families: a separator between groups and the family name on the right
        fam = g.drop_duplicates("what_code").set_index("what_code")["family"]
        start = 0
        for i in range(1, len(codes) + 1):
            if i == len(codes) or fam.get(codes[i]) != fam.get(codes[start]):
                if i < len(codes):
                    ax1.axhline(i - 0.5, color="#000000", lw=0.8)
                ax1.text(
                    len(stems) - 0.4,
                    (start + i - 1) / 2,
                    str(fam.get(codes[start], "")),
                    va="center",
                    fontsize=6,
                    color="#444444",
                )
                start = i
        handles = [Patch(color=STATE_COLOURS[s], label=s) for s in STATES]
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=4,
            fontsize=7,
            frameon=False,
        )
        path = figdir / f"lineage_{lin}.png"
        _save(fig, path)
        out.append(
            dict(
                file=f"figures/{path.name}",
                caption=f"Idea flow along lineage {lin}: each what code that appears "
                "in "
                "a rule or the notes, coloured by its state per version; the line "
                "above is the within-campaign standardised score. * marks codes "
                "the judge codes unreliably.",
            )
        )
    return out


def fig_funnel(ev, figdir):
    plt = _plt()
    import numpy as np

    stages = [
        ("considered", ev["notes_status"] == "considered"),
        ("tested", ev["tested"].fillna(0).astype(float) == 1),
        ("refused", ev["notes_status"] == "rejected"),
        ("adopted in notes", ev["notes_status"] == "adopted"),
        ("present in rule", ev["in_rule"].fillna(0) == 1),
    ]
    fams = sorted(ev["family"].unique())
    fig, ax = plt.subplots(figsize=(7, 0.5 + 0.35 * len(fams)))
    h = 0.8 / len(stages)
    rows = {}
    for k, (label, mask) in enumerate(stages):
        counts = [int((mask & (ev["family"] == f)).sum()) for f in fams]
        rows[label] = dict(zip(fams, counts))
        ax.barh(
            np.arange(len(fams)) + k * h,
            counts,
            height=h,
            color=LINE_COLOURS[k],
            label=label,
        )
    ax.set_yticks(np.arange(len(fams)) + 0.4 - h / 2)
    ax.set_yticklabels(fams)
    ax.invert_yaxis()
    ax.set_xlabel("version x code cells")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    path = figdir / "funnel.png"
    _save(fig, path)
    return [
        dict(
            file="figures/funnel.png",
            caption="Per what family: how often its codes are considered, tested, "
            "refused and adopted in the notes, and present in a rule (version x "
            "code cells).",
        )
    ], rows


def fig_value(val, figdir):
    plt = _plt()

    d = val[val["introduced_n_unfiltered"] > 0].copy()
    if d.empty:
        return []
    d = d.sort_values("introduced_mean_unfiltered")
    fig, ax = plt.subplots(figsize=(6, 0.5 + 0.25 * len(d)))
    for i, (_, r) in enumerate(d.iterrows()):
        grey = r["code_unreliable"] or r["introduced_n_unfiltered"] < 5
        m = r["introduced_mean_unfiltered"]
        se = r["introduced_se_unfiltered"]
        colour = "#999999" if grey else "#0072B2"
        ax.errorbar(
            m,
            i,
            xerr=None if se is None or math.isnan(se) else 1.96 * se,
            fmt="o",
            color=colour,
            ms=4,
            capsize=2,
        )
        ax.annotate(
            f"n={int(r['introduced_n_unfiltered'])}",
            (m, i),
            xytext=(6, -3),
            textcoords="offset points",
            fontsize=6,
            color=colour,
        )
    ax.axvline(0, color="#000000", lw=0.6)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["what_code"], fontsize=7)
    ax.set_xlabel("change in standardised score when the code is introduced (95% CI)")
    path = figdir / "idea_value.png"
    _save(fig, path)
    return [
        dict(
            file="figures/idea_value.png",
            caption="Mean change in within-campaign standardised score from parent to "
            "child when a code is introduced; grey: fewer than 5 events or a code "
            "the judge codes unreliably.",
        )
    ]


def fig_why_by_what(raw, figdir):
    plt = _plt()

    if raw.empty:
        return []
    piv = raw.groupby(["what_family", "why_family"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(
        figsize=(1.5 + 0.6 * piv.shape[1], 0.8 + 0.35 * piv.shape[0])
    )
    im = ax.imshow(piv.values, aspect="auto", cmap="cividis")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=7)
    vmax = piv.values.max() or 1
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if v:
                ax.text(
                    j,
                    i,
                    str(v),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black" if v > vmax * 0.6 else "white",
                )
    ax.set_xlabel("why family")
    ax.set_ylabel("what family")
    fig.colorbar(im, ax=ax, fraction=0.03, label="reasons")
    _save(fig, figdir / "why_by_what.png")
    return [
        dict(
            file="figures/why_by_what.png",
            caption="Reasons given for or against mechanisms, by what family of the "
            "mechanism and why family of the reason.",
        )
    ]


def _lines(df, x, y, group, ylabel, path, caption):
    plt = _plt()

    if df.empty:
        return []
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for k, (name, g) in enumerate(df.groupby(group)):
        g = g.sort_values(x)
        ax.plot(
            g[x],
            g[y],
            marker="o",
            ms=3,
            lw=1.2,
            color=LINE_COLOURS[k % len(LINE_COLOURS)],
            label=str(name),
        )
    ax.set_xlabel("version index")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#DDDDDD", lw=0.5)
    ax.legend(fontsize=7, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    _save(fig, path)
    return [dict(file=f"figures/{path.name}", caption=caption)]


def fig_basis(raw, figdir):
    if raw.empty:
        return []
    g = raw.groupby(["generation", "basis_family"]).size().reset_index(name="n")
    g["share"] = g["n"] / g.groupby("generation")["n"].transform("sum")
    return _lines(
        g,
        "generation",
        "share",
        "basis_family",
        "share of reasons",
        figdir / "basis_by_generation.png",
        "Share of reasons resting on each kind of support (basis family), by version "
        "index, all campaigns.",
    )


def fig_theory(by_gen, figdir):
    return _lines(
        by_gen,
        "generation",
        "share",
        "campaign",
        "share of versions naming a concept",
        figdir / "theory_by_generation.png",
        "Share of versions whose notes name a theory or concept explicitly, by "
        "version index and campaign.",
    )


def fig_reliability(ctx, figdir):
    plt = _plt()

    rows = [
        r
        for r in ctx.reliability
        if r["code"] != "__overall__" and _num(r.get("agreement")) is not None
    ]
    if not rows:
        return []
    rows.sort(key=lambda r: r["agreement"])
    fig, ax = plt.subplots(figsize=(6, 0.5 + 0.22 * len(rows)))
    colours = [
        "#D55E00" if r["agreement"] < ctx.min_agreement else "#0072B2" for r in rows
    ]
    ax.barh(range(len(rows)), [r["agreement"] for r in rows], color=colours)
    ax.axvline(ctx.min_agreement, color="#000000", lw=0.8, ls="--")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["code"] for r in rows], fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("test-retest agreement of the rule judge (repeat 1 vs 2)")
    _save(fig, figdir / "reliability.png")
    return [
        dict(
            file="figures/reliability.png",
            caption="Per-code test-retest agreement of the rule judge, sorted; codes "
            f"below {ctx.min_agreement} (dashed) are treated as unreliable.",
        )
    ]


# --- report ------------------------------------------------------------------


TABLE_TEXT = {
    "idea_events": "One row per version and what code (never-implemented codes "
    "included): rule event, notes status, idea_state and say-do flags.",
    "idea_flow_by_generation": "Per campaign, version index and code (and family): "
    "share of versions with the code in the rule and counts of each idea_state.",
    "idea_first_seen": "Per code and campaign: first version index in notes, tested "
    "and in a rule; in how many lineages it appears independently; re-refusals.",
    "idea_value": "Per code: mean change in standardised score on introduction and "
    "on dropping, with n, SE and share positive; unreliable codes excluded (raw "
    "numbers in the *_unfiltered columns).",
    "why_by_what": "Reasons by what and why (family and code level), split by stance.",
    "basis_by_generation": "Share of reasons per basis family and per kind of "
    "evidence pointer, by campaign and version index.",
    "theory": "Explicit theory mentions per version; theory_by_generation.csv, "
    "theory_codes.csv (codebook links and literature) and theory_mechanisms.csv "
    "(adopted mechanisms with an explicit or only an implicit link).",
    "quality": "Judge reliability, unreliable codes, prevalence, coverage and "
    "unassigned shares, quote checks, say-do gap, expert agreement, failures.",
}


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3g}"
    if isinstance(v, dict):
        return ", ".join(f"{k}: {_fmt(x)}" for k, x in v.items())
    if isinstance(v, list):
        if v and isinstance(v[0], dict):
            return "; ".join(_fmt(x) for x in v)
        return ", ".join(map(_fmt, v))
    return str(v)


def render_report(summary):
    lines = [
        f"# Analysis of run {summary['run']}",
        "",
        "Generated by `python -m aimanager.annotation analyse` from "
        "`analysis/summary.json`; do not edit by hand.",
        "",
        "## Tables",
        "",
    ]
    for name, t in summary["tables"].items():
        lines.append(f"- **`{t['file']}`**: {TABLE_TEXT.get(name, '')}")
        for k, v in t.get("headline", {}).items():
            lines.append(f"  - {k}: {_fmt(v)}")
    lines += ["", "## Figures", ""]
    for f in summary["figures"]:
        lines.append(f"- **`{f['file']}`**: {f['caption']}")
    if summary.get("not_computed"):
        lines += ["", "## Not computed", ""]
        lines += [f"- {x}" for x in summary["not_computed"]]
    return "\n".join(lines) + "\n"


# --- entry point -------------------------------------------------------------


def analyse(run, cfg, limit=None):
    require(run, "apply-notes")
    ctx = Context(run, cfg)
    adir = run.dir / "analysis"
    figdir = adir / "figures"
    tables, figures, not_computed = {}, [], []

    def table(name, df, headline, file=None):
        file = file or f"{name}.csv"
        _write_csv(df, adir / file)
        tables[name] = {"file": file, "rows": int(len(df)), "headline": headline}

    ev, h = idea_events(ctx)
    table("idea_events", ev, h)
    flow, h = idea_flow(ev)
    table("idea_flow_by_generation", flow, h)
    fs, h = first_seen(ev)
    table("idea_first_seen", fs, h)
    val, h = idea_value(ev)
    table("idea_value", val, h)
    wbw, wraw, h = why_by_what(ctx)
    table("why_by_what", wbw, h)
    bbg, braw, h = basis_by_generation(ctx)
    table("basis_by_generation", bbg, h)
    pv, tgen, tcodes, tmech, h = theory(ctx)
    table("theory", pv, h)
    _write_csv(tgen, adir / "theory_by_generation.csv")
    _write_csv(tcodes, adir / "theory_codes.csv")
    _write_csv(tmech, adir / "theory_mechanisms.csv")
    if ctx.literature is None:
        not_computed.append(
            "publications per code: codebook/literature.yml is not present yet"
        )
    qt, h = quality_table(ctx)
    table("quality", qt, h)
    if not ctx.reliability:
        not_computed.append("judge reliability: analysis/judge_reliability.csv missing")
    if not (ctx.quality.get("agreement_overall") or []):
        not_computed.append("expert-check agreement: no rater rows")

    figures += fig_prevalence(ev, flow, figdir)
    figures += fig_lineages(ev, figdir)
    f, funnel = fig_funnel(ev, figdir)
    figures += f
    tables["idea_events"]["headline"]["funnel_by_family"] = funnel
    figures += fig_value(val, figdir)
    figures += fig_why_by_what(wraw, figdir)
    figures += fig_basis(braw, figdir)
    figures += fig_theory(tgen, figdir)
    figures += fig_reliability(ctx, figdir)

    summary = {
        "run": run.run_id,
        "codebooks": {
            k: latest_codebook(run, k)[1].name for k in ("what", "why", "basis")
        },
        "families": {
            k: dict(Counter(ctx.families[k].values())) for k in ("what", "why", "basis")
        },
        "tables": tables,
        "figures": figures,
        "not_computed": not_computed,
    }
    write_json(adir / "summary.json", json.loads(json.dumps(summary, default=_js)))
    atomic_write_text(adir / "REPORT.md", render_report(summary))
    log(
        f"analyse: {len(tables)} tables, {len(figures)} figures, "
        f"{len(not_computed)} items not computed"
    )
    return summary


def _js(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


__all__ = ["analyse", "idea_state", "ref_kind"]
