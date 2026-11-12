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
import re
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
    "initial",
    "introduced",
    "kept",
    "dropped",
    "done_but_refused",
    "said_not_done",
    "tested_and_refused",
    "refused_untested",
    "considered",
    "not_judged",
    "absent",
]
# Okabe-Ito plus two of Paul Tol's colours, colour-blind safe; absent is a light
# grey and not_judged (no rule, or the judge failed) a mid grey
STATE_COLOURS = {
    "initial": "#117733",
    "introduced": "#009E73",
    "kept": "#0072B2",
    "dropped": "#D55E00",
    "done_but_refused": "#882255",
    "said_not_done": "#F0E442",
    "tested_and_refused": "#CC79A7",
    "refused_untested": "#E69F00",
    "considered": "#56B4E9",
    "not_judged": "#AAAAAA",
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
LINK_STRENGTHS = ("direct", "related")


# --- loading -----------------------------------------------------------------


class Context:
    """Everything the tables need, loaded once."""

    def __init__(self, run, cfg):
        import pandas as pd

        self.run, self.cfg = run, cfg
        adir = run.dir / "analysis"
        self.join_manifest = require_join(run)
        self.vc = pd.read_csv(
            adir / "version_codes.csv", keep_default_na=False, na_values=[""]
        )
        self.quality = read_json(adir / "quality.json")
        self.versions = {v["stem"]: v for v in load_versions(run)}
        self.records = load_records(run, coded=True)
        self.families = {}
        self.family_defs = {}
        self.code_info = {}
        used = self.join_manifest["codebooks"]
        for kind in ("what", "why", "basis"):
            path = run.dir / "codebook" / used[kind]["file"]
            codes = read_yaml(path)["codes"]
            defs = codebook_families(path) or codebook_families(
                run.dir / "codebook" / f"{kind}.v1.yml"
            )
            fams = {c["name"]: c.get("family") or "none" for c in codes}
            if kind == "what" and "never_implemented" in used:
                npath = run.dir / "codebook" / used["never_implemented"]["file"]
                never = read_yaml(npath).get("codes") or []
                fams.update({c["name"]: c.get("family") or "none" for c in never})
                defs = {**codebook_families(npath), **defs}
                codes = codes + never
            self.families[kind] = fams
            self.family_defs[kind] = defs
            self.code_info[kind] = {c["name"]: c for c in codes}
        self.never = set()
        if "never_implemented" in used:
            npath = run.dir / "codebook" / used["never_implemented"]["file"]
            self.never = {c["name"] for c in read_yaml(npath).get("codes") or []}
        lit = run.dir / "codebook" / "literature.yml"
        self.literature = _literature(read_yaml(lit)) if lit.exists() else None
        self.reliability = []
        rel = adir / "judge_reliability.csv"
        if rel.exists():
            self.reliability = pd.read_csv(rel).to_dict("records")
        from .join import reliability_class

        self.rel_class = {
            r["code"]: reliability_class(
                {k: ("" if v != v else v) for k, v in r.items()}, cfg
            )
            for r in self.reliability
            if r["code"] != "__overall__"
        }
        self.unreliable = {c for c, k in self.rel_class.items() if k == "unreliable"}
        self.rare = {c for c, k in self.rel_class.items() if k == "rare"}


def require_join(run):
    """The join manifest, after checking that join is current: nothing it read
    (rule codes incl. expert rows, coded records, codebook revisions) or wrote
    changed since, and no newer codebook revision exists."""
    path = run.dir / "manifests" / "join.json"
    if not path.exists():
        raise SystemExit(f"no {path}: run join (again) before analyse")
    try:
        doc = require(run, "join")
    except SystemExit as e:
        raise SystemExit(f"join is stale, rerun join before analyse: {e}")
    for kind, used in (doc.get("codebooks") or {}).items():
        _, latest, _ = latest_codebook(run, kind)
        if latest.name != used["file"]:
            raise SystemExit(
                f"join used {used['file']} but the latest {kind} codebook is "
                f"{latest.name}: rerun stage B and join before analyse"
            )
    return doc


def _reference(p):
    authors = p.get("authors") or []
    who = authors[0].split()[-1] + (" et al." if len(authors) > 2 else "")
    if len(authors) == 2:
        who = " and ".join(a.split()[-1] for a in authors)
    return (
        f"{who} ({p.get('year', 'n.d.')}). {p.get('title', '')}. {p.get('venue', '')}"
    )


def _literature(doc):
    """{code: [publication dict]} from literature.yml.

    Accepted shapes: the literature step's file (`publications: {id: {authors,
    year, title, venue, doi, url, verified}}` and `codes: {kind: {code: {links:
    [{publication: id, relation, connection, strength}]}}}`); a `codes:` list of
    {code, publications}; or a plain mapping code -> [publication].
    """
    if not doc or not isinstance(doc, dict):
        return {}
    codes = doc.get("codes")
    if isinstance(codes, dict) and isinstance(doc.get("publications"), dict):
        pubs, out = doc["publications"], {}
        for by_code in codes.values():
            for code, entry in (by_code or {}).items():
                links = []
                for link in (entry or {}).get("links") or []:
                    p = pubs.get(link.get("publication"), {})
                    links.append(
                        {
                            "id": link.get("publication"),
                            "reference": (
                                _reference(p) if p else link.get("publication")
                            ),
                            "doi": p.get("doi"),
                            "url": p.get("url"),
                            "verified": bool(p.get("verified")),
                            "relation": link.get("relation"),
                            "strength": link.get("strength"),
                            "connection": link.get("connection"),
                        }
                    )
                out[code] = links
        return out
    if isinstance(codes, list):
        return {c["code"]: c.get("publications") or [] for c in codes}
    return {k: (v or []) for k, v in doc.items() if isinstance(v, list)}


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


def idea_state(in_rule, event, notes_status, tested, has_parent=True):
    """One label per version and code from the rule codes and the notes.

    in_rule None means the rule was not judged (no rule, or a failed judge
    call): `not_judged`, never `absent`. A rule that has the code while the notes
    reject it is `done_but_refused`; notes that adopt a code the rule lacks are
    `said_not_done`. A first version's codes are `initial`, not `introduced`.
    """
    if in_rule is None:
        return "not_judged"
    if in_rule:
        if notes_status == "rejected":
            return "done_but_refused"
        if not has_parent:
            return "initial"
        return "introduced" if event == "introduced" else "kept"
    if event == "dropped":
        return "dropped"
    if notes_status == "adopted":
        return "said_not_done"
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
    judged = vc["in_rule"].notna()
    in_rule = vc["in_rule"].fillna(0).astype(float) == 1
    tested = vc["notes_tested"].fillna(0).astype(float) == 1
    has_parent = vc["parent"].notna()
    vc["family"] = vc["what_code"].map(ctx.families["what"]).fillna("none")
    vc["score_z_parent"] = vc["parent"].map(z)
    vc["score_z_change"] = vc["score_z"] - vc["score_z_parent"]
    vc["code_unreliable"] = vc["what_code"].isin(ctx.unreliable)
    vc["idea_state"] = [
        idea_state(r if j else None, e, n, t, hp)
        for r, j, e, n, t, hp in zip(
            in_rule, judged, vc["event"], vc["notes_status_label"], tested, has_parent
        )
    ]
    has_record = vc["notes_status"] != "no_record"
    adopted = vc["notes_status_label"] == "adopted"
    vc["said_not_done"] = adopted & ~in_rule & judged
    vc["done_not_said"] = in_rule & ~adopted & has_record
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
        "say_do": say_do(vc, judged, in_rule, adopted, has_parent, has_record),
    }


def _share(k, n):
    return {"n": int(n), "yes": int(k), "share": round(k / n, 4) if n else None}


def say_do(vc, judged, in_rule, adopted, has_parent, has_record):
    """The one say-do definition (cells of version x code with a notes record):

    - introductions_without_adopted_mention: of the rule introductions that have
      a parent, the share whose notes do not adopt the code;
    - kept_without_adopted_mention: the same for kept codes (notes rarely
      re-argue what they keep, so it is reported apart);
    - said_not_done: of the cells whose notes adopt the code, the share whose
      judged rule lacks it.
    """
    intro = (vc["event"] == "introduced") & has_parent & judged & has_record
    kept = (vc["event"] == "kept") & judged & has_record
    said = adopted & judged
    return {
        "introductions_without_adopted_mention": _share(
            (intro & ~adopted).sum(), intro.sum()
        ),
        "kept_without_adopted_mention": _share((kept & ~adopted).sum(), kept.sum()),
        "said_not_done": _share((said & ~in_rule).sum(), said.sum()),
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


IDEA_VALUE_CAVEAT = (
    "Descriptive and confounded: several codes often change in one step, the "
    "agent chose what to change, and scores differ in noise. n_transitions "
    "counts the other codes changing in the same step; the solo_* columns use "
    "only steps where exactly one code changed."
)


def idea_value(ev):
    import pandas as pd

    changes = ev[ev["event"].isin(["introduced", "dropped"]) & ev["parent"].notna()]
    per_step = changes.groupby("version")["what_code"].count()
    ev = ev.assign(n_transitions=ev["version"].map(per_step).fillna(0).astype(int) - 1)
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
            ev_rows = u[u["event"] == event]
            row[f"{event}_mean_n_transitions"] = (
                round(float(ev_rows["n_transitions"].mean()), 2)
                if len(ev_rows)
                else None
            )
            solo = ev_rows[ev_rows["n_transitions"] == 0]["score_z_change"].tolist()
            for k, v in _stats([] if row["code_unreliable"] else solo).items():
                row[f"{event}_solo_{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    ok = df[(df["introduced_n"] >= 5)].sort_values("introduced_mean", ascending=False)
    return df, {
        "caveat": IDEA_VALUE_CAVEAT,
        "events_used": int(len(usable)),
        "solo_steps": int((usable["n_transitions"] == 0).sum()),
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
    omitted = Counter()
    for rec in ctx.records.values():
        for m in rec["mechanisms"]:
            codes = m.get("what_codes")
            if codes is None:
                codes = [m["what_code"]] if m.get("what_code") else []
            for r in m["reasons"]:
                if not codes:
                    omitted["mechanism_without_what_code"] += 1
                elif not r.get("why_code"):
                    omitted["reason_without_why_code"] += 1
            for code in codes:
                for r in m["reasons"]:
                    if not r.get("why_code"):
                        continue
                    rows.append(
                        dict(
                            what_code=code,
                            what_family=ctx.families["what"].get(code, "none"),
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
            "reasons_omitted": dict(omitted),
            "note": "a reason counts once per what code of its mechanism",
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
    import re

    if "pf-" in r or re.search(r"\bv\d+\b", r):
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
                if bc:
                    fam = ctx.families["basis"].get(bc, "none")
                elif not (r.get("evidence") or "").strip():
                    fam = "no_evidence_phrase"
                else:
                    fam = "no_fitting_code"
                rows.append(
                    dict(
                        campaign=v["campaign"],
                        generation=v["version"],
                        basis_family=fam,
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
                    direct_links=sum(
                        p.get("strength") == "direct"
                        for p in pubs
                        if isinstance(p, dict)
                    ),
                )
            )
    cd = pd.DataFrame(codes)
    # linked: the code has at least one literature link of strength direct or
    # related (theory_candidates do not count); not computed without literature
    lit = ctx.literature
    linked = (
        None
        if lit is None
        else {
            code
            for code, pubs in lit.items()
            if any(
                isinstance(p, dict) and p.get("strength", "related") in LINK_STRENGTHS
                for p in pubs
            )
        }
    )
    mech = []
    for stem, rec in ctx.records.items():
        v = ctx.versions[stem]
        names = [t["name"] for t in rec.get("theory_mentions", []) if _named(t)]
        pats = [re.compile(r"\b" + re.escape(n) + r"\b", re.I) for n in names]
        for m in rec["mechanisms"]:
            if m["status"] not in ("adopted", "rejected"):
                continue
            codes = m.get("what_codes")
            if codes is None:
                codes = [m["what_code"]] if m.get("what_code") else []
            texts = [m.get("description", ""), m.get("quote", "")]
            for r in m["reasons"]:
                texts += [r.get("description", ""), r.get("quote", "")]
            hits = sorted(
                {
                    n
                    for n, pat in zip(names, pats)
                    if any(pat.search(t or "") for t in texts)
                }
            )
            explicit = bool(hits)
            is_linked = None if linked is None else any(c in linked for c in codes)
            if is_linked is None:
                combo = "explicit" if explicit else "not_explicit"
            else:
                combo = (
                    ("explicit" if explicit else "not_explicit")
                    + "_"
                    + ("linked" if is_linked else "not_linked")
                )
            mech.append(
                dict(
                    campaign=v["campaign"],
                    generation=v["version"],
                    version=stem,
                    status=m["status"],
                    what_codes=";".join(codes),
                    idea=idea_of(m),
                    theory_names=";".join(hits),
                    explicit=explicit,
                    linked=is_linked,
                    combination=combo,
                )
            )
    md = pd.DataFrame(mech)
    combos = (
        md.groupby(["status", "campaign", "generation", "combination"])
        .size()
        .reset_index(name="n")
        if len(md)
        else pd.DataFrame(
            columns=["status", "campaign", "generation", "combination", "n"]
        )
    )
    overall = {
        st: dict(Counter(md[md["status"] == st]["combination"])) if len(md) else {}
        for st in ("adopted", "rejected")
    }
    mention_names = Counter(n for s_ in pv["mentions"] if s_ for n in s_.split(";"))
    explicit_names = Counter(
        n for s_ in md.get("theory_names", []) if s_ for n in s_.split(";")
    )
    return (
        pv,
        by_gen,
        cd,
        md,
        combos,
        {
            "versions_with_mention": int((pv["n_mentions"] > 0).sum()),
            "versions_with_record": int(pv["has_record"].sum()),
            "theory_names": mention_names.most_common(),
            "names_tied_to_mechanisms": explicit_names.most_common(),
            "combinations_adopted": overall["adopted"],
            "combinations_rejected": overall["rejected"],
            "definition": (
                "explicit: a theory name of the record occurs (case-insensitive, "
                "word boundary) in the description or quote of the mechanism or "
                "of one of its reasons; linked: a what code of the mechanism has "
                "a literature link of strength direct or related"
            ),
            "literature_present": lit is not None,
            "codes_linked": None if linked is None else len(linked),
        },
    )


# --- 8. quality --------------------------------------------------------------


def quality_table(ctx, say_do_headline):
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
            add("judge_kappa", r["code"], _r(r["kappa"]))
            add("judge_class", r["code"], ctx.rel_class.get(r["code"]))
    add("judge", "unreliable_codes", ";".join(sorted(ctx.unreliable)))
    add("judge", "rare_codes", ";".join(sorted(ctx.rare)))
    for code, hits in (q.get("never_implemented_found") or {}).items():
        add("never_implemented_found", code, len(hits))
    for d in q.get("draft_codes_above_max_prevalence") or []:
        add("draft_above_max_prevalence", d["code"], d["prevalence"])
    for code, share in (q.get("code_prevalence") or {}).items():
        add("prevalence", code, share)
    cov = (q.get("coverage_what_on_notes_mechanisms") or {}).get("all") or {}
    add("coverage", "what_on_notes_mechanisms", cov.get("rate"))
    unassigned = q.get("unassigned_by_codebook") or _unassigned_from_members(ctx)
    for kind, revs in unassigned.items():
        for rev, d in revs.items():
            if isinstance(d, dict):
                add("unassigned", f"{kind}:{rev}", d.get("share"))
    for k, d in (q.get("quote_check") or {}).items():
        add("quote_check", k, d.get("rate"))
    for k, d in say_do_headline.items():
        add("say_do", k, d.get("share"))
    for d in q.get("agreement_overall") or []:
        pair = f"{d['rater']} vs {d['judge']}"
        kind = d.get("pair_type", "rater-judge")
        add(f"agreement_{kind}", pair, _r(d.get("agreement")))
        add(f"kappa_{kind}", pair, _r(d.get("kappa")))
    tot = ((q.get("calls") or {}).get("total")) or {}
    add("failures", "failed_items", tot.get("failed_items", 0))
    add("failures", "length_finishes", tot.get("length_finishes", 0))
    df = pd.DataFrame(rows)
    return df, {
        "judge_agreement": overall and _r(overall["agreement"]),
        "judge_kappa": overall and _r(overall["kappa"]),
        "unreliable_codes": sorted(ctx.unreliable),
        "rare_codes": sorted(ctx.rare),
        "never_implemented_found": {
            k: len(v) for k, v in (q.get("never_implemented_found") or {}).items()
        },
        "coverage_what": cov.get("rate"),
        "unassigned": {
            k: {r: (d.get("share") if isinstance(d, dict) else d) for r, d in v.items()}
            for k, v in unassigned.items()
        },
        "say_do": {k: d.get("share") for k, d in say_do_headline.items()},
        "failed_items": tot.get("failed_items", 0),
        "agreement": [
            {
                "pair": f"{d['rater']} vs {d['judge']}",
                "type": d.get("pair_type", "rater-judge"),
                "agreement": _r(d.get("agreement")),
                "kappa": _r(d.get("kappa")),
            }
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
        never = set(ev[ev["code_source"] == "never_implemented"]["what_code"])
        ax.set_yticklabels(
            [
                f"{c}  [{fam.get(c, '')}]" + ("  (never impl.)" if c in never else "")
                for c in codes
            ],
            fontsize=7,
        )
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
        never_codes = set(g[g["code_source"] == "never_implemented"]["what_code"])
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
            [
                c
                + (" *" if c in unrel else "")
                + (" (never impl.)" if c in never_codes else "")
                for c in codes
            ],
            fontsize=7,
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


RELIABILITY_COLOURS = {
    "reliable": "#0072B2",
    "unreliable": "#D55E00",
    "rare": "#AAAAAA",
}


def fig_reliability(ctx, figdir):
    plt = _plt()
    from matplotlib.patches import Patch

    rows = [
        r
        for r in ctx.reliability
        if r["code"] != "__overall__" and _num(r.get("kappa")) is not None
    ]
    rows += [
        r
        for r in ctx.reliability
        if r["code"] != "__overall__" and _num(r.get("kappa")) is None
    ]
    if not rows:
        return []
    rows.sort(
        key=lambda r: _num(r.get("kappa")) if _num(r.get("kappa")) is not None else 2.0
    )
    fig, ax = plt.subplots(figsize=(6, 0.6 + 0.22 * len(rows)))
    cls = [ctx.rel_class.get(r["code"], "reliable") for r in rows]
    vals = [
        _num(r.get("kappa")) if _num(r.get("kappa")) is not None else 1.0 for r in rows
    ]
    ax.barh(range(len(rows)), vals, color=[RELIABILITY_COLOURS[c] for c in cls])
    kappa_floor = ctx.cfg.get("judge", {}).get("min_kappa", 0.8)
    ax.axvline(kappa_floor, color="#000000", lw=0.8, ls="--")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [
            r["code"] + (" [never impl.]" if r["code"] in ctx.never else "")
            for r in rows
        ],
        fontsize=7,
    )
    ax.set_xlim(0, 1.02)
    ax.set_xlabel(
        "Cohen's kappa of the rule judge, repeat 0 vs repeat 1 (judge_repeats.jsonl)"
    )
    ax.legend(
        handles=[Patch(color=v, label=k) for k, v in RELIABILITY_COLOURS.items()],
        fontsize=7,
        frameon=False,
        loc="lower right",
    )
    _save(fig, figdir / "reliability.png")
    return [
        dict(
            file="figures/reliability.png",
            caption="Per-code test-retest kappa of the rule judge (repeats 0 and 1), "
            f"sorted; unreliable: kappa below {kappa_floor} (dashed) or positive "
            "agreement below the configured floor; rare: too few positives to rate. "
            "Codes without variation are drawn at 1.",
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
    pv, tgen, tcodes, tmech, tcombo, h = theory(ctx)
    table("theory", pv, h)
    _write_csv(tgen, adir / "theory_by_generation.csv")
    _write_csv(tcodes, adir / "theory_codes.csv")
    _write_csv(tmech, adir / "theory_mechanisms.csv")
    _write_csv(tcombo, adir / "theory_combinations.csv")
    if ctx.literature is None:
        not_computed.append(
            "publications per code and the `linked` side of the theory table: "
            "codebook/literature.yml is not present yet"
        )
    qt, h = quality_table(ctx, tables["idea_events"]["headline"]["say_do"])
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
        "codebooks": {k: d["file"] for k, d in ctx.join_manifest["codebooks"].items()},
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
