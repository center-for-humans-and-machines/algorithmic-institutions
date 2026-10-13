"""Live dashboard of policy-finder instances (#236, #227).

Reads the instance worktrees (`../policy-finder-worktrees`, or
$PF_WORKTREE_ROOT) and refreshes every few seconds: per instance its status,
rule, notes, scripts, final report and, once swept, its sweep result, which
run_sweep.sh (#241) commits in the worktree. Read-only: it never writes to a
worktree.

Run from the repo root (streamlit is not a project dependency):
    uv run --no-sync --with streamlit streamlit run scripts/policy_finder/observation_app/dashboard.py
"""

import base64
import hashlib
import html
import json
import math
import os
import re
import subprocess
from pathlib import Path

import altair as alt
from summarize import OBSERVATORY
import pandas as pd
import streamlit as st
import yaml

MAIN = Path(__file__).resolve().parents[3]
WT_ROOT = Path(
    os.environ.get("PF_WORKTREE_ROOT", MAIN.parent / "policy-finder-worktrees")
)
BASE = os.environ.get("PF_BASE", "policy-finder-base")
#: An instance's merged sweep, relative to its worktree (run_sweep.sh)
SWEEP_DIR = "plots/simulation/policy_finder/{name}_sweep"
#: The best point's matchup against `ah` (simulation/sweep.py's BEST_PLOTS)
BEST_PLOTS = {
    "comparison_pairing_side.jpg": "Best point vs ah, by side",
    "group_size_evolution_global.jpg": "Group sizes",
}
#: The reference sim with `zero` (never punish) against `ah`.
REFERENCE = MAIN / "plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched"
REFRESH_S = 5

# Palette of the project's report artifacts; the charts use the light values
CLAY, INK, MUTED, LINE = "#d97757", "#141413", "#87867f", "#e3dacc"

#: Fonts of the report artifacts; the stylesheet sits next to this script
FONTS = (
    "https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;"
    "6..72,500&family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500"
    "&display=swap"
)
#: Injected with st.html: st.markdown would parse the CSS as Markdown and print
#: everything after its first blank line as text
STYLE = (
    f'<style>@import url("{FONTS}");\n'
    f"{(Path(__file__).parent / 'dashboard.css').read_text()}\n"
    f"{(Path(__file__).parent / 'overview.css').read_text()}\n"
    f"{(Path(__file__).parent / 'summary.css').read_text()}</style>"
)
#: Under the page title: where the data outside the repo lives, and why
DATA_NOTE = (
    "Part of this page's data is kept outside the repo. The Summary's categories"
    " and the reference pools (summary.json, references.json) are kept on Raven in"
    f" {OBSERVATORY.split(':', 1)[1]}/, with working copies beside the instances in"
    " ../policy-finder-worktrees/; the overview text, the agent reports and their"
    " session logs stay on the machine that ran them. None of it is on"
    " policy-finder-base: every policy-finder agent starts from that branch and"
    " can read all of it, and this data describes earlier rules and their results."
)
#: The landing page's text, an HTML fragment in overview.css's classes. Kept
#: beside the instances, outside every worktree, as it describes earlier rules.
OVERVIEW = WT_ROOT / "overview.html"

STATUS = {
    "running": "running",
    "checked": "checked",
    "done": "done, not checked",
    "stopped": "stopped, no report",
}


def esc(text) -> str:
    return html.escape(str(text))


def read(path: Path):
    """A file's text, or None if it is missing or not text (an agent's cache)."""
    try:
        return path.read_text()
    except (OSError, UnicodeDecodeError):
        return None


def instances():
    if not WT_ROOT.is_dir():
        return []
    # instance worktrees only, not the <name>.sweep_parts beside them
    dirs = [d for d in WT_ROOT.iterdir() if (d / ".claude/policy_finder.json").exists()]
    return sorted(dirs, key=lambda d: d.stat().st_mtime, reverse=True)


def running_names():
    """Instances whose launcher is still running."""
    out = subprocess.run(["ps", "-eo", "command"], capture_output=True, text=True)
    names = set()
    for line in out.stdout.splitlines():
        if "new_instance.sh" in line:
            names.update(w for w in line.split() if w.startswith("pf-"))
    return names


def committed(name: str) -> int:
    """Commits on the instance's branch since the base (the check commits)."""
    out = subprocess.run(
        [
            "git",
            "-C",
            str(MAIN),
            "rev-list",
            "--count",
            f"{BASE}..policy-finder/{name}",
        ],
        capture_output=True,
        text=True,
    )
    return int(out.stdout.strip()) if out.returncode == 0 else 0


def status(name: str, running: set) -> str:
    if name in running:
        return "running"
    report = WT_ROOT / f"{name}.report.md"
    if report.exists() and report.stat().st_size > 0:  # the launcher makes it empty
        return "checked" if committed(name) else "done"
    return "stopped"


def pill(key: str) -> str:
    return f'<span class="pill {key}"><i></i>{STATUS[key]}</span>'


def empty(text: str):
    st.markdown(f'<div class="empty">{esc(text)}</div>', unsafe_allow_html=True)


def doc(markdown: str, key: str):
    with st.container(key=f"doc-{key}"):  # styled through its st-key-doc-* class
        st.markdown(markdown)


#: Claude Code keeps each project's session logs under this dir, named by the
#: project path with every non-alphanumeric character as "-"
SESSIONS = Path.home() / ".claude/projects"


def parse_notes(text: str) -> dict:
    """The notes' `## ` sections (`### Justification` folded in by its name)."""
    sections, current = {}, None
    for line in text.splitlines():
        heading = re.match(r"^#{2,3} (.+)$", line)
        if heading:
            current = heading.group(1).strip()
            sections[current] = []
        elif current:
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def items(section: str) -> list:
    """A numbered list's items, each on one line."""
    return [
        re.sub(r"\n\s+", " ", item.strip())
        for item in re.split(r"(?m)^\d+\.\s+", section)[1:]
    ]


def entries(section: str) -> list:
    """Exploration entries as [(title, body)]: the bold lead as the title.
    Notes written without one get their first clause (or first words) as a
    title and keep the whole text as the body, so nothing is cut."""
    out = []
    for text in items(section):
        lead = re.match(r"\*\*(.+?)\*\*\s*(.*)", text, re.S)
        if lead:
            out.append((lead.group(1).rstrip("."), lead.group(2)))
            continue
        clause = re.match(r"(.{12,110}?)[.:;,](\s|$)", text)
        if clause:
            title = clause.group(1)
        else:
            words = text.split()
            title = " ".join(words[:12]) + (" …" if len(words) > 12 else "")
        out.append((title, text))
    return out


def activity(wt: Path, name: str) -> pd.DataFrame:
    """The agent's writes to its notes, scripts and rule, from its session log."""
    folder = SESSIONS / re.sub(r"[^A-Za-z0-9]", "-", str(wt))
    rows = []
    # the agent runs as a subagent, logged under subagents/
    for log in folder.rglob("*.jsonl") if folder.is_dir() else []:
        for line in log.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            content = (event.get("message") or {}).get("content")
            if event.get("type") != "assistant" or not isinstance(content, list):
                continue
            for call in content:
                if call.get("type") != "tool_use" or call.get("name") not in (
                    "Write",
                    "Edit",
                ):
                    continue
                path = call["input"].get("file_path", "")
                kind = (
                    "notes"
                    if f"notes/policy_finder/{name}" in path
                    else (
                        "rule"
                        if "rule_based/" in path
                        else (
                            "scripts"
                            if f"scripts/policy_finder/{name}/" in path
                            else None
                        )
                    )
                )
                if kind:
                    rows.append(
                        {
                            "time": event["timestamp"],
                            "kind": kind,
                            "file": Path(path).name,
                            "action": call["name"],
                        }
                    )
    frame = pd.DataFrame(rows, columns=["time", "kind", "file", "action"])
    frame["time"] = pd.to_datetime(frame["time"], utc=True).dt.tz_convert(None)
    return frame.sort_values("time")


def activity_chart(frame: pd.DataFrame):
    kinds = ["notes", "scripts", "rule"]
    return (
        alt.Chart(frame)
        .mark_circle(size=110, opacity=0.9, stroke="white", strokeWidth=0.8)
        .encode(
            x=alt.X(
                "time:T",
                title=None,
                # one label per minute at most (tickMinStep is in ms)
                axis=alt.Axis(format="%H:%M", tickMinStep=60_000),
            ),
            y=alt.Y("kind:N", sort=kinds, title=None),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(domain=kinds, range=[CLAY, MUTED, INK]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("time:T", format="%H:%M:%S"),
                "kind",
                "file",
                "action",
            ],
        )
        .properties(height=130)
        .configure_view(stroke=None)
        .configure_axis(
            labelFont="Geist Mono",
            labelColor=MUTED,
            gridColor=LINE,
            gridOpacity=0.5,
            domain=False,
            tickColor=LINE,
        )
        .configure(background="transparent")
    )


def show_notes(wt: Path, name: str, text: str):
    sections = parse_notes(text)
    trail = entries(sections.get("Explorations", ""))
    findings = items(sections.get("Key findings", ""))
    log = activity(wt, name)
    scripts_dir = wt / "scripts/policy_finder" / name
    n_scripts = (
        len([p for p in scripts_dir.glob("*") if p.is_file()])
        if scripts_dir.is_dir()
        else 0
    )
    stats(
        [
            ("Exploration entries", f"{len(trail)}"),
            ("Scripts", f"{n_scripts}"),
            ("Key findings", f"{len(findings)}"),
            ("Notes writes", f"{int((log.kind == 'notes').sum())}"),
        ]
    )
    if len(log):
        st.markdown('<div class="section">Pace</div>', unsafe_allow_html=True)
        st.caption(
            "Every write to the notes (clay), the analysis scripts and the rule"
            " (ink), from the agent's session log. Notes that keep pace sit"
            " between the scripts."
        )
        st.altair_chart(activity_chart(log), use_container_width=True)

    st.markdown('<div class="section">Explorations</div>', unsafe_allow_html=True)
    if not trail:
        empty("No entries yet.")
    for i, (title, body) in enumerate(trail, 1):
        with st.container(key=f"tl-{i}"):
            title_html = re.sub(r"`([^`]+)`", r"<code>\1</code>", esc(title))
            head = f'<span class="tl-num">{i}</span><span>{title_html}</span>'
            st.markdown(f'<div class="tl-head">{head}</div>', unsafe_allow_html=True)
            if body:
                st.markdown(body)

    if findings:
        st.markdown('<div class="section">Key findings</div>', unsafe_allow_html=True)
        for i, text in enumerate(findings, 1):
            with st.container(key=f"kf-{i}"):
                st.markdown(f"**{i}.** {text}")
    if sections.get("Hypothesis"):
        st.markdown('<div class="section">Hypothesis</div>', unsafe_allow_html=True)
        with st.container(key="hypothesis"):
            st.markdown(sections["Hypothesis"])
    if sections.get("Justification"):
        with st.expander("Justification"):
            st.markdown(sections["Justification"])
    with st.expander("Raw notes"):
        st.code(text, language="markdown")


def fmt_range(spec) -> str:
    if isinstance(spec, list) and len(spec) >= 2:
        scale = "  ·  log" if len(spec) == 3 else ""
        return f"{spec[0]:g} – {spec[1]:g}{scale}"
    return f"fixed at {spec:g}" if isinstance(spec, (int, float)) else esc(spec)


def show_rule(text: str):
    try:
        rule = yaml.safe_load(text)
    except yaml.YAMLError as e:
        st.warning(f"not valid YAML yet: {e}")
        st.code(text, language="yaml")
        return
    params = rule.get("params") or {}
    sweep = rule.get("sweep_config") or {}
    cards = []
    for name, spec in params.items():
        spec = spec if isinstance(spec, dict) else {"definition": spec}
        cards.append(
            f'<div class="card"><div class="name">{esc(name)}'
            f'<span class="chip">{esc(spec.get("type", "?"))}</span></div>'
            f'<div class="def">{esc(spec.get("definition", ""))}</div>'
            f'<div class="range">{fmt_range(sweep.get(name, "no range"))}</div></div>'
        )
    st.markdown(f'<div class="cards">{"".join(cards)}</div>', unsafe_allow_html=True)
    if rule.get("constraints"):
        chips = "".join(
            f'<span class="chip">{esc(c)}</span>' for c in rule["constraints"]
        )
        st.markdown(
            f'<div class="chips"><span class="eyebrow">Constraints</span>{chips}</div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="section">Code</div>', unsafe_allow_html=True)
    st.code(rule.get("code", ""), language="python")
    with st.expander("Rule YAML"):
        st.code(text, language="yaml")


def show_scripts(wt: Path, name: str):
    folder = wt / "scripts/policy_finder" / name
    files = (
        sorted(p for p in folder.rglob("*") if p.is_file()) if folder.is_dir() else []
    )
    if not files:
        empty("No scripts yet.")
        return
    labels = [str(p.relative_to(folder)) for p in files]
    pick = st.selectbox(f"{len(files)} files", labels, key=f"script-{name}")
    path = folder / pick
    language = {".py": "python", ".yml": "yaml", ".yaml": "yaml", ".md": "markdown"}
    text = read(path)
    if text is None:
        empty(f"{pick} is not a text file.")
    else:
        st.code(text, language=language.get(path.suffix, None))


@st.cache_data(show_spinner=False)
def scores_vs_ah(per_round: str, mtime: float) -> pd.DataFrame:
    """Definition 4 per manager that played `ah` (pool_scores, pandas only)."""
    from aimanager.simulation.pool_scores import add_pool, against_anchor

    scores = against_anchor(add_pool(pd.read_parquet(per_round)), "ah")
    return pd.DataFrame(
        [
            {
                "manager": m,
                "episodes": len(s),
                "pool": s["pool"].mean(),
                "pool se": s["pool"].std(ddof=1) / len(s) ** 0.5,
                "members": s["members"].mean(),
            }
            for m, s in scores.items()
        ]
    )


#: Reference pools for the Sweep tab, {"source": ..., "rows": [{"name", "pool",
#: "members" (optional)}]}. Kept beside the instances, outside every worktree,
#: so no agent ever reads them; the tab leaves the section out without it.
REFERENCES = WT_ROOT / "references.json"


def reference_html(best_pool: float) -> str:
    """The Reference section: each reference pool and the best point against it."""
    refs = references()
    if not refs.get("rows"):
        return ""
    tiles = "".join(
        f'<div class="stat"><div class="k">{esc(r["name"])}</div>'
        f'<div class="v">{r["pool"]:.1f}'
        + (f'<small>{r["members"]:.2f} members</small>' if "members" in r else "")
        + f'</div><div class="k">best point {best_pool - r["pool"]:+.1f}</div></div>'
        for r in refs.get("rows", [])
    )
    return (
        '<div class="section">Reference</div>'
        f'<div class="refs">{tiles}</div>'
        f'<div class="ref-src">{esc(refs.get("source", ""))}</div>'
    )


def zero_pool():
    """`zero`'s pool against `ah` in the reference sim, or None."""
    per_round = REFERENCE / "per_round.parquet"
    try:
        table = scores_vs_ah(str(per_round), per_round.stat().st_mtime)
        return float(table.set_index("manager").loc["zero", "pool"])
    except Exception:
        return None


def stats(tiles):
    cells = "".join(
        f'<div class="stat"><div class="k">{esc(k)}</div>'
        f'<div class="v">{v}</div></div>'
        for k, v in tiles
    )
    st.markdown(f'<div class="stats">{cells}</div>', unsafe_allow_html=True)


#: Width of one landscape panel; a row of them scrolls sideways when it is wider
#: than the page
LANDSCAPE_WIDTH = 210


def landscape_axes(chart):
    """The landscape's axis styling (configure() would drop it, so it is set
    with configure_axis and a transparent background property)."""
    return (
        chart.properties(background="transparent")
        .configure_view(stroke=None)
        .configure_axis(
            labelFont="Geist Mono",
            titleFont="Geist",
            labelColor=MUTED,
            titleColor=MUTED,
            gridColor=LINE,
            gridOpacity=0.5,
            domain=False,
            tickColor=LINE,
        )
    )


def landscape(points: pd.DataFrame, param: str, zero):
    """One param's panel, on its own (the export draws one per param)."""
    return landscape_axes(landscape_layer(points, param, zero))


def landscape_row(points: pd.DataFrame, params: list, zero):
    """Every param's panel side by side, each LANDSCAPE_WIDTH wide."""
    panels = [
        landscape_layer(points, p, zero).properties(width=LANDSCAPE_WIDTH) for p in params
    ]
    return landscape_axes(alt.hconcat(*panels, spacing=18))


def landscape_layer(points: pd.DataFrame, param: str, zero):
    """Pool against one param: all points, the top 10 in clay, the best in ink."""
    integral = bool((points[param] % 1 == 0).all())
    rank = points["pool"].rank(ascending=False, method="first")
    data = points.assign(
        group=pd.cut(rank, [0, 1, 10, len(points)], labels=["best", "top 10", "other"])
    )
    dots = (
        alt.Chart(data)
        .mark_circle(stroke="white", strokeWidth=0.6)
        .encode(
            x=alt.X(
                f"{param}:Q",
                title=param,
                # whole-number ticks for an int param, not 0.5 steps
                axis=alt.Axis(tickMinStep=1) if integral else alt.Axis(),
            ),
            y=alt.Y("pool:Q", title="pool vs ah", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "group:N",
                scale=alt.Scale(
                    domain=["best", "top 10", "other"], range=[INK, CLAY, MUTED]
                ),
                legend=None,
            ),
            size=alt.Size(
                "group:N",
                scale=alt.Scale(
                    domain=["best", "top 10", "other"], range=[150, 60, 26]
                ),
                legend=None,
            ),
            opacity=alt.Opacity(
                "group:N",
                scale=alt.Scale(
                    domain=["best", "top 10", "other"], range=[1, 0.9, 0.4]
                ),
                legend=None,
            ),
            order=alt.Order("pool:Q"),
            tooltip=["name", param, alt.Tooltip("pool:Q", format=".1f")],
        )
    )
    layers = [dots]
    if zero is not None:
        rule = alt.Chart(pd.DataFrame({"y": [zero]})).mark_rule(
            strokeDash=[4, 4], color=MUTED
        )
        layers.append(rule.encode(y="y:Q"))
    return alt.layer(*layers).properties(height=220)


def show_sweep(wt: Path, name: str):
    folder = wt / SWEEP_DIR.format(name=name)
    path = folder / "sweep.json"
    if not path.exists():
        empty("Not swept yet.")
        return
    sweep = json.loads(path.read_text())
    points = pd.DataFrame(
        [
            {**p["params"], **{k: p[k] for k in ("name", "pool", "pool_se", "members")}}
            for p in sweep["points"]
        ]
    ).sort_values("pool", ascending=False)
    best = points.iloc[0]
    near = int((points.pool >= best.pool - 2 * best.pool_se).sum())
    zero = zero_pool()
    stats(
        [
            ("Best pool vs ah", f"{best.pool:.1f}<small>± {best.pool_se:.1f}</small>"),
            ("Median point", f"{points.pool.median():.1f}"),
            ("Members, best point", f"{best.members:.2f}"),
            ("Within 2 se of best", f"{near}<small>/ {len(points)}</small>"),
        ]
    )
    refs = reference_html(best.pool)
    if refs:
        st.markdown(refs, unsafe_allow_html=True)
    chips = "".join(
        f'<span class="chip">{esc(k)} <b>{v:g}</b></span>'
        for k, v in sweep["best"].items()
    )
    st.markdown(
        f'<div class="chips"><span class="eyebrow">Best</span>{chips}</div>',
        unsafe_allow_html=True,
    )
    plots = [(folder / f, c) for f, c in BEST_PLOTS.items() if (folder / f).exists()]
    if plots:
        st.markdown('<div class="section">Best point</div>', unsafe_allow_html=True)
        st.html(figures_html(plots))
    params = list(sweep["best"])
    st.markdown('<div class="section">Landscape</div>', unsafe_allow_html=True)
    st.caption(
        "Each dot is a design point. Ink: the best; clay: the top 10; "
        "dashed: never punishing against ah."
    )
    # one chart, fixed-width panels; its box scrolls sideways (dashboard.css)
    with st.container(key="landscape"):
        st.altair_chart(landscape_row(points, params, zero), theme=None)
    st.markdown('<div class="section">Top 10</div>', unsafe_allow_html=True)
    st.html(top10_html(points, params))


def figures_html(plots: list) -> str:
    """The best point's plots, embedded in the page: no media files to expire
    when the app refreshes."""
    figs = "".join(
        f'<figure><img src="data:image/jpeg;base64,'
        f'{base64.b64encode(path.read_bytes()).decode()}" alt="{esc(cap)}">'
        f"<figcaption>{esc(cap)}</figcaption></figure>"
        for path, cap in plots
    )
    return f'<div class="figs">{figs}</div>'


def top10_html(points: pd.DataFrame, params: list) -> str:
    """The sweep's ten best points, in the reports' table style."""
    cols = ["name", *params, "pool", "pool_se", "members"]
    head = "".join(f"<th>{esc(c)}</th>" for c in cols)
    rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{esc(r[c] if isinstance(r[c], str) else f'{r[c]:.3g}')}</td>"
            for c in cols
        )
        + "</tr>"
        for _, r in points.head(10).iterrows()
    )
    return (
        f'<div class="table"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
    )


def show_instance(wt: Path, key: str):
    name = wt.name
    config = json.loads(read(wt / ".claude/policy_finder.json") or "{}")
    st.markdown(
        f'<div class="inst"><h2>{esc(name)}</h2>{pill(key)}</div>'
        f'<div class="meta">params {config.get("min_params", "?")}–'
        f'{config.get("max_params", "?")}  ·  policy-finder/{esc(name)}  ·  '
        f"{esc(wt)}</div>",
        unsafe_allow_html=True,
    )
    tabs = st.tabs(["Notes", "Rule", "Scripts", "Report", "Sweep"])
    with tabs[0]:
        notes = read(wt / "notes/policy_finder" / f"{name}.md")
        if notes:
            show_notes(wt, name, notes)
        else:
            empty("No notes yet.")
    with tabs[1]:
        rule = read(wt / "configs/managers/rule_based" / f"{name}.yml")
        if rule:
            show_rule(rule)
        else:
            empty("No rule yet.")
    with tabs[2]:
        show_scripts(wt, name)
    with tabs[3]:
        report = read(WT_ROOT / f"{name}.report.md")
        if report:
            doc(report, "report")
        else:
            empty("No report yet: still running, or stopped.")
    with tabs[4]:
        show_sweep(wt, name)


def counts_html(found: list) -> str:
    """One line of totals for the overview: instances, sweeps, best so far."""
    bests = []
    for wt in found:
        path = wt / SWEEP_DIR.format(name=wt.name) / "sweep.json"
        try:
            sweep = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        best = max(sweep["points"], key=lambda p: p["pool"])
        bests.append((best["pool"], best.get("pool_se") or 0, wt.name))
    line = [f"<span><b>{len(found)}</b>instances</span>", f"<span><b>{len(bests)}</b>swept</span>"]
    if bests:
        pool, se, name = max(bests)
        line.append(
            f"<span><b>{pool:.1f}</b>best pool so far, ± {se:.1f}, {esc(name)}</span>"
        )
    return f'<div class="ov-counts">{"".join(line)}</div>'


def overview(found: list) -> str:
    text = read(OVERVIEW)
    body = text if text else f'<div class="empty">No overview: {esc(OVERVIEW)}</div>'
    return counts_html(found) + body


def references() -> dict:
    """references.json, or an empty set of rows."""
    try:
        return json.loads(REFERENCES.read_text())
    except (OSError, ValueError):
        return {"rows": []}


#: The Summary's categories and each rule's place in them, written by the
#: policy-finder-summarizer skill (summarize.py documents the file). Kept
#: beside the instances, outside every worktree.
SUMMARY = WT_ROOT / "summary.json"
NOT_SUMMARIZED = "not summarized"


def load_summary() -> dict:
    try:
        data = json.loads(SUMMARY.read_text())
    except (OSError, ValueError):
        data = {}
    data.setdefault("categories", {})
    data.setdefault("instances", {})
    return data


def categories(summary: dict, kind: str) -> list:
    return [c for c, spec in summary["categories"].items() if spec["kind"] == kind]


def rule_sha(path: Path):
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return None


MECHANISM_CAPTION = (
    "Categories Claude drew up by reading across the rule configs and grouping"
    " their strategies (the policy-finder-summarizer skill); a rule changed or"
    " added since shows as not summarized."
)


def summary_frame(found: list, keys: dict) -> pd.DataFrame:
    """One row per instance: its rule's make-up, its sweep and its notes."""
    from aimanager.manager.rule import RULE_INPUTS

    summary = load_summary()
    rows = []
    for wt in found:
        name = wt.name
        row = {"name": name, "status": STATUS[keys[name]]}
        rule_path = wt / "configs/managers/rule_based" / f"{name}.yml"
        text = read(rule_path)
        try:
            rule = yaml.safe_load(text) if text else None
            code = rule["code"] if isinstance(rule, dict) else None
        except (KeyError, yaml.YAMLError):  # a rule still being written
            rule, code = None, None
        if code:
            row["params"] = len(rule.get("params") or {})
            row["inputs"] = [i for i in RULE_INPUTS if re.search(rf"\b{i}\b", code)]
            entry = summary["instances"].get(name)
            if entry and entry.get("rule_sha") == rule_sha(rule_path):
                row.update(
                    bar=entry["bar"], mechanisms=entry["mechanisms"], idea=entry["idea"]
                )
            else:  # new, or the rule changed since it was summarized
                row.update(bar=NOT_SUMMARIZED, mechanisms=[], idea="")
        try:
            sweep = json.loads((wt / SWEEP_DIR.format(name=name) / "sweep.json").read_text())
        except (OSError, ValueError):
            sweep = None
        if sweep:
            pools = pd.Series([p["pool"] for p in sweep["points"]])
            best = max(sweep["points"], key=lambda p: p["pool"])
            se = best.get("pool_se") or 0
            row.update(
                {
                    "best pool": best["pool"],
                    "se": se,
                    "members": best.get("members"),
                    "median point": float(pools.median()),
                    "within 2 se": int((pools >= best["pool"] - 2 * se).sum()),
                    "points": len(pools),
                }
            )
        notes = read(wt / "notes/policy_finder" / f"{name}.md")
        if notes:
            sections = parse_notes(notes)
            row["explorations"] = len(entries(sections.get("Explorations", "")))
            row["findings"] = len(items(sections.get("Key findings", "")))
        scripts = wt / "scripts/policy_finder" / name
        row["scripts"] = (
            len([p for p in scripts.glob("*") if p.is_file()]) if scripts.is_dir() else 0
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    if "best pool" not in frame:
        frame["best pool"] = float("nan")
    return frame.sort_values("best pool", ascending=False, na_position="last")


def pool_chart(frame: pd.DataFrame, refs: list):
    """Each agent's best pool ± se against ah, the references as dashed lines."""
    data = frame.dropna(subset=["best pool"]).assign(
        lo=lambda f: f["best pool"] - f["se"], hi=lambda f: f["best pool"] + f["se"]
    )
    ref = pd.DataFrame(refs, columns=["name", "pool"]) if refs else None
    values = list(data.lo) + list(data.hi) + (list(ref.pool) if ref is not None else [])
    scale = alt.Scale(
        domain=[math.floor(min(values)) - 1, math.ceil(max(values)) + 1], zero=False
    )
    y = alt.Y(
        "name:N", sort=alt.EncodingSortField("best pool", order="descending"), title=None
    )
    bars = (
        alt.Chart(data)
        .mark_rule(strokeWidth=2.5, color=LINE)
        .encode(
            x=alt.X(
                "lo:Q",
                scale=scale,
                title="best pool vs ah",
                axis=alt.Axis(format="d", tickMinStep=1),
            ),
            x2="hi:Q",
            y=y,
            detail="best pool:Q",
        )
    )
    dots = (
        alt.Chart(data)
        .mark_circle(size=140, opacity=1, stroke="white", strokeWidth=1, color=INK)
        .encode(
            x=alt.X("best pool:Q", scale=scale),
            y=y,
            tooltip=[
                alt.Tooltip("name:N", title="agent"),
                alt.Tooltip("best pool:Q", title="best pool vs ah", format=".1f"),
                alt.Tooltip("se:Q", title="± se", format=".1f"),
                alt.Tooltip("members:Q", title="members", format=".2f"),
            ],
        )
    )
    layers = [bars, dots]
    if ref is not None:
        rules = alt.Chart(ref).mark_rule(strokeDash=[5, 4], color=MUTED).encode(x=alt.X("pool:Q", scale=scale))
        labels = (
            alt.Chart(ref)
            .mark_text(align="left", dx=4, dy=-6, baseline="bottom", color=MUTED, fontSize=11)
            .encode(x=alt.X("pool:Q", scale=scale), y=alt.value(0), text="name:N")
        )
        layers += [rules, labels]
    return (
        alt.layer(*layers)
        .properties(height={"step": 30}, padding={"top": 18})
        .configure_view(stroke=None)
        .properties(background="transparent")
        .configure_axis(
            labelColor=MUTED,
            titleColor=MUTED,
            gridColor=LINE,
            gridOpacity=0.35,
            domain=False,
            tickColor=LINE,
            labelFont="Geist Mono",
            titleFont="Geist",
            titleFontWeight=500,
        )
        .configure_text(font="Geist")
    )


def inputs_chart(frame: pd.DataFrame):
    """Which inputs each rule reads: a filled cell per input read."""
    from aimanager.manager.rule import RULE_INPUTS

    data = frame.dropna(subset=["inputs"])
    cells = pd.DataFrame(
        [
            {
                "name": r["name"],
                "input": i,
                "read": "read" if i in r["inputs"] else "not read",
            }
            for _, r in data.iterrows()
            for i in RULE_INPUTS
        ]
    )
    return (
        alt.Chart(cells)
        .mark_rect(cornerRadius=3, tooltip=False)  # self-explanatory, no hover
        .encode(
            x=alt.X(
                "input:N",
                sort=list(RULE_INPUTS),
                title=None,
                axis=alt.Axis(orient="top", labelAngle=0),
                scale=alt.Scale(paddingInner=0.12),
            ),
            y=alt.Y(
                "name:N",
                sort=list(data.name),
                title=None,
                scale=alt.Scale(paddingInner=0.18),
            ),
            color=alt.Color(
                "read:N", scale=alt.Scale(domain=["read", "not read"], range=[CLAY, LINE]), legend=None
            ),
        )
        .properties(height=30 * len(data))
        .configure_view(stroke=None)
        .properties(background="transparent")
        .configure_axis(labelColor=MUTED, domain=False, ticks=False, labelFont="Geist Mono")
    )


def mechanism_counts(frame: pd.DataFrame) -> list:
    """(label, rules in it, rules summarized, meaning) for every category."""
    summary = load_summary()
    ruled = frame.dropna(subset=["inputs"])
    ruled = ruled[ruled.bar != NOT_SUMMARIZED]
    n = len(ruled)
    meaning = {c: spec["meaning"] for c, spec in summary["categories"].items()}
    out = [
        (f"below {b}", int((ruled.bar == b).sum()), n, meaning[b])
        for b in categories(summary, "bar")
    ]
    out += [
        (m, int(ruled.mechanisms.apply(lambda ms: m in ms).sum()), n, meaning[m])
        for m in categories(summary, "mechanism")
    ]
    return out


def summary_stats(frame: pd.DataFrame) -> list:
    swept = frame.dropna(subset=["best pool"])
    if swept.empty:
        return [("Instances", f"{len(frame)}"), ("Swept", "0")]
    top = swept.iloc[0]
    return [
        ("Instances", f"{len(frame)}"),
        ("Swept", f"{len(swept)}"),
        ("Best pool", f"{top['best pool']:.1f}<small>{esc(top['name'])}</small>"),
        ("Median of the bests", f"{swept['best pool'].median():.1f}"),
    ]


SUMMARY_CAPTION = (
    "Each agent's best design point against ah, ± its standard error over 500"
    " episodes. Dashed: the references, on 1000 episodes. A best point is the top"
    " of a noisy sweep, so it reads high by luck."
)
#: The colour key under the pool chart: who each kind of rule punishes
def reference_chips(refs: dict) -> str:
    chips = "".join(
        f'<span class="chip">{esc(r["name"])} <b>{r["pool"]:.1f}</b></span>'
        for r in refs.get("rows", [])
    )
    if not chips:
        return ""
    return (
        f'<div class="chips"><span class="eyebrow">References</span>{chips}</div>'
        f'<div class="ref-src">{esc(refs.get("source", ""))}</div>'
    )


def mechanism_html(frame: pd.DataFrame) -> str:
    rows = "".join(
        f'<div class="mech"><span>{esc(label)}</span>'
        f'<span class="mech-bar"><i style="width:{100 * k / max(n, 1):.0f}%"></i></span>'
        f'<span class="mech-n">{k}/{n}</span>'
        f'<span class="mech-def">{esc(meaning)}</span></div>'
        for label, k, n, meaning in mechanism_counts(frame)
    )
    return f'<div class="mechs">{rows}</div>'


def agents_table_html(frame: pd.DataFrame, refs: dict) -> str:
    """The Summary's agents, one row each; a row opens its agent (the app
    through a `?agent=` link, the export through its data-open row)."""
    swept = frame["best pool"].dropna()
    lo = min([60.0] + [r["pool"] for r in refs.get("rows", [])]) - 5
    hi = max([75.0] + list(swept))

    def num(row, key, fmt):
        v = row.get(key)
        return format(v, fmt) if v is not None and v == v else ""

    rows = []
    for _, r in frame.iterrows():
        pool = r.get("best pool")
        if pool == pool:  # swept
            width = 100 * (pool - lo) / (hi - lo)
            cell = (
                f'<div class="poolcell"><span class="pbar"><i style="width:{width:.0f}%">'
                f'</i></span><span class="pv">{pool:.1f}</span>'
                f'<span class="se">± {r["se"]:.1f}</span></div>'
            )
        else:
            cell = '<span class="se">not swept</span>'
        mechs = r.get("mechanisms")
        tags = "".join(
            f'<span class="chip">{esc(m)}</span>'
            for m in (mechs if isinstance(mechs, list) else [])
        )
        bar = r.get("bar") if isinstance(r.get("bar"), str) else ""
        name = esc(r["name"])
        idea = r.get("idea") if isinstance(r.get("idea"), str) else ""
        rows.append(
            f'<tr data-open="{name}"><td class="name">'
            f'<a class="open" href="?agent={name}" target="_self">{name}</a>'
            + (f'<div class="idea">{esc(idea)}</div>' if idea else "")
            + "</td>"
            f'<td class="txt">{cell}</td><td>{num(r, "members", ".2f")}</td>'
            f'<td>{num(r, "within 2 se", ".0f")}</td><td>{num(r, "params", ".0f")}</td>'
            f'<td class="txt">{esc(bar)}</td><td class="txt tags">{tags}</td></tr>'
        )
    heads = ("agent", "best pool", "members", "within 2 se", "params", "punishes below",
             "mechanisms")
    head = "".join(
        f'<th class="{"txt" if h in ("agent", "best pool", "punishes below", "mechanisms") else ""}">'
        f"{h}</th>"
        for h in heads
    )
    return (
        f'<div class="agents"><table><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


#: Agents the Summary's chart, table and input grid show: the best by pool
SUMMARY_TOP = 10


def top_note(frame: pd.DataFrame) -> str:
    """A note when the lists show only the top agents."""
    if len(frame) <= SUMMARY_TOP:
        return ""
    return (
        f"Showing the {SUMMARY_TOP} best of {len(frame)} agents; every agent is in the"
        " Agent browser."
    )


def show_summary(found: list):
    running = running_names()
    keys = {w.name: status(w.name, running) for w in found}
    frame = summary_frame(found, keys)
    refs = references()
    shown = frame.head(SUMMARY_TOP)
    stats(summary_stats(frame))
    if shown["best pool"].notna().any():
        st.markdown('<div class="section">Best pool against ah</div>', unsafe_allow_html=True)
        st.caption(" ".join(filter(None, [SUMMARY_CAPTION, top_note(frame)])))
        ref_rows = [(r["name"], r["pool"]) for r in refs.get("rows", [])]
        # theme=None: the chart's own axis styling, not Streamlit's
        st.altair_chart(pool_chart(shown, ref_rows), use_container_width=True, theme=None)
        st.html(reference_chips(refs))
    st.markdown('<div class="section">Agents</div>', unsafe_allow_html=True)
    st.caption("Tap a row to open the agent.")
    st.html(agents_table_html(shown, refs))
    if "inputs" in frame:
        st.markdown('<div class="section">What the rules do</div>', unsafe_allow_html=True)
        st.caption(MECHANISM_CAPTION)
        st.html(mechanism_html(frame))  # every agent
        st.markdown('<div class="section">What the rules read</div>', unsafe_allow_html=True)
        st.altair_chart(inputs_chart(shown), use_container_width=True, theme=None)


@st.fragment(run_every=REFRESH_S)
def board():
    found = instances()
    st.caption(f"{len(found)} under {WT_ROOT} · refreshes every {REFRESH_S} s")
    if not found:
        empty("No instances yet. Start one with the policy-finder skill.")
        return
    running = running_names()
    keys = {d.name: status(d.name, running) for d in found}
    pick = st.selectbox(
        "Instance",
        [d.name for d in found],
        format_func=lambda n: f"{n}   ·   {STATUS[keys[n]]}",
        key="instance",
    )
    show_instance(WT_ROOT / pick, keys[pick])


# `streamlit run` runs this file as __main__; export_html.py imports it for
# its data and charts without drawing the page
if __name__ == "__main__":
    st.set_page_config(page_title="Policy finders", page_icon="◐", layout="wide")
    st.html(STYLE)
    st.markdown(
        '<div class="hero"><div class="eyebrow">Algorithmic institutions</div>'
        "<h1>Policy finder</h1>"
        "<p>Agents that design rule-based punishment managers, swept against the AH"
        f' manager</p><p class="data-note">{esc(DATA_NOTE)}</p></div>',
        unsafe_allow_html=True,
    )
    # a Summary row links here with ?agent=<name>: open that agent
    agent = st.query_params.get("agent")
    if agent:
        st.session_state["view"] = "Agent browser"
        st.session_state["instance"] = agent
        st.query_params.clear()
    view = st.segmented_control(
        "View",
        ["Overview", "Summary", "Agent browser"],
        default="Overview",
        key="view",
        label_visibility="collapsed",
    )
    if view == "Summary":
        show_summary(instances())
    elif view == "Agent browser":
        board()
    else:
        st.html(overview(instances()))
