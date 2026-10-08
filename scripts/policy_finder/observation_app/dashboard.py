"""Live dashboard of policy-finder instances (#236, #227).

Reads the instance worktrees (`../policy-finder-worktrees`, or
$PF_WORKTREE_ROOT) and this checkout's sweep outputs, and refreshes every few
seconds: per instance its status, rule, notes, scripts, final report and, once
swept, its sweep result. Read-only: it never writes to a worktree.

Run from the repo root (streamlit is not a project dependency):
    uv run --no-sync --with streamlit streamlit run scripts/policy_finder/observation_app/dashboard.py
"""

import html
import json
import os
import re
import subprocess
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import yaml

MAIN = Path(__file__).resolve().parents[3]
WT_ROOT = Path(
    os.environ.get("PF_WORKTREE_ROOT", MAIN.parent / "policy-finder-worktrees")
)
BASE = os.environ.get("PF_BASE", "policy-finder-base")
SWEEPS = MAIN / "plots/simulation/policy_finder"
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
    f"{(Path(__file__).parent / 'dashboard.css').read_text()}</style>"
)

STATUS = {
    "running": "running",
    "checked": "checked",
    "done": "done, not checked",
    "stopped": "stopped, no report",
}


def esc(text) -> str:
    return html.escape(str(text))


def read(path: Path):
    try:
        return path.read_text()
    except OSError:
        return None


def instances():
    if not WT_ROOT.is_dir():
        return []
    dirs = [d for d in WT_ROOT.iterdir() if d.is_dir() and d.name.startswith("pf-")]
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
    st.code(read(path) or "", language=language.get(path.suffix, None))


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


def landscape(points: pd.DataFrame, param: str, zero):
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
    return (
        alt.layer(*layers)
        .properties(height=220)
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
        .configure(background="transparent")
    )


def show_sweep(name: str):
    path = SWEEPS / f"{name}_sweep" / "sweep.json"
    if not path.exists():
        empty(f"No merged sweep yet: {path.relative_to(MAIN)}")
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
            ("Never punish vs ah", f"{zero:.1f}" if zero is not None else "–"),
            ("Within 2 se of best", f"{near}<small>/ {len(points)}</small>"),
        ]
    )
    chips = "".join(
        f'<span class="chip">{esc(k)} <b>{v:g}</b></span>'
        for k, v in sweep["best"].items()
    )
    st.markdown(
        f'<div class="chips"><span class="eyebrow">Best</span>{chips}</div>',
        unsafe_allow_html=True,
    )
    params = list(sweep["best"])
    st.markdown('<div class="section">Landscape</div>', unsafe_allow_html=True)
    st.caption(
        "Each dot is a design point. Ink: the best; clay: the top 10; "
        "dashed: never punishing against ah."
    )
    cols = st.columns(len(params))
    for col, p in zip(cols, params):
        col.altair_chart(landscape(points, p, zero), use_container_width=True)
    st.markdown('<div class="section">Top 10</div>', unsafe_allow_html=True)
    st.dataframe(
        points.head(10)[["name", *params, "pool", "pool_se", "members"]].round(3),
        hide_index=True,
        width="stretch",
    )
    for other in sorted(SWEEPS.glob(f"{name}_*")):
        per_round = other / "per_round.parquet"
        if other.name.startswith(f"{name}_sweep") or not per_round.exists():
            continue
        st.markdown(
            f'<div class="section">{esc(other.name)}</div>', unsafe_allow_html=True
        )
        try:
            table = scores_vs_ah(str(per_round), per_round.stat().st_mtime)
            st.dataframe(table.round(2), hide_index=True, width="stretch")
        except Exception as e:  # never break the page over one run
            st.caption(f"could not score: {e}")


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
        show_sweep(name)


st.set_page_config(page_title="Policy finders", page_icon="◐", layout="wide")
st.html(STYLE)


@st.fragment(run_every=REFRESH_S)
def board():
    found = instances()
    st.markdown(
        '<div class="hero"><div class="eyebrow">Policy finder</div>'
        "<h1>Instances</h1>"
        f"<p>{len(found)} under {esc(WT_ROOT)} · refreshes every {REFRESH_S} s</p>"
        "</div>",
        unsafe_allow_html=True,
    )
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


board()
