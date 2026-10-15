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
from summarize import OBSERVATORY, versions
import pandas as pd
import streamlit as st
import yaml

MAIN = Path(__file__).resolve().parents[3]
WT_ROOT = Path(
    os.environ.get("PF_WORKTREE_ROOT", MAIN.parent / "policy-finder-worktrees")
)
BASE = os.environ.get("PF_BASE", "policy-finder-base")
#: An instance's merged sweep, relative to its worktree (run_sweep.sh), with
#: its Gaussian process fit (landscapes.json) and, in validate/, the
#: validation sim whose best is the rule's final result (#238)
SWEEP_DIR = "plots/simulation/policy_finder/{name}_sweep"
LANDSCAPES = "landscapes.json"
VALIDATE = "validate"
#: The final best's matchup against `ah` (simulation/sweep.py's BEST_PLOTS)
BEST_PLOTS = {
    "comparison_pairing_side.jpg": "Final best vs ah, by side",
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


def committed(wt: Path, version: str) -> bool:
    """Whether the version's rule is committed on the branch (the check commits)."""
    rule = f"configs/managers/rule_based/{version}.yml"
    out = subprocess.run(
        ["git", "-C", str(wt), "ls-files", "--error-unmatch", rule],
        capture_output=True,
    )
    return out.returncode == 0


def status(wt: Path, version: str, running: set) -> str:
    """A version's status; a running launcher runs the instance's latest."""
    if wt.name in running and version == versions(wt.name)[-1]:
        return "running"
    report = WT_ROOT / f"{version}.report.md"
    if report.exists() and report.stat().st_size > 0:  # the launcher makes it empty
        return "checked" if committed(wt, version) else "done"
    return "stopped"


def version_label(version: str, all_versions: list) -> str:
    return f"v{all_versions.index(version) + 1}"


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


def activity(wt: Path, version: str) -> pd.DataFrame:
    """The agent's writes to the version's notes and rule and the instance's
    scripts, from the session logs that wrote the version's notes or rule
    (every version of an instance runs in the same worktree)."""
    folder = SESSIONS / re.sub(r"[^A-Za-z0-9]", "-", str(wt))
    notes = f"notes/policy_finder/{version}.md"
    rule = f"configs/managers/rule_based/{version}.yml"
    scripts = f"scripts/policy_finder/{wt.name}/"
    rows = []
    # the agent runs as a subagent, logged under subagents/
    for log in folder.rglob("*.jsonl") if folder.is_dir() else []:
        log_rows = []
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
                    if path.endswith(notes)
                    else (
                        "rule"
                        if path.endswith(rule)
                        else ("scripts" if scripts in path else None)
                    )
                )
                if kind:
                    log_rows.append(
                        {
                            "time": event["timestamp"],
                            "kind": kind,
                            "file": Path(path).name,
                            "action": call["name"],
                        }
                    )
        if any(r["kind"] != "scripts" for r in log_rows):  # this version's session
            rows += log_rows
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


def show_notes(wt: Path, version: str, text: str):
    sections = parse_notes(text)
    trail = entries(sections.get("Explorations", ""))
    findings = items(sections.get("Key findings", ""))
    log = activity(wt, version)
    scripts_dir = wt / "scripts/policy_finder" / wt.name
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
    """Each reference pool and the final best against it ("" without any)."""
    refs = references()
    if not refs.get("rows"):
        return ""
    tiles = "".join(
        f'<div class="stat"><div class="k">{esc(r["name"])}</div>'
        f'<div class="v">{r["pool"]:.1f}'
        + (f'<small>{r["members"]:.2f} members</small>' if "members" in r else "")
        + f'</div><div class="k">final best {best_pool - r["pool"]:+.1f}</div></div>'
        for r in refs.get("rows", [])
    )
    return (
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


def stats_html(tiles) -> str:
    cells = "".join(
        f'<div class="stat"><div class="k">{esc(k)}</div>'
        f'<div class="v">{v}</div></div>'
        for k, v in tiles
    )
    return f'<div class="stats">{cells}</div>'


def stats(tiles):
    st.markdown(stats_html(tiles), unsafe_allow_html=True)


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


def landscape(points: pd.DataFrame, land: dict, param: str, final, zero):
    """One param's panel, on its own (the export draws one per param)."""
    return landscape_axes(landscape_layer(points, land, param, final, zero))


def landscape_row(points: pd.DataFrame, land: dict, final, zero):
    """Every fitted param's panel side by side, each LANDSCAPE_WIDTH wide."""
    panels = [
        landscape_layer(points, land, p, final, zero).properties(width=LANDSCAPE_WIDTH)
        for p in land["params"]
    ]
    return landscape_axes(alt.hconcat(*panels, spacing=18))


def landscape_layer(points: pd.DataFrame, land: dict, param: str, final, zero):
    """Pool against one param: the design points, the Gaussian process's
    profile through its peak with a band of 2 sd, the peak's value dashed and
    the final best in ink."""
    spec = land["params"][param]
    # only what the panel draws, rounded: every panel embeds its own data, and
    # the full design (every param, se, members, fit) per panel made the
    # exported page tens of MB
    points = points[["name", param, "pool"]].round({param: 4, "pool": 2})
    prof = pd.DataFrame(land["profiles"][param]).rename(
        columns={"x": "_x", "mean": "_mean", "sd": "_sd"}
    )
    prof["_lo"], prof["_hi"] = prof._mean - 2 * prof._sd, prof._mean + 2 * prof._sd
    prof = prof.round(4)
    # the swept range exactly: no padding out to round or log-decade ticks
    scale = alt.Scale(
        type="log" if spec["log"] else "linear", domain=spec["range"], nice=False
    )
    # whole-number ticks for an int param, not 0.5 steps
    axis = alt.Axis(tickMinStep=1) if spec["type"] == "int" else alt.Axis()

    def x(field):
        return alt.X(f"{field}:Q", title=param, scale=scale, axis=axis)

    y = alt.Y("pool:Q", title="pool vs ah", scale=alt.Scale(zero=False))
    layers = [
        alt.Chart(points)
        .mark_circle(size=24, color=MUTED, opacity=0.4)
        .encode(
            x=x(param),
            y=y,
            tooltip=["name", param, alt.Tooltip("pool:Q", format=".1f")],
        ),
        alt.Chart(prof)
        .mark_area(color=CLAY, opacity=0.18)
        .encode(x=x("_x"), y="_lo:Q", y2="_hi:Q"),
        alt.Chart(prof)
        .mark_line(color=CLAY, strokeWidth=2)
        .encode(x=x("_x"), y="_mean:Q"),
        alt.Chart(pd.DataFrame({"_x": [land["peak"]["params"][param]]}))
        .mark_rule(color=CLAY, strokeDash=[2, 3])
        .encode(x=x("_x")),
    ]
    if zero is not None:
        layers.append(
            alt.Chart(pd.DataFrame({"_y": [zero]}))
            .mark_rule(strokeDash=[4, 4], color=MUTED)
            .encode(y="_y:Q")
        )
    if final is not None:
        layers.append(
            alt.Chart(
                pd.DataFrame(
                    {"_x": [final["params"][param]], "pool": [final["pool"]],
                     "point": [final["name"]]}
                )
            )
            .mark_circle(size=150, color=INK, opacity=1, stroke="white", strokeWidth=1)
            .encode(
                x=x("_x"),
                y=y,
                tooltip=[
                    alt.Tooltip("point:N", title="final best"),
                    alt.Tooltip("pool:Q", title="validated pool", format=".1f"),
                ],
            )
        )
    title = f"{param} · moves the pool {spec['effect']:.1f}"
    if param in land["peak"]["at_edge"]:
        title += " · peak on its edge"
    return alt.layer(*layers).properties(
        height=220,
        title=alt.TitleParams(
            title, anchor="start", font="Geist", fontSize=12, fontWeight=500,
            color=MUTED,
        ),
    )


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def sweep_files(wt: Path, name: str) -> dict:
    """An instance's sweep, its fit and its validation, each None until written."""
    folder = wt / SWEEP_DIR.format(name=name)
    return {
        "folder": folder,
        "sweep": read_json(folder / "sweep.json"),
        "land": read_json(folder / LANDSCAPES),
        "validate": read_json(folder / VALIDATE / "sweep.json"),
    }


def final_best(validate):
    """The validation's best point (its sweep.json's best_name), or None."""
    if not validate:
        return None
    return next(p for p in validate["points"] if p["name"] == validate["best_name"])


def point_source(name: str) -> str:
    """Where a validated point came from: the GP's peak or a design point."""
    return "GP peak" if name.endswith("_gp") else f"sweep {name.rsplit('_', 1)[-1]}"


VALIDATE_CAPTION = (
    "The Gaussian process's peak and the sweep's best measured points, rerun on"
    " fresh seeds at {episodes} episodes each. The top of these is the final"
    " best. Before: the point's pool in the sweep (500 episodes, read high by"
    " luck) or, for the peak, the fit's prediction."
)
LANDSCAPE_CAPTION = (
    "Grey: the design points, against each param. Clay: the Gaussian process"
    " fitted to them, cut along the param through its peak (dashed), with a band"
    " of 2 sd. Ink: the final best at its validated pool. Dashed line: never"
    " punishing against ah. Each panel's title says how far the param moves the"
    " fitted pool, and when the peak sits on the edge of its range."
)


def sweep_blocks(wt: Path, name: str, zero) -> list:
    """The Sweep tab as (kind, content) blocks, which the app and the export
    each render: "html", "section", "caption", "empty" and "landscape"
    (points, landscapes, final best). The landscape shows once the sweep is
    fitted, the final best once it is validated."""
    f = sweep_files(wt, name)
    sweep, land, validate = f["sweep"], f["land"], f["validate"]
    if not sweep:
        return [("empty", "Not swept yet.")]
    points = pd.DataFrame(
        [
            {**p["params"], **{k: p[k] for k in ("name", "pool", "pool_se", "members")}}
            for p in sweep["points"]
        ]
    ).sort_values("pool", ascending=False)
    if land:
        points["gp"] = points["name"].map({p["name"]: p["mean"] for p in land["points"]})
    final = final_best(validate)
    # the tiles are the final result only, all from the validation sim; the
    # sweep's own figures go with the sections that show the sweep
    if final:
        tiles = [
            ("Final best vs ah", f"{final['pool']:.1f}<small>± {final['pool_se']:.1f}</small>"),
            ("Members", f"{final['members']:.2f}"),
            ("From", esc(point_source(final["name"]))),
            (
                "Validated on",
                f"{len(validate['points'])} points"
                f"<small>× {validate['n_episodes']} episodes, fresh seeds</small>",
            ),
        ]
    else:
        tiles = [
            ("Final best vs ah", "–<small>awaiting validation</small>"),
            ("Members", "–"),
            ("From", "–"),
            ("Validated on", "–"),
        ]
    design = (
        f"The sweep: {len(points)} design points × {sweep['n_episodes']} episodes,"
        f" median point {points.pool.median():.1f}."
    )
    out = [("html", stats_html(tiles))]
    if final:
        refs = reference_html(final["pool"])
        if refs:
            out += [("section", "Reference"), ("html", refs)]
        chips = "".join(
            f'<span class="chip">{esc(k)} <b>{v:g}</b></span>'
            for k, v in final["params"].items()
        )
        out.append(
            (
                "html",
                f'<div class="chips"><span class="eyebrow">Final best</span>{chips}</div>',
            )
        )
        folder = f["folder"] / VALIDATE
        plots = [(folder / p, c) for p, c in BEST_PLOTS.items() if (folder / p).exists()]
        if plots:
            out += [("section", "Final best"), ("html", figures_html(plots))]
        out += [
            ("section", "Validation"),
            ("caption", VALIDATE_CAPTION.format(episodes=validate["n_episodes"])),
            ("html", validation_html(validate, sweep, land)),
        ]
    else:
        out += [
            ("section", "Validation"),
            ("empty", "The final best shows once the validation sim is done."),
        ]
    out.append(("section", "Landscape"))
    if land:
        out += [("caption", f"{design} {LANDSCAPE_CAPTION}"), ("landscape", (points, land, final))]
    else:
        out.append(("empty", "The landscape shows once the Gaussian process is fitted."))
    params = [n for n in sweep["points"][0]["params"]]
    out += [
        ("section", "Top 10 measured"),
        (
            "caption",
            f"{design} Its own ranking, one estimate of {sweep['n_episodes']}"
            " episodes per point: its top reads high by luck."
            + (" gp: the fitted pool at the point." if land else ""),
        ),
        ("html", top10_html(points, params)),
    ]
    return out


def show_sweep(wt: Path, name: str):
    zero = zero_pool()
    for kind, content in sweep_blocks(wt, name, zero):
        if kind == "section":
            st.markdown(f'<div class="section">{esc(content)}</div>', unsafe_allow_html=True)
        elif kind == "caption":
            st.caption(content)
        elif kind == "empty":
            empty(content)
        elif kind == "landscape":
            points, land, final = content
            # one chart, fixed-width panels; its box scrolls sideways (dashboard.css)
            with st.container(key="landscape"):
                st.altair_chart(landscape_row(points, land, final, zero), theme=None)
        else:
            st.html(content)


def validation_html(validate: dict, sweep: dict, land) -> str:
    """The validation's points, best first: pool, before, where from."""
    measured = {p["name"]: p["pool"] for p in sweep["points"]}
    predicted = land["peak"]["mean"] if land else None
    rows = []
    for p in sorted(validate["points"], key=lambda p: p["pool"], reverse=True):
        before = measured.get(p["name"], predicted)
        rows.append(
            "<tr>"
            f"<td>{esc(p['name'])}</td><td>{esc(point_source(p['name']))}</td>"
            f"<td>{p['pool']:.1f}</td><td>{p['pool_se']:.1f}</td>"
            f"<td>{'' if before is None else f'{before:.1f}'}</td>"
            f"<td>{p['members']:.2f}</td></tr>"
        )
    head = "".join(
        f"<th>{h}</th>"
        for h in ("point", "from", "pool", "pool_se", "before", "members")
    )
    return (
        f'<div class="table"><table><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


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
    """The sweep's ten best measured points, in the reports' table style."""
    cols = ["name", *params, "pool", "pool_se", *(["gp"] if "gp" in points else []),
            "members"]
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


def version_meta(wt: Path, version: str) -> str:
    """The version's param limits, branch and worktree, under its title. The
    instance file holds the latest version's; an earlier version shows none."""
    config = json.loads(read(wt / ".claude/policy_finder.json") or "{}")
    latest = config.get("rule_name", wt.name) == version
    limits = (
        f"params {config.get('min_params', '?')}–"
        f"{'any' if config.get('max_params', '?') is None else config.get('max_params', '?')}  ·  "
        if latest
        else ""
    )
    return f"{limits}policy-finder/{esc(wt.name)}  ·  {esc(wt)}"


def show_version(wt: Path, version: str, key: str):
    """One version of an instance: its own notes, rule, report and sweep; the
    scripts folder is the instance's, shared by every version."""
    all_versions = versions(wt.name)
    title = esc(wt.name) + (
        f' <span class="ver">{version_label(version, all_versions)}</span>'
        if len(all_versions) > 1
        else ""
    )
    st.markdown(
        f'<div class="inst"><h2>{title}</h2>{pill(key)}</div>'
        f'<div class="meta">{version_meta(wt, version)}</div>',
        unsafe_allow_html=True,
    )
    tabs = st.tabs(["Notes", "Rule", "Scripts", "Report", "Sweep"])
    with tabs[0]:
        notes = read(wt / "notes/policy_finder" / f"{version}.md")
        if notes:
            show_notes(wt, version, notes)
        else:
            empty("No notes yet.")
    with tabs[1]:
        rule = read(wt / "configs/managers/rule_based" / f"{version}.yml")
        if rule:
            show_rule(rule)
        else:
            empty("No rule yet.")
    with tabs[2]:
        show_scripts(wt, wt.name)
    with tabs[3]:
        report = read(WT_ROOT / f"{version}.report.md")
        if report:
            doc(report, "report")
        else:
            empty("No report yet: still running, or stopped.")
    with tabs[4]:
        show_sweep(wt, version)


def counts_html(found: list) -> str:
    """One line of totals, atop the Overview and the Summary alike: lineages,
    versions, validated versions and the final best of any version."""
    bests, n_versions = [], 0
    for wt in found:
        all_versions = versions(wt.name)
        n_versions += len(all_versions)
        for v in all_versions:
            best = final_best(sweep_files(wt, v)["validate"])
            if best:
                label = v if len(all_versions) == 1 else (
                    f"{wt.name} {version_label(v, all_versions)}"
                )
                bests.append((best["pool"], best["pool_se"], label))
    line = [
        f"<span><b>{len(found)}</b>lineages</span>",
        f"<span><b>{n_versions}</b>versions</span>",
        f"<span><b>{len(bests)}</b>validated</span>",
    ]
    if bests:
        pool, se, name = max(bests)
        line.append(
            f"<span><b>{pool:.1f}</b>final best so far, ± {se:.1f}, {esc(name)}</span>"
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


RULES_CAPTION = (
    "The rules the policies are built from, each lineage counted once by its"
    " best version: Claude drew them up by reading across the rule configs"
    " (the policy-finder-summarizer skill); a rule changed or added since shows"
    " as not summarized."
)


def summary_frame(found: list, running: set) -> pd.DataFrame:
    """One row per lineage (#244): its best validated version, or its latest
    while none is validated, with the version's rule, sweep and notes."""
    summary = load_summary()
    rows = []
    for wt in found:
        all_versions = versions(wt.name)
        candidates = [
            version_row(wt, v, status(wt, v, running), summary) for v in all_versions
        ]
        validated = [r for r in candidates if "best pool" in r]
        row = (
            max(validated, key=lambda r: r["best pool"]) if validated else candidates[-1]
        )
        rows.append({**row, "versions": len(all_versions)})
    frame = pd.DataFrame(rows)
    if "best pool" not in frame:
        frame["best pool"] = float("nan")
    return frame.sort_values("best pool", ascending=False, na_position="last")


def version_row(wt: Path, version: str, key: str, summary: dict) -> dict:
    """One version's rule make-up, final best and notes, for the Summary."""
    from aimanager.manager.rule import RULE_INPUTS

    name = wt.name
    all_versions = versions(name)
    row = {
        "name": name,
        "version": version,
        "version label": version_label(version, all_versions),
        "status": STATUS[key],
    }
    rule_path = wt / "configs/managers/rule_based" / f"{version}.yml"
    text = read(rule_path)
    try:
        rule = yaml.safe_load(text) if text else None
        code = rule["code"] if isinstance(rule, dict) else None
    except (KeyError, yaml.YAMLError):  # a rule still being written
        rule, code = None, None
    if code:
        row["params"] = len(rule.get("params") or {})
        row["inputs"] = [i for i in RULE_INPUTS if re.search(rf"\b{i}\b", code)]
        entry = summary["instances"].get(version)
        if entry and entry.get("rule_sha") == rule_sha(rule_path):
            row.update(
                **{
                    "policy category": entry.get("policy_category") or NOT_SUMMARIZED,
                    "rules included": entry.get("rules_included", []),
                },
                idea=entry["idea"],
            )
        else:  # new, or the rule changed since it was summarized
            row.update({"policy category": NOT_SUMMARIZED, "rules included": [], "idea": ""})
    files = sweep_files(wt, version)
    row["stage"] = (
        "validated" if files["validate"] else "validating" if files["sweep"]
        else "not swept"
    )
    best = final_best(files["validate"])
    if best:  # only a validated best: a sweep's own best reads high
        row.update(
            {
                "best pool": best["pool"],
                "se": best["pool_se"],
                "members": best["members"],
                "from": point_source(best["name"]),
            }
        )
    notes = read(wt / "notes/policy_finder" / f"{version}.md")
    if notes:
        sections = parse_notes(notes)
        row["explorations"] = len(entries(sections.get("Explorations", "")))
        row["findings"] = len(items(sections.get("Key findings", "")))
    scripts = wt / "scripts/policy_finder" / name
    row["scripts"] = (
        len([p for p in scripts.glob("*") if p.is_file()]) if scripts.is_dir() else 0
    )
    return row


#: Colours of the policy categories (summary.json, kind "policy category"); a
#: category added later takes the next of POLICY_SPARE, an unknown one grey
POLICY_COLOURS = {
    "front-loaded": CLAY,
    "switch-timed": "#4f7aa8",
    "context-adaptive": "#2f6b3f",
}
POLICY_SPARE = ["#b08968", "#8c6bb1", "#c45a8c", "#2b9c9c"]
#: How much of a lineage stays when another is picked in the evolution chart
DIMMED = 0.18
#: Line colours of the lineages that iterated
#: (mid tones: they read on both the light and the dark theme)
LINEAGE_COLOURS = ["#e6a23c", "#8c6bb1", "#2b9c9c", "#c45a8c", "#7f8c3a", "#a8a39a"]


def made(wt: Path, version: str) -> int:
    """When a version was made: its rule's first commit (0 if uncommitted)."""
    rule = f"configs/managers/rule_based/{version}.yml"
    out = subprocess.run(
        ["git", "-C", str(wt), "log", "--diff-filter=A", "--format=%ct", "--", rule],
        capture_output=True,
        text=True,
    )
    stamps = out.stdout.split()
    return int(stamps[-1]) if stamps else 0


def short(lineage: str, k: int) -> str:
    """A node's label: the lineage without its year and month, and its version."""
    name = re.sub(r"^pf-\d{6}(\d\d)-", r"\1-", lineage)
    return name if k == 1 else f"{name} v{k}"


def evolution_frame(found: list) -> pd.DataFrame:
    """Every validated version in the order the versions were made: its
    lineage, version number, final best, policy category and whether it beat
    its lineage's best so far, for the Summary's evolution chart."""
    entries = load_summary()["instances"]
    rows = []
    for wt in found:
        all_versions = versions(wt.name)
        lineage_best = None
        for k, v in enumerate(all_versions, 1):
            best = final_best(sweep_files(wt, v)["validate"])
            if not best:
                continue
            entry = entries.get(v) or {}
            rows.append(
                {
                    "lineage": wt.name,
                    "version": f"v{k}",
                    "rule_name": v,
                    "k": k,
                    "label": short(wt.name, k),
                    "made": made(wt, v),
                    "pool": best["pool"],
                    "se": best["pool_se"],
                    "policy category": entry.get("policy_category") or NOT_SUMMARIZED,
                    "idea": entry.get("idea", ""),
                    "versions": len(all_versions),
                    "improved": lineage_best is None or best["pool"] > lineage_best,
                }
            )
            lineage_best = max(lineage_best or best["pool"], best["pool"])
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(["made", "lineage", "k"]).reset_index(drop=True)
    frame["order"] = range(1, len(frame) + 1)
    frame["lo"], frame["hi"] = frame["pool"] - frame["se"], frame["pool"] + frame["se"]
    return frame


def root_reference(refs: dict):
    """The tree's root: the reference row marked `root` (ah against ah)."""
    return next((r for r in refs.get("rows", []) if r.get("root")), None)


def evolution_chart(frame: pd.DataFrame, refs: dict, links: bool = True):
    """The policies as a tree in the order they were made, after the Atlas's
    progress tree: the root is ah against ah; each v1 branches from it and
    each later version from the one before, in its lineage's colour when the
    lineage iterated (grey otherwise), dashed when the version did not beat
    its lineage's best; nodes in their policy category's colour; the best so
    far as a dotted step; the other references dashed across."""
    root = root_reference(refs)
    others = pd.DataFrame(
        [(r["name"], r["pool"]) for r in refs.get("rows", []) if not r.get("root")],
        columns=["name", "pool"],
    )
    by_version = {(r.lineage, r.k): r for r in frame.itertuples()}
    edges = []
    for r in frame.itertuples():
        parent = next(
            (by_version[(r.lineage, j)] for j in range(r.k - 1, 0, -1)
             if (r.lineage, j) in by_version),
            None,
        )
        if parent is not None:
            x1, y1 = parent.order, parent.pool
        elif root is not None:
            x1, y1 = 0, root["pool"]
        else:
            continue
        edges.append(
            {
                "x1": x1, "y1": y1, "x2": r.order, "y2": r.pool,
                "lineage": r.lineage, "iterated": r.versions > 1,
                "improved": bool(r.improved),
            }
        )
    edges = pd.DataFrame(
        edges, columns=["x1", "y1", "x2", "y2", "lineage", "iterated", "improved"]
    )
    # the best so far, from the root on
    start = [(0, root["pool"])] if root is not None else []
    best, steps = None, []
    for o, pool in start + list(zip(frame["order"], frame["pool"])):
        best = pool if best is None else max(best, pool)
        steps.append({"order": o, "best": best})
    steps = pd.DataFrame(steps)

    policies = list(
        dict.fromkeys(list(POLICY_COLOURS) + sorted(frame["policy category"].unique()))
    )
    spare = iter(POLICY_SPARE)
    colours = [
        POLICY_COLOURS.get(c) or (MUTED if c == NOT_SUMMARIZED else next(spare, MUTED))
        for c in policies
    ]
    lineages = sorted(frame.loc[frame["versions"] > 1, "lineage"].unique())
    values = list(frame["lo"]) + list(frame["hi"]) + list(others["pool"])
    values += [root["pool"]] if root is not None else []
    y_scale = alt.Scale(domain=[math.floor(min(values)) - 1, math.ceil(max(values)) + 1])
    n = int(frame["order"].max())
    # room right of the last node for the references' labels
    x_scale = alt.Scale(domain=[-0.6, n + 2.6], nice=False)
    x = alt.X("order:Q", title="experiment order", scale=x_scale,
              axis=alt.Axis(labels=False, ticks=False, grid=False))
    y = alt.Y("pool:Q", title="final best pool vs ah", scale=y_scale)
    tooltip = [
        alt.Tooltip("lineage:N"),
        alt.Tooltip("version:N"),
        alt.Tooltip("pool:Q", title="final best", format=".1f"),
        alt.Tooltip("se:Q", title="± se", format=".1f"),
        alt.Tooltip("policy category:N"),
        alt.Tooltip("idea:N"),
    ]
    edge_x = alt.X("x1:Q", scale=x_scale)
    edge_y = alt.Y("y1:Q", scale=y_scale)
    # tapping an edge keeps its lineage and dims the rest, within the chart;
    # tapping empty space shows all again. A node links to its version
    # (`links`); the export opens it from a click listener instead
    lineage_pick = alt.selection_point(name="lineage", fields=["lineage"], on="click")
    frame = frame.assign(href="?agent=" + frame["lineage"] + "&version=" + frame["rule_name"])

    def lit(full):
        return alt.condition(lineage_pick, alt.value(full), alt.value(DIMMED))

    dash = alt.StrokeDash(
        "improved:N",
        scale=alt.Scale(domain=[True, False], range=[[1, 0], [6, 4]]),
        legend=None,
    )
    layers = []
    if len(others):
        layers += [
            alt.Chart(others).mark_rule(strokeDash=[5, 4], color=MUTED, opacity=0.7)
            .encode(y=alt.Y("pool:Q", scale=y_scale)),
            alt.Chart(others).transform_calculate(order=f"{n + 2.5}")
            .mark_text(align="right", dy=-6, color=MUTED, fontSize=11)
            .encode(x=x, y=alt.Y("pool:Q", scale=y_scale), text="name:N"),
        ]
    layers += [
        alt.Chart(steps)
        .mark_line(interpolate="step-after", strokeDash=[2, 4], color=MUTED, strokeWidth=1.2)
        .encode(x=x, y=alt.Y("best:Q", scale=y_scale)),
        alt.Chart(frame)
        .mark_rule(color=MUTED, strokeWidth=4, strokeCap="round")
        .encode(x=x, y=alt.Y("lo:Q", scale=y_scale), y2="hi:Q", opacity=lit(0.18)),
        # each edge's wide, near-invisible twin catches the tap
        alt.Chart(edges)
        .mark_rule(strokeWidth=12, color=MUTED, opacity=0.01, cursor="pointer")
        .encode(x=edge_x, y=edge_y, x2="x2:Q", y2="y2:Q")
        .add_params(lineage_pick),
        alt.Chart(edges[~edges["iterated"]])
        .mark_rule(color=MUTED, strokeWidth=1)
        .encode(x=edge_x, y=edge_y, x2="x2:Q", y2="y2:Q", strokeDash=dash, opacity=lit(0.55)),
        alt.Chart(edges[edges["iterated"]])
        .mark_rule(strokeWidth=2.4)
        .encode(
            x=edge_x, y=edge_y, x2="x2:Q", y2="y2:Q", strokeDash=dash, opacity=lit(0.9),
            color=alt.Color(
                "lineage:N",
                scale=alt.Scale(domain=lineages, range=LINEAGE_COLOURS[: max(len(lineages), 1)]),
                legend=alt.Legend(title="lineage", orient="right"),
            ),
        ),
        alt.Chart(frame)
        .mark_circle(size=130, stroke="white", strokeWidth=1.2, cursor="pointer")
        .encode(
            x=x, y=y, tooltip=tooltip, opacity=lit(1.0),
            **({"href": "href:N"} if links else {}),
            color=alt.Color(
                "policy category:N",
                scale=alt.Scale(domain=policies, range=colours),
                legend=alt.Legend(title="policy category", orient="right"),
            ),
        ),
        alt.Chart(frame)
        .mark_text(dy=-13, fontSize=9.5, color=MUTED)
        .encode(x=x, y=y, text="label:N", opacity=lit(1.0)),
    ]
    if root is not None:
        root_row = pd.DataFrame(
            [{"order": 0, "pool": root["pool"], "name": "ah vs ah",
              "se": root.get("se", float("nan"))}]
        )
        layers += [
            alt.Chart(root_row)
            .mark_point(shape="diamond", size=190, filled=True, color=INK, opacity=1,
                        stroke="white", strokeWidth=1.2)
            .encode(x=x, y=y, tooltip=[alt.Tooltip("name:N"),
                                       alt.Tooltip("pool:Q", title="pool", format=".1f")]),
            alt.Chart(root_row)
            .mark_text(dy=-14, fontSize=10, color=INK)
            .encode(x=x, y=y, text="name:N"),
        ]
    return (
        alt.layer(*layers)
        .resolve_scale(color="independent", strokeDash="independent")
        .properties(height=360, padding={"top": 10, "right": 10})
        .configure_view(stroke=None)
        .properties(background="transparent")
        .configure_axis(
            labelColor=MUTED,
            titleColor=MUTED,
            gridColor=LINE,
            gridOpacity=0.12,
            domain=False,
            tickColor=LINE,
            labelFont="Geist Mono",
            titleFont="Geist",
            titleFontWeight=500,
        )
        .configure_legend(labelColor=MUTED, titleColor=MUTED, labelFont="Geist", titleFont="Geist")
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


def rule_counts(frame: pd.DataFrame) -> list:
    """(rule, policies including it, policies summarized, meaning) per rule."""
    summary = load_summary()
    ruled = frame.dropna(subset=["inputs"])
    ruled = ruled[ruled["policy category"] != NOT_SUMMARIZED]
    n = len(ruled)
    meaning = {c: spec["meaning"] for c, spec in summary["categories"].items()}
    return [
        (r, int(ruled["rules included"].apply(lambda rs: r in rs).sum()), n, meaning[r])
        for r in categories(summary, "rule")
    ]


EVOLUTION_CAPTION = (
    "Every validated version's final best against ah (± se), in the order the"
    " versions were made. The tree starts at ah against ah; each v1 branches"
    " from it and each later version from the one before, in its lineage's"
    " colour when the lineage iterated (grey otherwise), dashed when it did not"
    " beat its lineage's best. A node's colour is its policy category. Dotted:"
    " the best so far; dashed across: the references, under the same"
    " conditions. Hover a node for its idea, tap it to open that version; tap"
    " a branch to pick out its lineage, tap empty space to show all again."
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


def rules_html(frame: pd.DataFrame) -> str:
    rows = "".join(
        f'<div class="mech"><span>{esc(label)}</span>'
        f'<span class="mech-bar"><i style="width:{100 * k / max(n, 1):.0f}%"></i></span>'
        f'<span class="mech-n">{k}/{n}</span>'
        f'<span class="mech-def">{esc(meaning)}</span></div>'
        for label, k, n, meaning in rule_counts(frame)
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
        if pool == pool:  # validated
            width = 100 * (pool - lo) / (hi - lo)
            cell = (
                f'<div class="poolcell"><span class="pbar"><i style="width:{width:.0f}%">'
                f'</i></span><span class="pv">{pool:.1f}</span>'
                f'<span class="se">± {r["se"]:.1f}</span></div>'
            )
        else:
            cell = f'<span class="se">{esc(r.get("stage", "not swept"))}</span>'
        rules = r.get("rules included")
        tags = "".join(
            f'<span class="chip">{esc(x)}</span>'
            for x in (rules if isinstance(rules, list) else [])
        )
        category = r.get("policy category")
        category = category if isinstance(category, str) else ""
        dot = POLICY_COLOURS.get(category, MUTED)
        name = esc(r["name"])
        href = f"?agent={name}&version={esc(r['version'])}"
        idea = r.get("idea") if isinstance(r.get("idea"), str) else ""
        # every cell carries its own link over itself: a link stretched over
        # the row would size to the table (a row is no containing block), and
        # the last row's would cover every other
        hit = (
            f'<a class="hit" href="{href}" target="_self" tabindex="-1"'
            ' aria-hidden="true"></a>'
        )
        rows.append(
            f'<tr data-open="{name}" data-version="{esc(r["version"])}">'
            f'<td class="name">{hit}'
            f'<a class="open" href="{href}" target="_self">{name}</a>'
            + (f'<div class="idea">{esc(idea)}</div>' if idea else "")
            + "</td>"
            f'<td class="txt">{hit}{cell}</td><td>{hit}{num(r, "members", ".2f")}</td>'
            f'<td class="txt">{hit}{esc(r["version label"])}'
            f'<span class="se"> of {r["versions"]}</span></td>'
            f'<td class="txt">{hit}{esc(r["from"]) if isinstance(r.get("from"), str) else ""}</td>'
            f'<td>{hit}{num(r, "params", ".0f")}</td>'
            f'<td class="txt">{hit}<span class="pcat"><i style="background:{dot}"></i>'
            f"{esc(category)}</span></td>"
            f'<td class="txt tags">{hit}{tags}</td></tr>'
        )
    heads = ("agent", "final best", "members", "version", "from", "params",
             "policy category", "rules included")
    head = "".join(
        f'<th class="{"txt" if h in ("agent", "final best", "version", "from", "policy category", "rules included") else ""}">'
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
    frame = summary_frame(found, running_names())
    refs = references()
    shown = frame.head(SUMMARY_TOP)
    st.html(counts_html(found))  # the overview's counts, from the same files
    evolution = evolution_frame(found)
    if len(evolution):
        st.markdown('<div class="section">Evolution</div>', unsafe_allow_html=True)
        st.caption(EVOLUTION_CAPTION)
        # theme=None: the chart's own axis styling, not Streamlit's; a node
        # links to its version through the ?agent=&version= route of a table
        # row (Streamlit takes no selections from a layered chart)
        st.altair_chart(evolution_chart(evolution, refs), use_container_width=True, theme=None)
        st.html(reference_chips(refs))
    st.markdown('<div class="section">Agents</div>', unsafe_allow_html=True)
    st.caption(" ".join(filter(None, ["Tap a row to open the agent.", top_note(frame)])))
    st.html(agents_table_html(shown, refs))
    if "inputs" in frame:
        st.markdown('<div class="section">Rules included</div>', unsafe_allow_html=True)
        st.caption(RULES_CAPTION)
        st.html(rules_html(frame))  # every lineage
        st.markdown('<div class="section">What the rules read</div>', unsafe_allow_html=True)
        st.altair_chart(inputs_chart(shown), use_container_width=True, theme=None)


@st.fragment(run_every=REFRESH_S)
def board():
    """A lineage (an instance and its versions, #244), then one of its versions."""
    found = instances()
    st.caption(f"{len(found)} under {WT_ROOT} · refreshes every {REFRESH_S} s")
    if not found:
        empty("No instances yet. Start one with the policy-finder skill.")
        return
    running = running_names()
    lineage = {d.name: versions(d.name) for d in found}
    latest = {d.name: status(d, lineage[d.name][-1], running) for d in found}

    def label(n):
        count = len(lineage[n])
        return f"{n}   ·   {count} version{'s' * (count > 1)}   ·   {STATUS[latest[n]]}"

    pick = st.selectbox("Lineage", list(lineage), format_func=label, key="instance")
    wt, all_versions = WT_ROOT / pick, lineage[pick]
    version = all_versions[-1]
    if len(all_versions) > 1:
        version = st.segmented_control(
            "Version",
            all_versions,
            default=all_versions[-1],
            format_func=lambda v: version_label(v, all_versions),
            key=f"version-{pick}",
        ) or all_versions[-1]
    show_version(wt, version, status(wt, version, running))


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
    # a Summary row links here with ?agent=<name>&version=<version>: open it
    agent = st.query_params.get("agent")
    if agent:
        st.session_state["view"] = "Agent browser"
        st.session_state["instance"] = agent
        version = st.query_params.get("version")
        if version:
            st.session_state[f"version-{agent}"] = version
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
