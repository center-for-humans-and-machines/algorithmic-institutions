"""Export the observation app as one self-contained HTML page, to share.

Draws what dashboard.py shows (the overview, the summary and every instance's
notes, rule, scripts, report and sweep) from the same data and the same code,
as a static page: the sweep plots are embedded, the charts are Vega-Lite specs
drawn in the browser and the notes and reports are Markdown rendered there
(both libraries from a CDN, so the page needs a network connection to draw
them). Reads the instance worktrees and the files beside them only.

Run from the repo root (streamlit is not a project dependency):
    uv run --no-sync --with streamlit python \
        scripts/policy_finder/observation_app/export_html.py [--out <file>] [--fragment]

Writes ../policy-finder-worktrees/dashboard.html by default, beside the
instances and outside this checkout: it holds their results, which must not
reach policy-finder-base. --fragment leaves out the <!doctype>/<html> wrapper,
for a host that adds its own (a claude.ai artifact).
"""

import argparse
import html
import json
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import dashboard as d  # noqa: E402  (its page runs only under `streamlit run`)

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).parent


def esc(text) -> str:
    return html.escape(str(text))


def md(text: str) -> str:
    """Markdown the browser renders (marked, then DOMPurify)."""
    return f'<div class="md" data-md="{esc(text)}"></div>'


class Charts:
    """Vega-Lite specs, drawn by the page's script into their placeholders."""

    def __init__(self):
        self.specs = []

    def place(self, chart) -> str:
        spec = chart.to_dict()
        spec["width"] = "container"
        self.specs.append(spec)
        return f'<div class="chart" data-chart="{len(self.specs) - 1}"></div>'


def stats(tiles) -> str:
    cells = "".join(
        f'<div class="stat"><div class="k">{esc(k)}</div><div class="v">{v}</div></div>'
        for k, v in tiles
    )
    return f'<div class="stats">{cells}</div>'


def empty(text: str) -> str:
    return f'<div class="empty">{esc(text)}</div>'


def section(title: str) -> str:
    return f'<h3 class="section">{esc(title)}</h3>'


def caption(text: str) -> str:
    return f'<p class="caption">{esc(text)}</p>'


def pre(text: str, lang: str = "") -> str:
    cls = f' class="language-{lang}"' if lang else ""
    return f'<pre class="code"><code{cls}>{esc(text)}</code></pre>'


def details(summary: str, body: str) -> str:
    return f"<details><summary>{esc(summary)}</summary>{body}</details>"


# -- the summary ----------------------------------------------------------


def summary_view(found: list, keys: dict, charts: Charts) -> str:
    """The Summary: the best agents side by side, each row opening its agent;
    "What the rules do" counts every agent."""
    frame = d.summary_frame(found, keys)
    refs = d.references()
    shown = frame.head(d.SUMMARY_TOP)
    out = [stats(d.summary_stats(frame))]
    if shown["best pool"].notna().any():
        ref_rows = [(r["name"], r["pool"]) for r in refs.get("rows", [])]
        out += [
            section("Best pool against ah"),
            caption(" ".join(filter(None, [d.SUMMARY_CAPTION, d.top_note(frame)]))),
            charts.place(d.pool_chart(shown, ref_rows)),
            d.reference_chips(refs),
        ]
    out += [section("Agents"), caption("Tap a row to open the agent."), d.agents_table_html(shown, refs)]
    if "inputs" in frame:
        out += [
            section("What the rules do"),
            caption(d.MECHANISM_CAPTION),
            d.mechanism_html(frame),
            section("What the rules read"),
            charts.place(d.inputs_chart(shown)),
        ]
    return "".join(out)


# -- one instance ---------------------------------------------------------


def notes_tab(wt: Path, name: str, charts: Charts) -> str:
    text = d.read(wt / "notes/policy_finder" / f"{name}.md")
    if not text:
        return empty("No notes yet.")
    sections = d.parse_notes(text)
    trail = d.entries(sections.get("Explorations", ""))
    findings = d.items(sections.get("Key findings", ""))
    log = d.activity(wt, name)
    scripts = wt / "scripts/policy_finder" / name
    n_scripts = len([p for p in scripts.glob("*") if p.is_file()]) if scripts.is_dir() else 0
    out = [
        stats(
            [
                ("Exploration entries", len(trail)),
                ("Scripts", n_scripts),
                ("Key findings", len(findings)),
                ("Notes writes", int((log.kind == "notes").sum())),
            ]
        )
    ]
    if len(log):
        log = log.assign(time=log["time"].dt.strftime("%Y-%m-%dT%H:%M:%S"))
        out += [
            section("Pace"),
            caption(
                "Every write to the notes (clay), the analysis scripts and the rule"
                " (ink), from the agent's session log. Notes that keep pace sit"
                " between the scripts."
            ),
            charts.place(d.activity_chart(log)),
        ]
    out.append(section("Explorations"))
    if not trail:
        out.append(empty("No entries yet."))
    for i, (title, body) in enumerate(trail, 1):
        title_html = re.sub(r"`([^`]+)`", r"<code>\1</code>", esc(title))
        out.append(
            f'<div class="tl"><div class="tl-head"><span class="tl-num">{i}</span>'
            f"<span>{title_html}</span></div>{md(body) if body else ''}</div>"
        )
    if findings:
        out.append(section("Key findings"))
        out += [f'<div class="kf">{md(f"**{i}.** {t}")}</div>' for i, t in enumerate(findings, 1)]
    if sections.get("Hypothesis"):
        out += [section("Hypothesis"), f'<div class="hyp">{md(sections["Hypothesis"])}</div>']
    if sections.get("Justification"):
        out.append(details("Justification", md(sections["Justification"])))
    out.append(details("Raw notes", pre(text, "markdown")))
    return "".join(out)


def rule_tab(wt: Path, name: str) -> str:
    text = d.read(wt / "configs/managers/rule_based" / f"{name}.yml")
    if not text:
        return empty("No rule yet.")
    try:
        rule = yaml.safe_load(text)
    except yaml.YAMLError:
        return pre(text, "yaml")
    params, sweep = rule.get("params") or {}, rule.get("sweep_config") or {}
    cards = []
    for p, spec in params.items():
        spec = spec if isinstance(spec, dict) else {"definition": spec}
        cards.append(
            f'<div class="card"><div class="name">{esc(p)}'
            f'<span class="chip">{esc(spec.get("type", "?"))}</span></div>'
            f'<div class="def">{esc(spec.get("definition", ""))}</div>'
            f'<div class="range">{d.fmt_range(sweep.get(p, "no range"))}</div></div>'
        )
    out = [f'<div class="cards">{"".join(cards)}</div>']
    if rule.get("constraints"):
        chips = "".join(f'<span class="chip">{esc(c)}</span>' for c in rule["constraints"])
        out.append(f'<div class="chips"><span class="eyebrow">Constraints</span>{chips}</div>')
    out += [section("Code"), pre(rule.get("code", ""), "python"), details("Rule YAML", pre(text, "yaml"))]
    return "".join(out)


def scripts_tab(wt: Path, name: str) -> str:
    folder = wt / "scripts/policy_finder" / name
    files = sorted(p for p in folder.rglob("*") if p.is_file()) if folder.is_dir() else []
    if not files:
        return empty("No scripts yet.")
    language = {".py": "python", ".yml": "yaml", ".yaml": "yaml", ".md": "markdown"}
    options, bodies = [], []
    for i, path in enumerate(files):
        label = str(path.relative_to(folder))
        options.append(f'<option value="{i}">{esc(label)}</option>')
        text = d.read(path)
        body = empty(f"{label} is not a text file.") if text is None else pre(text, language.get(path.suffix, ""))
        bodies.append(f'<div class="script" data-script="{i}"{"" if i == 0 else " hidden"}>{body}</div>')
    return (
        f'<label class="pick"><span class="eyebrow">{len(files)} files</span>'
        f'<select class="script-pick">{"".join(options)}</select></label>{"".join(bodies)}'
    )


def report_tab(name: str) -> str:
    report = d.read(d.WT_ROOT / f"{name}.report.md")
    return f'<div class="doc">{md(report)}</div>' if report else empty("No report yet: still running, or stopped.")


def sweep_tab(wt: Path, name: str, charts: Charts, zero) -> str:
    folder = wt / d.SWEEP_DIR.format(name=name)
    path = folder / "sweep.json"
    if not path.exists():
        return empty("Not swept yet.")
    sweep = json.loads(path.read_text())
    points = pd.DataFrame(
        [{**p["params"], **{k: p[k] for k in ("name", "pool", "pool_se", "members")}} for p in sweep["points"]]
    ).sort_values("pool", ascending=False)
    best = points.iloc[0]
    near = int((points.pool >= best.pool - 2 * best.pool_se).sum())
    out = [
        stats(
            [
                ("Best pool vs ah", f"{best.pool:.1f}<small>± {best.pool_se:.1f}</small>"),
                ("Median point", f"{points.pool.median():.1f}"),
                ("Members, best point", f"{best.members:.2f}"),
                ("Within 2 se of best", f"{near}<small>/ {len(points)}</small>"),
            ]
        ),
        d.reference_html(best.pool)
        .replace('<div class="section">', '<h3 class="section">')
        .replace("Reference</div>", "Reference</h3>"),
    ]
    chips = "".join(f'<span class="chip">{esc(k)} <b>{v:g}</b></span>' for k, v in sweep["best"].items())
    out.append(f'<div class="chips"><span class="eyebrow">Best</span>{chips}</div>')
    plots = [(folder / f, c) for f, c in d.BEST_PLOTS.items() if (folder / f).exists()]
    if plots:
        out += [section("Best point"), d.figures_html(plots)]
    params = list(sweep["best"])
    out += [
        section("Landscape"),
        caption("Each dot is a design point. Ink: the best; clay: the top 10; dashed: never punishing against ah."),
        '<div class="land">' + "".join(charts.place(d.landscape(points, p, zero)) for p in params) + "</div>",
        section("Top 10"),
    ]
    out.append(d.top10_html(points, params))
    return "".join(out)


TABS = ("Notes", "Rule", "Scripts", "Report", "Sweep")


def instance(wt: Path, key: str, charts: Charts, zero, first: bool) -> str:
    name = wt.name
    config = json.loads(d.read(wt / ".claude/policy_finder.json") or "{}")
    tabs = "".join(
        f'<button class="tab" data-tab="{t}"{" aria-selected=true" if t == "Notes" else ""}>{t}</button>'
        for t in TABS
    )
    panels = [
        notes_tab(wt, name, charts),
        rule_tab(wt, name),
        scripts_tab(wt, name),
        report_tab(name),
        sweep_tab(wt, name, charts, zero),
    ]
    body = "".join(
        f'<section class="panel" data-panel="{t}"{"" if t == "Notes" else " hidden"}>{p}</section>'
        for t, p in zip(TABS, panels)
    )
    return (
        f'<article class="instance" data-instance="{esc(name)}"{"" if first else " hidden"}>'
        f'<div class="inst"><h2>{esc(name)}</h2><span class="pill {key}"><i></i>{d.STATUS[key]}</span></div>'
        f'<div class="meta">params {config.get("min_params", "?")}–{config.get("max_params", "?")}'
        f" · policy-finder/{esc(name)}</div>"
        f'<nav class="tabs">{tabs}</nav>{body}</article>'
    )


def build() -> str:
    found = d.instances()
    running = d.running_names()
    keys = {w.name: d.status(w.name, running) for w in found}
    charts, zero = Charts(), d.zero_pool()
    summary = summary_view(found, keys, charts)
    options = "".join(
        f'<option value="{esc(w.name)}">{esc(w.name)}  ·  {d.STATUS[keys[w.name]]}</option>' for w in found
    )
    articles = "".join(instance(w, keys[w.name], charts, zero, i == 0) for i, w in enumerate(found))
    style = STYLE + (HERE / "overview.css").read_text() + (HERE / "summary.css").read_text()
    return (
        PAGE.replace("__STYLE__", style)
        .replace("__OVERVIEW__", d.overview(found))
        .replace("__SUMMARY__", summary)
        .replace("__COUNT__", str(len(found)))
        .replace("__STAMP__", datetime.now().strftime("%Y-%m-%d %H:%M"))
        .replace("__DATA_NOTE__", esc(d.DATA_NOTE))
        .replace("__OPTIONS__", options)
        .replace("__ARTICLES__", articles or empty("No instances."))
        .replace("__SPECS__", json.dumps(charts.specs).replace("</", "<\\/"))
        .replace("__INK__", d.INK)
        .replace("__LINE__", d.LINE)
        .replace("__MUTED__", d.MUTED)
    )


STYLE = """
/* Layout: one column, the app's pickers, pill tabs and cards, in the reports' palette */
:root {
  --ground: #faf9f5; --surface: #ffffff; --sunk: #f0eee6;
  --ink: #141413; --ink-2: #3d3d3a; --muted: #87867f;
  --line: #e3dacc; --line-soft: #ece8de;
  --clay: #d97757; --clay-soft: #f3d9cd;
  --green: #2f6b3f; --green-soft: #e3eedf; --blue: #4f7aa8; --blue-soft: #e2eaf3;
  --code-kw: #a2482c; --code-str: #2f6b3f; --code-com: #87867f; --code-num: #4f7aa8;
  --serif: "Newsreader", "Iowan Old Style", Georgia, serif;
  --sans: "Geist", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "Geist Mono", ui-monospace, "SF Mono", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --ground: #1a1918; --surface: #232220; --sunk: #2b2a27;
  --ink: #f0eee6; --ink-2: #cfccc2; --muted: #9a988f;
  --line: #3a3833; --line-soft: #2f2e2a;
  --clay: #e08b6d; --clay-soft: #4a3026;
  --green: #7fbf8e; --green-soft: #26302a; --blue: #8fb3dc; --blue-soft: #262d36;
  --code-kw: #e08b6d; --code-str: #7fbf8e; --code-com: #9a988f; --code-num: #8fb3dc;
  color-scheme: dark; } }
:root[data-theme="dark"] {
  --ground: #1a1918; --surface: #232220; --sunk: #2b2a27;
  --ink: #f0eee6; --ink-2: #cfccc2; --muted: #9a988f;
  --line: #3a3833; --line-soft: #2f2e2a;
  --clay: #e08b6d; --clay-soft: #4a3026;
  --green: #7fbf8e; --green-soft: #26302a; --blue: #8fb3dc; --blue-soft: #262d36;
  --code-kw: #e08b6d; --code-str: #7fbf8e; --code-com: #9a988f; --code-num: #8fb3dc;
  color-scheme: dark; }
body { background: var(--ground); color: var(--ink-2); font-family: var(--sans);
  font-size: 15px; margin: 0; }
main { max-width: 1120px; margin: 0 auto; padding-inline: 20px; padding-block: 48px 64px; }
h1, h2, h3 { font-family: var(--serif); font-weight: 400; letter-spacing: -0.015em;
  color: var(--ink); text-wrap: balance; }
code { font-family: var(--mono); background: var(--sunk); color: var(--ink);
  border-radius: 5px; padding: 1px 6px; font-size: .86em; }
pre.code { background: var(--sunk); border: 1px solid var(--line-soft); border-radius: 12px;
  padding: 14px 16px; overflow-x: auto; font-size: 13px; line-height: 1.55; margin: 0 0 12px; }
pre.code code { background: none; padding: 0; font-size: inherit; color: var(--ink-2); }
.hljs-keyword, .hljs-built_in, .hljs-literal { color: var(--code-kw); }
.hljs-string { color: var(--code-str); }
.hljs-comment { color: var(--code-com); font-style: italic; }
.hljs-number { color: var(--code-num); }
.hljs-attr, .hljs-title, .hljs-section { color: var(--ink); font-weight: 500; }
.eyebrow { font-size: 11.5px; letter-spacing: .09em; text-transform: uppercase;
  color: var(--muted); font-weight: 500; }
.hero { display: grid; gap: 6px; margin-bottom: 22px; }
.hero h1 { font-size: clamp(34px, 4.6vw, 46px); line-height: 1.08; margin: 0; }
.hero p { color: var(--muted); font-size: 13.5px; margin: 0; }
.pick { display: grid; gap: 6px; margin-bottom: 18px; max-width: 520px; }
select { font: 14px var(--sans); color: var(--ink); background: var(--surface);
  border: 1px solid var(--line); border-radius: 12px; padding: 9px 12px; width: 100%; }
select:focus-visible, .tab:focus-visible, summary:focus-visible { outline: 2px solid var(--clay);
  outline-offset: 2px; }
.inst { display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px 16px; margin: 8px 0 4px; }
.inst h2 { font-size: 30px; margin: 0; }
.meta { font-family: var(--mono); font-size: 12.5px; color: var(--muted); margin-bottom: 18px;
  overflow-wrap: anywhere; }
.pill { font-size: 12.5px; font-weight: 500; padding: 3px 11px; border-radius: 999px;
  display: inline-flex; align-items: center; gap: 7px; }
.pill i { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.pill.running { background: var(--clay-soft); color: var(--clay); }
.pill.checked { background: var(--green-soft); color: var(--green); }
.pill.done { background: var(--blue-soft); color: var(--blue); }
.pill.stopped { background: var(--sunk); color: var(--muted); }
.tabs { display: flex; flex-wrap: wrap; gap: 4px; background: var(--sunk); padding: 4px;
  border-radius: 999px; width: fit-content; max-width: 100%; margin-bottom: 22px; }
.tab { font: 13.5px var(--sans); color: var(--ink-2); background: none; border: 0;
  padding: 6px 18px; border-radius: 999px; cursor: pointer; }
.tab[aria-selected="true"] { background: var(--surface); color: var(--ink); font-weight: 500;
  box-shadow: 0 1px 2px rgba(0,0,0,.08); }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 12px; margin-bottom: 16px; }
.card { background: var(--surface); border: 1px solid var(--line-soft); border-radius: 16px;
  padding: 16px 18px; display: grid; gap: 8px; align-content: start; min-width: 0; }
.card .name { font-family: var(--mono); font-size: 15px; color: var(--ink); font-weight: 500; }
.card .def { font-size: 14px; line-height: 1.5; }
.card .range { font-family: var(--mono); font-size: 12.5px; color: var(--clay); }
.chip { font-family: var(--mono); font-size: 11.5px; padding: 1px 8px; border-radius: 999px;
  background: var(--sunk); color: var(--ink-2); margin-left: 8px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 4px 0 16px; }
.chips .chip { margin: 0; font-size: 12.5px; padding: 3px 10px; }
.chips .chip b { color: var(--ink); font-weight: 500; }
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px;
  background: var(--line-soft); border: 1px solid var(--line-soft); border-radius: 16px;
  overflow: hidden; margin-bottom: 18px; }
@media (max-width: 760px) { .stats { grid-template-columns: repeat(2, 1fr); } }
.stat { background: var(--surface); padding: 16px 18px; display: grid; gap: 4px; }
.stat .v { font-family: var(--serif); font-size: 30px; line-height: 1.1; color: var(--ink);
  font-variant-numeric: tabular-nums; }
.stat .v small { font-family: var(--mono); font-size: 13px; color: var(--muted); margin-left: 4px;
  white-space: nowrap; }
.stat .k { font-size: 12.5px; color: var(--muted); }
.section { font-size: 21px; margin: 26px 0 10px; }
.caption { font-size: 13px; color: var(--muted); margin: 0 0 10px; max-width: 80ch; }
.empty { color: var(--muted); font-size: 14px; padding: 28px 0; }
/* vega-embed makes its container inline-block; keep it full width */
.chart { display: block !important; box-sizing: border-box; width: 100%;
  background: var(--surface); border: 1px solid var(--line-soft); border-radius: 14px;
  padding: 8px; min-width: 0; }
/* one panel per param; a row of more than fit scrolls sideways */
.land { display: flex; gap: 10px; overflow-x: auto; padding-bottom: 4px; }
.land .chart { flex: 1 0 240px; width: auto; }
.md { line-height: 1.65; }
.md p, .md li { font-size: 14.5px; }
.md table { border-collapse: collapse; margin: 8px 0; font-size: 13.5px; }
.md th, .md td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line-soft); }
.md pre { background: var(--sunk); border-radius: 10px; padding: 10px 14px; overflow-x: auto; }
.doc { max-width: 76ch; }
.doc .md p, .doc .md li { font-size: 15px; line-height: 1.68; }
.tl { position: relative; padding: 0 0 10px 46px; max-width: 80ch; }
.tl::before { content: ""; position: absolute; left: 13px; top: 30px; bottom: -4px;
  width: 1.5px; background: var(--line); }
.tl:last-of-type::before { display: none; }
.tl-head { font-family: var(--serif); font-size: 18.5px; color: var(--ink); line-height: 1.3;
  margin-left: -46px; display: flex; gap: 14px; align-items: baseline; }
.tl-head code { font-size: 12.5px; word-break: break-all; }
.tl-num { flex: none; width: 28px; height: 28px; border-radius: 50%; background: var(--clay-soft);
  color: var(--clay); font-family: var(--mono); font-size: 13px; display: inline-grid;
  place-items: center; transform: translateY(-2px); }
.kf { background: var(--surface); border: 1px solid var(--line-soft); border-radius: 14px;
  padding: 4px 16px; max-width: 80ch; margin-bottom: 8px; }
.hyp { background: var(--clay-soft); border-radius: 16px; padding: 6px 22px; max-width: 80ch; }
.hyp .md p, .hyp .md li { color: var(--ink); }
details { background: var(--surface); border: 1px solid var(--line-soft); border-radius: 12px;
  padding: 10px 14px; margin: 12px 0; }
summary { cursor: pointer; color: var(--ink); font-weight: 500; }
details[open] summary { margin-bottom: 10px; }
footer { margin-top: 40px; font-size: 12.5px; color: var(--muted); }
"""

PAGE = """<title>Policy Finder Instances</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500&family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&display=swap">
<style>__STYLE__</style>
<main>
  <header class="hero">
    <div class="eyebrow">Algorithmic institutions</div>
    <h1>Policy finder</h1>
    <p>Agents that design rule-based punishment managers, swept against the AH manager ·
    snapshot of __STAMP__</p>
    <p class="data-note">__DATA_NOTE__</p>
  </header>
  <nav class="tabs" id="views">
    <button class="tab view" data-view="overview" aria-selected="true">Overview</button>
    <button class="tab view" data-view="summary">Summary</button>
    <button class="tab view" data-view="browser">Agent browser</button>
  </nav>
  <section class="view-panel" data-view-panel="overview">__OVERVIEW__</section>
  <section class="view-panel" data-view-panel="summary" hidden>__SUMMARY__</section>
  <section class="view-panel" data-view-panel="browser" hidden>
    <label class="pick"><span class="eyebrow">Instance · __COUNT__</span>
      <select id="instance">__OPTIONS__</select></label>
    __ARTICLES__
  </section>
  <footer>Exported from the policy-finder observation app.</footer>
</main>
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/dompurify/3.1.6/purify.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/vega@5.30.0/build/vega.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5.20.1/build/vega-lite.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6.26.0/build/vega-embed.min.js"></script>
<script id="specs" type="application/json">__SPECS__</script>
<script>
(function () {
  const specs = JSON.parse(document.getElementById("specs").textContent);
  const drawn = new Set();
  // the app's charts carry its light colours; draw them in the page's theme
  const LIGHT = { "__INK__": "--ink", "__LINE__": "--line", "__MUTED__": "--muted" };
  function themed(spec) {
    const css = getComputedStyle(document.documentElement);
    let text = JSON.stringify(spec);
    Object.keys(LIGHT).forEach(function (hex) {
      text = text.split(hex).join(css.getPropertyValue(LIGHT[hex]).trim() || hex);
    });
    const out = JSON.parse(text);
    // the charts' configure(background=...) drops their axis config; restore it
    const muted = css.getPropertyValue("--muted").trim();
    const line = css.getPropertyValue("--line").trim();
    out.background = "transparent";
    out.config = Object.assign({}, out.config, {
      background: "transparent",
      view: Object.assign({}, (out.config || {}).view, { stroke: null }),
      axis: Object.assign({}, (out.config || {}).axis, { labelColor: muted, titleColor: muted,
        gridColor: line, tickColor: line, domain: false, labelFont: "Geist Mono",
        titleFont: "Geist", titleFontWeight: 500 }),
    });
    return out;
  }
  function render(scope) {
    if (!scope) return;
    scope.querySelectorAll(".md[data-md]").forEach(function (el) {
      if (el.dataset.done) return;
      const raw = el.dataset.md;
      el.innerHTML = window.marked && window.DOMPurify ? DOMPurify.sanitize(marked.parse(raw)) : "";
      if (!el.innerHTML) el.textContent = raw;
      el.dataset.done = "1";
    });
    if (window.hljs) scope.querySelectorAll("pre.code code[class]").forEach(function (el) {
      if (!el.dataset.highlighted) hljs.highlightElement(el);
    });
    if (window.vegaEmbed) scope.querySelectorAll("[data-chart]").forEach(function (el) {
      const i = el.dataset.chart;
      if (drawn.has(i) || el.offsetParent === null) return;
      drawn.add(i);
      vegaEmbed(el, themed(specs[i]), { actions: false, renderer: "svg" }).catch(function () {});
    });
  }
  function current() {
    return document.querySelector(".instance:not([hidden])");
  }
  function show(name) {
    document.querySelectorAll(".instance").forEach(function (a) {
      a.hidden = a.dataset.instance !== name;
    });
    render(current());
  }
  function view(name) {
    document.querySelectorAll(".tab.view").forEach(function (t) {
      t.setAttribute("aria-selected", t.dataset.view === name ? "true" : "false");
    });
    document.querySelectorAll(".view-panel").forEach(function (p) {
      p.hidden = p.dataset.viewPanel !== name;
    });
    render(document.querySelector(".view-panel:not([hidden])"));
    render(current());
  }
  const pick = document.getElementById("instance");
  pick.addEventListener("change", function () { show(pick.value); });
  document.addEventListener("click", function (e) {
    // a Summary row opens its agent in the Agent browser
    const row = e.target.closest("tr[data-open]");
    if (row) {
      e.preventDefault();
      view("browser");
      pick.value = row.dataset.open;
      show(row.dataset.open);
      window.scrollTo(0, 0);
      return;
    }
    const tab = e.target.closest(".tab");
    if (!tab) return;
    if (tab.classList.contains("view")) { view(tab.dataset.view); return; }
    const art = tab.closest(".instance");
    art.querySelectorAll(".tab").forEach(function (t) {
      t.setAttribute("aria-selected", t === tab ? "true" : "false");
    });
    art.querySelectorAll(".panel").forEach(function (p) {
      p.hidden = p.dataset.panel !== tab.dataset.tab;
    });
    render(art);
  });
  document.addEventListener("change", function (e) {
    if (!e.target.classList.contains("script-pick")) return;
    const panel = e.target.closest(".panel");
    panel.querySelectorAll(".script").forEach(function (s) {
      s.hidden = s.dataset.script !== e.target.value;
    });
    render(panel);
  });
  const ready = function () { render(document.querySelector(".view-panel:not([hidden])")); };
  if (document.readyState === "complete") ready();
  else window.addEventListener("load", ready);
})();
</script>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(d.WT_ROOT / "dashboard.html"))
    ap.add_argument("--fragment", action="store_true", help="no <html> wrapper")
    args = ap.parse_args()
    page = build()
    if not args.fragment:
        page = (
            '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"</head><body>\n{page}\n</body></html>\n"
        )
    Path(args.out).write_text(page)
    print(f"Wrote {args.out} ({len(page) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
