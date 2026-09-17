"""Build the interactive campaign report (summary doc section 5).

One self-contained report.html with the progress tree as the welcome
layer and the score views as toggleable layers; the machinery and
leaderboard pages (built by their own scripts) are linked from the same
nav. Tree nodes and score points are hoverable (PR number, title, mean)
and clickable: PRs with a machinery story page open it, everything else
opens the PR on GitHub. All data comes from the frozen corpus files;
the PNG scripts remain as static previews of these layers.

Usage:
    python scripts/data_analysis/autoresearch_summary/build_report.py
"""

import html
import json
import math
import re
from pathlib import Path

from progress_tree import (
    CATEGORY_STYLE, FAIL_GREY, ROOT_MEAN, lineage_root, resolve_means,
)
from score_progressions import (
    DATA, FAMILY, FAMILY_COLOR, METRICS, ROOT_RUN, ROOT_SCORES_CSV,
    TREE_COLOR, parse_scores, spines,
)

OUT = Path("plots/data_analysis/autoresearch_summary/report.html")
METRIC_DEFS = Path("notes/evaluation_metric_defs.md")
REPO = "center-for-humans-and-machines/algorithmic-institutions"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#eceae6"


def esc(s):
    return html.escape(str(s), quote=True)


def clean_title(e):
    return e["title"].replace("[SUCCESS]", "").replace("[FAIL]", "").strip()


class LogScale:
    def __init__(self, lo, hi, px0, px1):
        self.a, self.b = math.log(lo), math.log(hi)
        self.px0, self.px1 = px0, px1

    def __call__(self, v):
        t = (math.log(v) - self.a) / (self.b - self.a)
        return self.px1 + (self.px0 - self.px1) * t  # hi -> px0 (top)


def marker(shape, x, y, color, size=7.5):
    """An SVG marker centred on (x, y); shapes mirror the PNG's."""
    s = size
    if shape == "o":
        return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{s}" fill="{color}" '
                f'stroke="#fcfcfb" stroke-width="1.5"/>')
    if shape == "s":
        return (f'<rect x="{x - s:.1f}" y="{y - s:.1f}" width="{2 * s}" '
                f'height="{2 * s}" fill="{color}" stroke="#fcfcfb" '
                f'stroke-width="1.5"/>')
    pts = {
        "^": [(x, y - s * 1.2), (x - s * 1.1, y + s), (x + s * 1.1, y + s)],
        "v": [(x, y + s * 1.2), (x - s * 1.1, y - s), (x + s * 1.1, y - s)],
        "D": [(x, y - s * 1.25), (x + s * 1.05, y), (x, y + s * 1.25),
              (x - s * 1.05, y)],
    }[shape]
    path = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
    return (f'<polygon points="{path}" fill="{color}" stroke="#fcfcfb" '
            f'stroke-width="1.5"/>')


def metric_defs():
    """{row: (name, one-line gloss)}, parsed from the definitions note.

    Each row is defined there as `**CA -- <name>:** <body>`; the gloss is
    the body's first sentence (the split is on period+space, so the bin
    edges' decimals survive). Never hand-copied — the note is the source
    of truth for what a row measures.
    """
    text = METRIC_DEFS.read_text()
    defs = {}
    for m in re.finditer(r"^\*\*([A-Z]{2,4}) -- ([^:]+):\*\* *(.+?)(?=\n\n|\Z)",
                         text, re.M | re.S):
        gloss = re.split(r"\.\s", m.group(3).replace("\n", " "))[0]
        defs[m.group(1)] = (m.group(2).strip(), gloss.strip())
    missing = [m for m in METRICS if m not in defs]
    if missing:
        raise SystemExit(f"{METRIC_DEFS}: no definition for {missing}")
    return defs


def metric_legend(rail=False):
    """Static what-the-rows-mean legend, by slot; `rail` = narrow column."""
    defs = metric_defs()
    cols = []
    for fam, color in FAMILY_COLOR.items():
        rows = [m for m in METRICS if FAMILY[m] == fam]
        items = "".join(
            f'<div class="mrow"><b>{m}</b><span><span class="n">'
            f'{esc(defs[m][0])}</span>'
            + ("" if rail else f'<span class="g">{esc(defs[m][1])}</span>')
            + '</span></div>' for m in rows)
        cols.append(f'<div class="mcol"><h4 style="color:{color}">{fam}'
                    f' &middot; {len(rows)} rows</h4>{items}</div>')
    return (f'<div class="mlegend{" rail" if rail else ""}">'
            f'<div class="mlhead">What the 21 score rows measure '
            f'<small>(verbatim from <code>{METRIC_DEFS}</code>)</small></div>'
            f'{"".join(cols)}</div>')


def node_link(pr, pr_slug):
    if pr in pr_slug:
        return f'href="machinery_pages/{pr_slug[pr]}.html"'
    return f'href="https://github.com/{REPO}/pull/{pr}" target="_blank"'


# --------------------------------------------------------------------- #
# layer 1: the progress tree
# --------------------------------------------------------------------- #
def tree_svg(experiments, categories, pr_slug):
    experiments = sorted(experiments, key=lambda e: e["pr"])
    means = resolve_means(experiments)
    roots = lineage_root(experiments)
    xs = {e["pr"]: i + 1 for i, e in enumerate(experiments)}
    gnn = {pr for pr, r in roots.items() if r in (160, 161)}
    gmlp = {pr for pr, r in roots.items() if r == 167}

    W, H = 1150, 560
    L, R, T, B = 56, 24, 30, 46
    ys = LogScale(min(means.values()) * 0.96,
                  max(max(means.values()), ROOT_MEAN) * 1.05, T, H - B)
    xstep = (W - L - R) / (len(experiments) + 1)

    def xp(order):
        return L + order * xstep

    s = []
    # y grid + ticks
    for tick in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0):
        if not (ys.b >= math.log(tick) >= ys.a):
            continue
        y = ys(tick)
        s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" '
                 f'stroke="{GRID}"/>'
                 f'<text x="{L - 8}" y="{y + 3:.1f}" text-anchor="end" '
                 f'font-size="10" fill="{INK_2}">{tick}</text>')
    s.append(f'<text x="14" y="{(T + H - B) / 2:.0f}" font-size="11" '
             f'fill="{INK_2}" transform="rotate(-90 14 {(T + H - B) / 2:.0f})"'
             f' text-anchor="middle">stack mean score (lower is better)'
             f'</text>')
    s.append(f'<text x="{(L + W - R) / 2}" y="{H - 10}" font-size="11" '
             f'fill="{INK_2}" text-anchor="middle">experiment order</text>')

    def tags(e):
        tree = ("gnn" if e["pr"] in gnn else
                "gmlp" if e["pr"] in gmlp else "none")
        verdict = "success" if e["verdict"] == "SUCCESS" else "fail"
        return f't-{tree} v-{verdict}'

    by_pr = {e["pr"]: e for e in experiments}

    def on_spine(pr):
        """Success whose whole ancestry back to main is successes — only
        those edges read as the adopted spine (e.g. #164, a success on
        the failed #161 branch, stays a dashed side-branch)."""
        while pr:
            if by_pr[pr]["verdict"] != "SUCCESS":
                return False
            pr = by_pr[pr]["parent_pr"]
        return True

    # edges
    for e in experiments:
        pr = e["pr"]
        px = xp(xs[e["parent_pr"]]) if e["parent_pr"] else xp(0)
        py = ys(means[e["parent_pr"]] if e["parent_pr"] else ROOT_MEAN)
        x, y = xp(xs[pr]), ys(means[pr])
        in_tree = pr in gnn or pr in gmlp
        tree = TREE_COLOR["gnn"] if pr in gnn else TREE_COLOR["gmlp"]
        if in_tree and on_spine(pr):
            style = f'stroke="{tree}" stroke-width="2.4"'
        elif in_tree:
            style = (f'stroke="{tree}" stroke-width="1.1" '
                     f'stroke-dasharray="6 4" stroke-opacity="0.55"')
        elif e["verdict"] == "SUCCESS":
            style = f'stroke="{FAIL_GREY}" stroke-width="1.0"'
        else:
            style = (f'stroke="#d6d5d0" stroke-width="0.9" '
                     f'stroke-dasharray="2 4"')
        s.append(f'<line class="{tags(e)}" x1="{px:.1f}" y1="{py:.1f}" '
                 f'x2="{x:.1f}" y2="{y:.1f}" {style}/>')

    # frontier step (successes on the two trees only)
    fx, fy, best = [xp(0)], [ys(ROOT_MEAN)], ROOT_MEAN
    for e in experiments:
        m = e["metrics"].get("mean")
        if (e["pr"] in gnn or e["pr"] in gmlp) and \
                on_spine(e["pr"]) and m is not None and m < best:
            best = m
        fx.append(xp(xs[e["pr"]]))
        fy.append(ys(best))
    d = f"M {fx[0]:.1f} {fy[0]:.1f}"
    for i in range(1, len(fx)):
        d += f" H {fx[i]:.1f} V {fy[i]:.1f}"
    s.append(f'<path class="t-none v-success" d="{d}" fill="none" '
             f'stroke="#8f8e89" stroke-dasharray="2 4"/>')

    # root
    s.append(marker("D", xp(0), ys(ROOT_MEAN), INK, 6.5))
    s.append(f'<text x="{xp(0) - 4:.1f}" y="{ys(ROOT_MEAN) - 12:.1f}" '
             f'font-size="10" fill="{INK}" text-anchor="end">main</text>')

    # nodes (clickable, hoverable)
    for e in experiments:
        pr, x, y = e["pr"], xp(xs[e["pr"]]), ys(means[e["pr"]])
        no_eval = e["metrics"].get("mean") is None
        cat = categories.get(pr, "")
        if e["verdict"] == "SUCCESS":
            color, shape = CATEGORY_STYLE.get(cat, (INK_2, "o"))
            m = marker(shape, x, y, color)
            dy = -13
        else:
            fill = "#fcfcfb" if no_eval else FAIL_GREY
            m = (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{fill}" '
                 f'stroke="{FAIL_GREY}" stroke-width="1.4"/>')
            dy = 16
        mean_txt = ("no evaluation" if no_eval
                    else f"stack mean {means[pr]:.3f}")
        tip = (f"#{pr} {e['verdict']} - {clean_title(e)[:110]} "
               f"| {mean_txt}" + (f" | {cat}" if cat else ""))
        s.append(
            f'<a {node_link(pr, pr_slug)} class="node {tags(e)}" '
            f'data-tip="{esc(tip)}">{m}'
            f'<text x="{x:.1f}" y="{y + dy:.1f}" text-anchor="middle" '
            f'font-size="9" fill="{INK_2}">{pr}</text></a>'
        )
    return (f'<svg viewBox="0 0 {W} {H}" font-family="system-ui, sans-serif">'
            + "".join(s) + "</svg>")


def slot_row(slot, text):
    return (f'<div class="srow"><span class="schip" style="background:'
            f'{FAMILY_COLOR[slot]}">{slot}</span><span>{text}</span></div>')


STACK_INTRO = f"""
<div class="stacks">
<div class="scard">
<h3 style="color:{TREE_COLOR['gnn']}">the gnn stack <small>(tip
#179)</small></h3>
{slot_row("contribution", "a graph neural network: members exchange "
          "messages each round, each keeps a small recurrent memory, and "
          "a per-group virtual node carries persistent group state; draws "
          "are coupled by a herding copula")}
{slot_row("switch", "a graph-network switch predictor; on decision rounds "
          "a joint head draws how many leave each group, then a "
          "conditional-Bernoulli step picks who")}
{slot_row("punisher", "a multinomial logistic regression (a linear "
          "model over 31 punishment levels), its group draws coupled by "
          "the severity copula")}
</div>
<div class="scard">
<h3 style="color:{TREE_COLOR['gmlp']}">the gaussian-MLP stack <small>(tip
#177)</small></h3>
{slot_row("contribution", "a tiny 2-layer neural network predicting a "
          "bell curve (centre and spread) per member, with extra "
          "probability spikes at repeat / 0 / 20 and a group copula on "
          "the draws")}
{slot_row("switch", "the same graph-network switch predictor and joint "
          "exodus head, with the group sizes one-hot encoded")}
{slot_row("punisher", "the same severity-copula multinomial logistic "
          "regression -- the one part both stacks share unchanged")}
</div>
</div>
"""

TREE_LEGEND = f"""
<p class="legend">
{"".join(f'<span style="color:{c}">&#9632;</span> {cat} &nbsp; '
         for cat, (c, _) in CATEGORY_STYLE.items())}
<span style="color:{FAIL_GREY}">&#9679;</span> failed attempt &nbsp;
<span style="color:{TREE_COLOR['gnn']}">&#9473;</span> gnn tree &nbsp;
<span style="color:{TREE_COLOR['gmlp']}">&#9473;</span> gaussian-MLP tree
&nbsp; (solid = success spine, dashed = failed branch, dotted step = best
mean so far) &mdash; hover a node for the PR, click it for its story.
</p>
"""


# --------------------------------------------------------------------- #
# layers 2 + 3: spine scores
# --------------------------------------------------------------------- #
def load_spines():
    experiments = json.loads((DATA / "experiments.json").read_text())
    trees = spines(experiments)
    cache = json.loads((DATA / "spine_scores.json").read_text())
    root = parse_scores(ROOT_SCORES_CSV.read_text())[ROOT_RUN]
    return experiments, trees, cache, root


def panel_svg(metric, trees, cache, root, pr_slug, W=210, H=150):
    L, R, T, B = 30, 8, 22, 20
    ys = LogScale(0.35, 13.0, T, H - B)
    nmax = max(len(c) for c in trees.values())
    s = [f'<text x="{(L + W - R) / 2}" y="13" text-anchor="middle" '
         f'font-size="11" font-weight="600" '
         f'fill="{FAMILY_COLOR[FAMILY[metric]]}">{metric}</text>']
    for band in (1.0, 2.0, 5.0):
        y = ys(band)
        s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" '
                 f'stroke="{GRID}"/>'
                 f'<text x="{L - 4}" y="{y + 3:.1f}" text-anchor="end" '
                 f'font-size="8" fill="{INK_2}">{band:.0f}</text>')
    for tree, chain in trees.items():
        vals = [root[metric]] + [cache[str(pr)][metric] for pr in chain]
        pts = [(L + i * (W - L - R) / nmax, ys(v))
               for i, v in enumerate(vals)]
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        s.append(f'<polyline points="{d}" fill="none" '
                 f'stroke="{TREE_COLOR[tree]}" stroke-width="2"/>')
        for i, (x, y) in enumerate(pts):
            label = "main" if i == 0 else f"#{chain[i - 1]}"
            tip = f"{label} - {metric} {vals[i]:.3f} ({tree} spine)"
            dot = (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" '
                   f'fill="{TREE_COLOR[tree]}" stroke="#fcfcfb" '
                   f'stroke-width="1"/>')
            s.append(
                f'<a {node_link(chain[i - 1], pr_slug)} class="node" '
                f'data-tip="{esc(tip)}">{dot}</a>' if i else
                f'<g data-tip="{esc(tip)}">{dot}</g>')
    return (f'<svg viewBox="0 0 {W} {H}" '
            f'font-family="system-ui, sans-serif">{"".join(s)}</svg>')


def breakdown_svg(tree, chain, cache, root, pr_slug, W=560, H=400):
    L, R, T, B = 40, 60, 34, 26
    ys = LogScale(0.35, 13.0, T, H - B)
    n = len(chain)
    s = [f'<text x="{(L + W - R) / 2}" y="16" text-anchor="middle" '
         f'font-size="12.5" font-weight="600" fill="{INK}">'
         f'{tree} spine: main &#8594; #{chain[-1]}</text>']
    for band in (1.0, 2.0, 5.0):
        y = ys(band)
        s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" '
                 f'stroke="{GRID}"/>'
                 f'<text x="{L - 5}" y="{y + 3:.1f}" text-anchor="end" '
                 f'font-size="9" fill="{INK_2}">{band:.0f}</text>')

    def px(i):
        return L + i * (W - L - R) / n

    for i in range(n + 1):
        label = "main" if i == 0 else f"#{chain[i - 1]}"
        s.append(f'<text x="{px(i):.1f}" y="{H - 8}" text-anchor="middle" '
                 f'font-size="9" fill="{INK_2}">{label}</text>')
    for metric in METRICS:
        vals = [root[metric]] + [cache[str(pr)][metric] for pr in chain]
        pts = " ".join(f"{px(i):.1f},{ys(v):.1f}"
                       for i, v in enumerate(vals))
        path = " -> ".join(f"{v:.2f}" for v in vals)
        tip = f"{metric} ({FAMILY[metric]}): {path}"
        fam = f"f-{FAMILY[metric]}"
        s.append(f'<polyline points="{pts}" fill="none" '
                 f'class="bline {fam}" '
                 f'stroke="{FAMILY_COLOR[FAMILY[metric]]}" stroke-width="1.6"'
                 f' stroke-opacity="0.7" data-tip="{esc(tip)}"/>')
        if vals[-1] > 1.25:
            s.append(f'<text class="{fam}" x="{px(n) + 5:.1f}" '
                     f'y="{ys(vals[-1]) + 3:.1f}" font-size="8.5" '
                     f'fill="{FAMILY_COLOR[FAMILY[metric]]}">{metric}</text>')
    mean_vals = [sum(root.values()) / len(METRICS)] + [
        sum(cache[str(pr)].values()) / len(METRICS) for pr in chain]
    pts = " ".join(f"{px(i):.1f},{ys(v):.1f}"
                   for i, v in enumerate(mean_vals))
    tip = "mean of all 21 rows: " + " -> ".join(f"{v:.3f}"
                                                for v in mean_vals)
    s.append(f'<polyline points="{pts}" fill="none" class="bline" '
             f'stroke="{INK}" stroke-width="3" data-tip="{esc(tip)}"/>')
    s.append(f'<text x="{px(n) + 5:.1f}" y="{ys(mean_vals[-1]) + 3:.1f}" '
             f'font-size="9" font-weight="700" fill="{INK}">mean</text>')
    return (f'<svg viewBox="0 0 {W} {H}" '
            f'font-family="system-ui, sans-serif">{"".join(s)}</svg>')


# --------------------------------------------------------------------- #
# layer 4: before / after evaluation figures
# --------------------------------------------------------------------- #
def before_after_layer():
    manifest = json.loads((DATA / "stack_visuals.json").read_text())
    order = ["main", "gnn", "gmlp"]
    per_metric = {}
    for stack in order:
        for name in manifest[stack]["files"]:
            per_metric.setdefault(name.split("_")[0], set()).add(name)
    metrics = [m for m in METRICS if m in per_metric]

    nav, cards = [], []
    for i, m in enumerate(metrics):
        color = FAMILY_COLOR[FAMILY[m]]
        nav.append(f'<button data-m="{m}" data-color="{color}" '
                   f'class="{"on" if not i else ""}" '
                   f'style="border-color:{color};color:'
                   f'{"#fcfcfb" if not i else color};background:'
                   f'{color if not i else "none"}">{m}</button>')
        rows = []
        for name in sorted(per_metric[m]):
            cells = []
            for stack in order:
                if name not in manifest[stack]["files"]:
                    continue
                cells.append(
                    f'<figure><img src="stack_visuals/{stack}/{name}" '
                    f'loading="lazy" alt="{esc(name)} ({stack})">'
                    f'<figcaption>{esc(manifest[stack]["label"])}'
                    f'</figcaption></figure>')
            rows.append(f'<div class="barow">{"".join(cells)}</div>')
        cards.append(f'<div class="bacard {"on" if not i else ""}" '
                     f'id="ba-{m}">{"".join(rows)}</div>')
    return "\n".join(nav), "\n".join(cards)


# --------------------------------------------------------------------- #
# page assembly
# --------------------------------------------------------------------- #
STYLE = """<style>
body { margin: 0; background: #fcfcfb; color: #1a1a19;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 18px 20px 40px; }
h1 { font-size: 20px; margin: 6px 0 2px; }
.sub { color: #52514e; font-size: 13px; margin: 0 0 14px; }
nav { display: flex; gap: 4px; border-bottom: 1px solid #e1e0d9;
  margin-bottom: 18px; flex-wrap: wrap; }
nav button, nav a { border: 0; background: none; color: #52514e;
  padding: 9px 14px; font: inherit; font-size: 13.5px; cursor: pointer;
  text-decoration: none; border-bottom: 2px solid transparent; }
nav button.on { color: #0b0b0b; font-weight: 600;
  border-bottom-color: #2a78d6; }
nav button:hover, nav a:hover { color: #0b0b0b; }
nav a::after { content: " \\2197"; font-size: 11px; }
.layer { display: none; }
.layer.on { display: block; }
svg { max-width: 100%; height: auto; }
.legend { color: #52514e; font-size: 12.5px; }
.grid21 { display: grid; grid-template-columns: repeat(7, 1fr); gap: 4px; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 900px) {
  .grid21 { grid-template-columns: repeat(3, 1fr); }
  .two { grid-template-columns: 1fr; }
}
.stacks { display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
  margin: 2px 0 16px; }
@media (max-width: 900px) { .stacks { grid-template-columns: 1fr; } }
.scard { border: 1px solid rgba(11,11,11,0.10); border-radius: 12px;
  background: #ffffff; padding: 12px 16px; }
.scard h3 { margin: 0 0 8px; font-size: 14px; }
.scard h3 small { color: #898781; font-weight: 500; font-size: 11.5px; }
.srow { display: flex; gap: 8px; align-items: baseline; font-size: 12.5px;
  color: #52514e; margin: 6px 0; line-height: 1.45; }
.schip { flex: none; display: inline-block; padding: 1px 8px;
  border-radius: 999px; font-size: 10.5px; font-weight: 600; color: #fff; }
.treehead { display: flex; align-items: baseline; gap: 8px; }
.treehead .legend { margin: 0; }
.focus { margin-left: auto; display: flex; gap: 6px; }
.focus button { border: 1.5px solid; border-radius: 999px; background: none;
  font: inherit; font-size: 12.5px; padding: 4px 13px; cursor: pointer; }
.focus button.on { color: #fcfcfb; font-weight: 600; }
#treebox.f-gnn .t-gmlp, #treebox.f-gnn .t-none,
#treebox.f-gmlp .t-gnn, #treebox.f-gmlp .t-none,
#treebox.f-both .v-fail { display: none; }
#breakbox.sel .f-contribution, #breakbox.sel .f-switch,
#breakbox.sel .f-punisher { display: none; }
#breakbox.sel.s-contribution .f-contribution,
#breakbox.sel.s-switch .f-switch,
#breakbox.sel.s-punisher .f-punisher { display: initial; }
.mlegend { display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 2px 26px; border-top: 1px solid #e1e0d9; margin-top: 20px;
  padding-top: 12px; }
.mlegend .mlhead { grid-column: 1 / -1; font-size: 11.5px; color: #898781;
  text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
.mlegend .mlhead small { text-transform: none; letter-spacing: 0; }
.mlegend .mlhead code { background: #f0efec; border-radius: 4px;
  padding: 1px 5px; font-size: 11px; color: #52514e; }
.mcol h4 { margin: 4px 0 5px; font-size: 12px; }
.mrow { display: grid; grid-template-columns: 36px 1fr; gap: 6px;
  margin: 5px 0; font-size: 12px; line-height: 1.4; align-items: baseline; }
.mrow b { color: #1a1a19; font-variant-numeric: tabular-nums; }
.mrow .n { color: #1a1a19; }
.mrow .g { display: block; color: #898781; font-size: 11.5px; }
.mlegend.rail { grid-template-columns: 1fr; border-top: 0; margin-top: 0;
  padding-top: 0; gap: 0; }
@media (max-width: 900px) { .mlegend { grid-template-columns: 1fr; } }
.storywrap { display: grid; grid-template-columns: 1fr 232px; gap: 26px;
  align-items: start; }
.storyrail { position: sticky; top: 14px; border: 1px solid
  rgba(11,11,11,0.10); border-radius: 12px; background: #ffffff;
  padding: 13px 16px; max-height: calc(100vh - 40px); overflow-y: auto; }
@media (max-width: 900px) {
  .storywrap { grid-template-columns: 1fr; }
  .storyrail { position: static; max-height: none; }
}
.mnav { display: flex; gap: 5px; flex-wrap: wrap; margin: 6px 0 14px; }
.mnav button { border: 1.5px solid; border-radius: 999px; background: none;
  font: inherit; font-size: 12px; padding: 3px 11px; cursor: pointer; }
.mnav button.on { font-weight: 600; }
.bacard { display: none; }
.bacard.on { display: block; }
.barow { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;
  margin-bottom: 14px; }
.barow figure { margin: 0; }
.barow img { width: 100%; border: 1px solid #e1e0d9; border-radius: 6px; }
.barow figcaption { font-size: 11.5px; color: #52514e; margin-top: 3px; }
@media (max-width: 900px) { .barow { grid-template-columns: 1fr; } }
a.node { cursor: pointer; }
a.node:hover circle, a.node:hover polygon, a.node:hover rect {
  stroke: #0b0b0b; stroke-width: 2; }
.bline { cursor: default; }
.bline:hover { stroke-opacity: 1; stroke-width: 3.5; }
#tip { position: fixed; display: none; background: #1a1a19; color: #fcfcfb;
  font-size: 12px; padding: 6px 9px; border-radius: 6px; max-width: 420px;
  pointer-events: none; z-index: 10; line-height: 1.4; }
</style>"""

SCRIPT = """<script>
const tip = document.getElementById("tip");
document.addEventListener("mouseover", e => {
  const t = e.target.closest("[data-tip]");
  if (!t) { tip.style.display = "none"; return; }
  tip.textContent = t.dataset.tip;
  tip.style.display = "block";
});
document.addEventListener("mousemove", e => {
  if (tip.style.display === "none") return;
  const x = Math.min(e.clientX + 14, window.innerWidth - 440);
  tip.style.left = x + "px";
  tip.style.top = (e.clientY + 16) + "px";
});
document.querySelectorAll("nav button").forEach(b => b.onclick = () => {
  document.querySelectorAll("nav button").forEach(x =>
    x.classList.toggle("on", x === b));
  document.querySelectorAll(".layer").forEach(l =>
    l.classList.toggle("on", l.id === b.dataset.layer));
});
function wireFocus(btnSel, boxId, toClass) {
  const box = document.getElementById(boxId);
  const btns = document.querySelectorAll(btnSel);
  btns.forEach(b => {
    b.style.borderColor = b.dataset.color;
    b.style.color = b.dataset.color;
    b.onclick = () => {
      b.classList.toggle("on");
      const on = b.classList.contains("on");
      b.style.background = on ? b.dataset.color : "";
      b.style.color = on ? "#fcfcfb" : b.dataset.color;
      box.className = toClass([...btns]
        .filter(x => x.classList.contains("on")).map(x => x.dataset.key));
    };
  });
}
wireFocus("#tree .focus button", "treebox", on =>
  on.length === 2 ? "f-both" : on.length === 1 ? "f-" + on[0] : "");
wireFocus("#breakdown .focus button", "breakbox", on =>
  "two" + (on.length ? " sel " + on.map(f => "s-" + f).join(" ") : ""));
const mbtns = document.querySelectorAll(".mnav button");
mbtns.forEach(b => b.onclick = () => {
  mbtns.forEach(x => {
    const on = x === b;
    x.classList.toggle("on", on);
    x.style.background = on ? x.dataset.color : "none";
    x.style.color = on ? "#fcfcfb" : x.dataset.color;
  });
  document.querySelectorAll(".bacard").forEach(c =>
    c.classList.toggle("on", c.id === "ba-" + b.dataset.m));
});
</script>"""


def main():
    experiments, trees, cache, root = load_spines()
    categories = {c["pr"]: c["category"] for c in
                  json.loads((DATA / "categories.json").read_text())}
    notes = json.loads((DATA / "machinery_notes.json").read_text())
    pr_slug = {pr: slug for slug, note in notes.items()
               for pr in note["prs"]}

    ba_nav, ba_cards = before_after_layer()
    panels = "\n".join(panel_svg(m, trees, cache, root, pr_slug)
                       for m in METRICS)
    breakdowns = "\n".join(
        breakdown_svg(tree, chain, cache, root, pr_slug)
        for tree, chain in trees.items())
    n_s = sum(e["verdict"] == "SUCCESS" for e in experiments)
    n_f = len(experiments) - n_s

    OUT.write_text(f"""<title>Autoresearch Campaign Report</title>
{STYLE}
<div class="wrap">
<h1>Autoresearch campaign report</h1>
<p class="sub">PRs #{min(e['pr'] for e in experiments)}&ndash;#{max(
        e['pr'] for e in experiments)} (frozen corpus): {n_s} successes,
{n_f} fails across two stacks. Hover any milestone for its PR; click it
for the plain-language story.</p>
<nav>
  <button class="on" data-layer="tree">Progress tree</button>
  <button data-layer="scores">All 21 scores</button>
  <button data-layer="breakdown">Score breakdown</button>
  <button data-layer="beforeafter">Before / after</button>
  <a href="machinery.html">Machinery</a>
  <a href="leaderboard.html">Leaderboard</a>
</nav>
<section class="layer on" id="tree">
{STACK_INTRO}
<div class="treehead">
  <p class="legend">Focus a tree, or select both to hide the failed
  attempts:</p>
  <div class="focus">
    <button data-key="gnn" data-color="{TREE_COLOR['gnn']}">gnn
    tree</button>
    <button data-key="gmlp" data-color="{TREE_COLOR['gmlp']}">gaussian-MLP
    tree</button>
  </div>
</div>
<div id="treebox">
{tree_svg(experiments, categories, pr_slug)}
</div>
{TREE_LEGEND}
</section>
<section class="layer" id="scores">
<p class="legend">Every evaluation row along the two success spines
(<span style="color:{TREE_COLOR['gnn']}">&#9473; gnn</span>,
<span style="color:{TREE_COLOR['gmlp']}">&#9473; gaussian-MLP</span>;
guides at the 1 / 2 / 5 band edges, log scale; titles colored by slot:
{" ".join(f'<span style="color:{c}">{f}</span>'
          for f, c in FAMILY_COLOR.items())}).</p>
<div class="grid21">{panels}</div>
{metric_legend()}
</section>
<section class="layer" id="breakdown">
<div class="treehead">
<p class="legend">All 21 rows per spine, colored by the slot each row
measures; the bold line is the 21-row mean (always shown) &mdash; hover a
line for its values, or focus one or more slots:</p>
  <div class="focus">
{"".join(f'    <button data-key="{f}" data-color="{c}">{f}</button>'
         for f, c in FAMILY_COLOR.items())}
  </div>
</div>
<div class="two" id="breakbox">{breakdowns}</div>
{metric_legend()}
</section>
<section class="layer" id="beforeafter">
<p class="legend">The evaluation suite's own figure for each score row,
before the campaign (the reference stack) and at both frontier tips,
side by side. Pick a row &mdash; rows with two figures show both pairs.
(SA has no figure; its score is a single rate.)</p>
<div class="mnav">{ba_nav}</div>
{ba_cards}
{metric_legend()}
</section>
</div>
<div id="tip"></div>
{SCRIPT}""")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
