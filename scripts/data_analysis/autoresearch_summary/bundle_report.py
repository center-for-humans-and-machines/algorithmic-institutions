"""Bundle the whole report set into ONE self-contained HTML file.

Everything the multi-file report links to becomes an in-page layer:
machinery and leaderboard join the nav, the eight story pages become a
Stories layer (machinery pills and tree nodes jump to them in place),
and the 72 before/after figures are inlined as data URIs. The output is
what gets shared (as a file, or published as a Claude artifact) — the
multi-file report stays the working copy.

Usage:
    python scripts/data_analysis/autoresearch_summary/bundle_report.py
"""

import base64
import json
import re
from pathlib import Path

import build_report as br
import machinery as mach
from score_progressions import DATA, FAMILY_COLOR, METRICS, TREE_COLOR

OUT = Path("plots/data_analysis/autoresearch_summary/report_bundle.html")
PLOTS = OUT.parent


def bundle_node_link(pr, pr_slug):
    """In the bundle, story links stay in-page as #story-<slug> jumps."""
    if pr in pr_slug:
        return f'href="#story-{pr_slug[pr]}"'
    return f'href="https://github.com/{br.REPO}/pull/{pr}" target="_blank"'


def inline_images(html_str):
    def to_data_uri(m):
        data = (PLOTS / m.group(1)).read_bytes()
        return ('src="data:image/jpeg;base64,'
                + base64.b64encode(data).decode() + '"')
    return re.sub(r'src="(stack_visuals/[^"]+)"', to_data_uri, html_str)


def machinery_layer(pr_slug):
    machines = json.loads((DATA / "stack_parts.json").read_text())["machines"]
    mach.PR_SLUG.update({f"#{pr}": slug for pr, slug in pr_slug.items()})
    body = mach.LEGEND + "\n".join(mach.render_machine(m) for m in machines)
    return body.replace('href="machinery_pages/', 'href="#story-') \
               .replace('.html"><title>the story behind',
                        '"><title>the story behind')


def stories_layer(notes):
    cards = []
    for slug, note in notes.items():
        prs = " + ".join(
            f'<a href="https://github.com/{br.REPO}/pull/{pr}" '
            f'target="_blank">#{pr}</a>' for pr in note["prs"])
        chip = (f'<span class="chip" style="background:'
                f'{mach.CAT_COLOR[note["category"]]}">'
                f'{br.esc(note["category"])}</span>')
        maths = "".join(f"<li>{br.esc(p)}</li>" for p in note["maths"])
        code = "".join(
            f'<li><code>{br.esc(c["path"])}</code> &mdash; '
            f'{br.esc(c["role"])}</li>' for c in note["code"])
        cards.append(f"""
<article class="story" id="story-{slug}">
<h3>{br.esc(note['title'])}</h3>
<p class="meta">{chip} installed by PR {prs} &middot;
{br.esc(note['where'])}</p>
<h4>The problem</h4><p>{br.esc(note['problem'])}</p>
<h4>The change</h4><p>{br.esc(note['change'])}</p>
<h4>The maths, in plain English</h4><ol class="maths">{maths}</ol>
<h4>Where it lives in the code</h4><ul class="code">{code}</ul>
<p class="meta">(paths as changed on the PR)</p>
<h4>What it bought</h4><p>{br.esc(note['bought'])}</p>
</article>""")
    return "\n".join(cards)


def leaderboard_layer():
    html_str = (PLOTS / "leaderboard.html").read_text()
    rows = re.search(r"const ROWS = (\[.*?\]);", html_str).group(1)
    return rows


EXTRA_STYLE = """<style>
/* machinery + stories + leaderboard, bundled */
#machinery h2 { font-size: 16px; margin: 18px 0 6px; }
#machinery svg a { cursor: pointer; }
#machinery svg a:hover rect { stroke: #2a78d6; stroke-width: 1.5; }
.story { border: 1px solid rgba(11,11,11,0.10); border-radius: 12px;
  background: #ffffff; padding: 18px 22px; margin: 0 0 18px;
  max-width: 720px; }
.story h3 { margin: 0 0 4px; font-size: 16px; }
.story h4 { font-size: 11.5px; text-transform: uppercase;
  letter-spacing: 0.05em; color: #898781; margin: 14px 0 3px; }
.story p { margin: 3px 0; line-height: 1.5; font-size: 13.5px; }
.story .meta { color: #52514e; font-size: 12px; }
.story .meta a { color: #2a78d6; text-decoration: none; }
.chip { display: inline-block; padding: 2px 9px; border-radius: 999px;
  font-size: 11px; font-weight: 600; color: #fff; margin-right: 6px; }
ol.maths { padding-left: 1.3rem; margin: 4px 0; }
ol.maths li { margin: 7px 0; line-height: 1.5; font-size: 13.5px; }
ul.code { list-style: none; padding: 0; margin: 4px 0; }
ul.code li { margin: 5px 0; font-size: 12.5px; color: #52514e; }
code { background: #f0efec; border-radius: 4px; padding: 1px 5px;
  font-size: 11.5px; color: #1a1a19; }
#lb table { width: 100%; border-collapse: collapse; background: #ffffff;
  border: 1px solid rgba(11,11,11,0.10); border-radius: 12px;
  overflow: hidden; }
#lb thead th { text-align: left; font-size: 11px; text-transform: uppercase;
  letter-spacing: 0.04em; color: #898781; font-weight: 600;
  padding: 9px 11px; border-bottom: 1px solid #e1e0d9; }
#lb thead th.num, #lb td.num { text-align: right;
  font-variant-numeric: tabular-nums; }
#lb tbody td { padding: 8px 11px; border-bottom: 1px solid #e1e0d9;
  font-size: 13px; }
#lb tbody tr:last-child td { border-bottom: 0; }
#lb td.on, #lb thead th.on { background: #f0f4fb; }
#lb .rank { display: inline-grid; place-items: center; width: 23px;
  height: 23px; border-radius: 50%; font-size: 11.5px; font-weight: 700;
  color: #52514e; background: #fcfcfb; border: 1px solid #e1e0d9; }
#lb tr:nth-child(1) .rank { background: #f5c518; color: #3a2f00;
  border-color: transparent; }
#lb tr:nth-child(2) .rank { background: #c8c8c4; color: #333;
  border-color: transparent; }
#lb tr:nth-child(3) .rank { background: #d9a068; color: #402a10;
  border-color: transparent; }
#lb .pr a { color: #2a78d6; text-decoration: none; font-weight: 600; }
#lb .label { font-weight: 550; }
#lb .note { color: #898781; font-size: 11px; margin-top: 1px; }
#lb .pill { display: inline-block; padding: 2px 8px; border-radius: 999px;
  font-size: 10.5px; font-weight: 600; color: #fff; }
#lb .stack { color: #52514e; font-size: 11.5px; }
#lb .good { color: #006300; font-weight: 600; }
#lb .bad { color: #d03b3b; font-weight: 600; }
#lb .zero { color: #898781; }
#lb .means { color: #898781; font-size: 10.5px; display: block; }
#lb .seg { display: inline-flex; border: 1px solid #e1e0d9;
  border-radius: 9px; overflow: hidden; margin-bottom: 14px; }
#lb .seg button { border: 0; background: #ffffff; color: #52514e;
  padding: 7px 13px; font: inherit; font-size: 12.5px; cursor: pointer;
  border-right: 1px solid #e1e0d9; }
#lb .seg button:last-child { border-right: 0; }
#lb .seg button.on { background: #f0f4fb; color: #0b0b0b;
  font-weight: 600; }
</style>"""

LB_SCRIPT = """<script>
(() => {
const ROWS = __ROWS__;
const CRITERIA = [
  {key: "d_mean", name: "\\u0394 mean", dir: 1},
  {key: "d_le1", name: "\\u0394 rows \\u2264 1", dir: -1},
  {key: "d_gt2", name: "\\u0394 rows > 2", dir: 1},
  {key: "upgrades", name: "Band upgrades", dir: -1},
];
const SLOT = {contribution: "#2a78d6", switch: "#eb6834",
              punisher: "#1baf7a"};
let active = "d_mean";
const fmt = (v, k) => {
  const good = k === "d_mean" || k === "d_gt2" ? v < 0 : v > 0;
  const bad = k === "d_mean" || k === "d_gt2" ? v > 0
            : k === "upgrades" ? false : v < 0;
  const cls = good ? "good" : bad ? "bad" : "zero";
  const s = k === "d_mean"
    ? (v > 0 ? "+" : "\\u2212") + Math.abs(v).toFixed(3)
    : (v > 0 ? "+" : v < 0 ? "\\u2212" : "\\u00b1") + Math.abs(v);
  return [s, cls];
};
function render() {
  const dir = CRITERIA.find(c => c.key === active).dir;
  const rows = [...ROWS].sort((a, b) =>
    dir * (a[active] - b[active]) || a.d_mean - b.d_mean);
  document.getElementById("lb-body").innerHTML = rows.map((r, i) => {
    const cells = CRITERIA.map(c => {
      const [s, cls] = fmt(r[c.key], c.key);
      const extra = c.key === "d_mean"
        ? `<span class="means">${r.base_mean.toFixed(3)} \\u2192 ` +
          `${r.cand_mean.toFixed(3)}</span>` : "";
      const tip = c.key === "upgrades" && r.up_rows
        ? ` title="upgraded: ${r.up_rows}` +
          (r.down_rows ? ` / downgraded: ${r.down_rows}` : "") + `"` : "";
      return `<td class="num ${c.key === active ? "on" : ""}"${tip}>` +
             `<span class="${cls}">${s}</span>${extra}</td>`;
    }).join("");
    return `<tr><td><span class="rank">${i + 1}</span></td>
      <td class="pr"><a target="_blank"
        href="https://github.com/__REPO__/pull/${r.pr}">#${r.pr}</a></td>
      <td><div class="label">${r.label}</div>
        ${r.note ? `<div class="note">${r.note}</div>` : ""}</td>
      <td><span class="pill" style="background:${SLOT[r.slot]}">
        ${r.slot}</span></td>
      <td class="stack">${r.stack}</td>${cells}</tr>`;
  }).join("");
  document.querySelectorAll("#lb thead th[data-c]").forEach(th =>
    th.classList.toggle("on", th.dataset.c === active));
}
const seg = document.getElementById("lb-seg");
seg.innerHTML = CRITERIA.map(c =>
  `<button data-c="${c.key}">${c.name}</button>`).join("");
seg.querySelectorAll("button").forEach(b => b.onclick = () => {
  active = b.dataset.c;
  seg.querySelectorAll("button").forEach(x =>
    x.classList.toggle("on", x === b));
  render();
});
seg.querySelector("button").classList.add("on");
render();
})();
// in-page story jumps: activate the Stories layer, then scroll
function showLayer(id) {
  document.querySelectorAll("nav button").forEach(x =>
    x.classList.toggle("on", x.dataset.layer === id));
  document.querySelectorAll(".layer").forEach(l =>
    l.classList.toggle("on", l.id === id));
}
document.addEventListener("click", e => {
  const a = e.target.closest('a[href^="#story-"]');
  if (!a) return;
  e.preventDefault();
  showLayer("stories");
  const el = document.getElementById(a.getAttribute("href").slice(1));
  if (el) el.scrollIntoView({behavior: "smooth", block: "start"});
});
</script>"""


def main():
    experiments, trees, cache, root = br.load_spines()
    categories = {c["pr"]: c["category"] for c in
                  json.loads((DATA / "categories.json").read_text())}
    notes = json.loads((DATA / "machinery_notes.json").read_text())
    pr_slug = {pr: slug for slug, note in notes.items()
               for pr in note["prs"]}

    br.node_link = bundle_node_link  # story links stay in-page
    ba_nav, ba_cards = br.before_after_layer()
    panels = "\n".join(br.panel_svg(m, trees, cache, root, pr_slug)
                       for m in METRICS)
    breakdowns = "\n".join(
        br.breakdown_svg(tree, chain, cache, root, pr_slug)
        for tree, chain in trees.items())
    n_s = sum(e["verdict"] == "SUCCESS" for e in experiments)
    lo, hi = (min(e["pr"] for e in experiments),
              max(e["pr"] for e in experiments))

    page = f"""<meta charset="utf-8">
<title>Autoresearch Atlas</title>
{br.STYLE}
{EXTRA_STYLE}
<div class="wrap">
<h1>Autoresearch Atlas</h1>
<p class="sub">An autonomous research campaign on the artificial-humans
stacks: PRs #{lo}&ndash;#{hi} (frozen corpus), {n_s} successes,
{len(experiments) - n_s} fails across two stacks. Hover any milestone
for its PR; click it for the plain-language story.</p>
<nav>
  <button class="on" data-layer="tree">Progress tree</button>
  <button data-layer="scores">All 21 scores</button>
  <button data-layer="breakdown">Score breakdown</button>
  <button data-layer="beforeafter">Before / after</button>
  <button data-layer="machinery">Machinery</button>
  <button data-layer="lb">Leaderboard</button>
  <button data-layer="stories">Stories</button>
</nav>
<section class="layer on" id="tree">
{br.STACK_INTRO}
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
{br.tree_svg(experiments, categories, pr_slug)}
</div>
{br.TREE_LEGEND}
</section>
<section class="layer" id="scores">
<p class="legend">Every evaluation row along the two success spines
(<span style="color:{TREE_COLOR['gnn']}">&#9473; gnn</span>,
<span style="color:{TREE_COLOR['gmlp']}">&#9473; gaussian-MLP</span>;
guides at the 1 / 2 / 5 band edges, log scale; titles colored by slot:
{" ".join(f'<span style="color:{c}">{f}</span>'
          for f, c in FAMILY_COLOR.items())}).</p>
<div class="grid21">{panels}</div>
{br.metric_legend()}
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
{br.metric_legend()}
</section>
<section class="layer" id="beforeafter">
<p class="legend">The evaluation suite's own figure for each score row,
before the campaign (the reference stack) and at both frontier tips,
side by side. Pick a row &mdash; rows with two figures show both pairs.
(SA has no figure; its score is a single rate.)</p>
<div class="mnav">{ba_nav}</div>
{inline_images(ba_cards)}
{br.metric_legend()}
</section>
<section class="layer" id="machinery">
{machinery_layer(pr_slug)}
</section>
<section class="layer" id="lb">
<p class="legend">The {len(pr_slug) + 3} unique successful methods, each
scored against its own baseline stack. Pick the ranking criterion:</p>
<div class="seg" id="lb-seg"></div>
<table>
  <thead><tr>
    <th></th><th>PR</th><th>Method</th><th>Slot</th><th>Stack</th>
    <th class="num" data-c="d_mean">&Delta; mean</th>
    <th class="num" data-c="d_le1">&Delta; rows &le; 1</th>
    <th class="num" data-c="d_gt2">&Delta; rows &gt; 2</th>
    <th class="num" data-c="upgrades">Band upgrades</th>
  </tr></thead>
  <tbody id="lb-body"></tbody>
</table>
<p class="legend">&Delta; values are candidate &minus; baseline over the
21 rows; &Delta; rows &gt; 2 ranks reversed (fewer badly-missed rows is
better). Hover a band-upgrade count for the rows. Duplicates folded:
#146 into #160, #172 into #171. Ties break on &Delta; mean.</p>
</section>
<section class="layer" id="stories">
<p class="legend">The plain-language story of every unique successful
method &mdash; also reachable by clicking tree nodes and machinery
pills.</p>
<div class="storywrap">
<div>{stories_layer(notes)}</div>
<aside class="storyrail">{br.metric_legend(rail=True)}</aside>
</div>
</section>
</div>
<div id="tip"></div>
{br.SCRIPT}
{LB_SCRIPT.replace("__ROWS__", leaderboard_layer())
          .replace("__REPO__", br.REPO)}"""

    OUT.write_text(page)
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
