"""Rebaseline Atlas: sibling of the Autoresearch Atlas (report_bundle.html), new data.

CSS and JS are copied verbatim from the template; the sections are re-rendered
from the CSVs of the punisher re-baseline (worktree D), the RCE comparison
(worktree R) and the held-out teacher-forced test (worktree H).
"""
import base64, html, io, json, math, re
from pathlib import Path

import pandas as pd

D = Path("/Users/brinkmann/repros/algorithmic-institutions/.claude/worktrees/agent-a80d622b939db4c1c")
R = Path("/Users/brinkmann/repros/algorithmic-institutions/.claude/worktrees/agent-a91a63a720dcceb19")
H = Path("/Users/brinkmann/repros/algorithmic-institutions/.claude/worktrees/agent-a73127aa5d3033a41")
TPL = Path("/Users/brinkmann/Downloads/report_bundle.html")
HERE = Path(__file__).resolve().parent
OUT = HERE / "rebaseline_atlas.html"
PC = D / "plots/data_analysis/evaluation/punisher_current_contr"
PRURL = "https://github.com/center-for-humans-and-machines/algorithmic-institutions/pull/"
SIZE_LIMIT = 14 * 1024 * 1024

esc = lambda s: html.escape(str(s), quote=True)

# ---------------------------------------------------------------- template
tpl = TPL.read_text()
STYLES = re.findall(r"<style>.*?</style>", tpl, re.S)
SCRIPTS = re.findall(r"<script>.*?</script>", tpl, re.S)
assert len(STYLES) == 2 and len(SCRIPTS) == 2

# ---------------------------------------------------------------- data
SPINE = {"gnn": "#6b6a66", "gmlp": "#b08968"}
SLOT = {"contribution": "#2a78d6", "switch": "#eb6834", "punisher": "#1baf7a"}
INK, PAPER, GREY, MUTED, GRID = "#0b0b0b", "#fcfcfb", "#c9c8c3", "#52514e", "#eceae6"
FIX = SLOT["punisher"]

# case key, spine, PR, label, punisher pair, short label, story for the before node
CASES = [
    ("e_lin", "gnn", 184, "main sweep gnn x gnn, plain multinomial punisher",
     "lin multinomial (lagged) -> lin multinomial current-contr", "main · lin"),
    ("e_gnn", "gnn", 184, "main sweep gnn x gnn, GNN punisher",
     "GNN punisher (lagged) -> GNN punisher current-contr", "main · gnn"),
    ("a_vnode", "gnn", 179, "PR #179 group vnode",
     "lin multinomial + severity copula -> current-contr + copula", "#179"),
    ("b_skip", "gnn", 181, "PR #181 stimulus skip",
     "lin multinomial + severity copula -> current-contr + copula", "#181"),
    ("d_kexo", "gmlp", 174, "PR #174 k-one-hot switch on the Gaussian-MLP v2 group copula",
     "lin multinomial + severity copula -> current-contr + copula", "#174"),
    ("c_infl", "gmlp", 177, "PR #177 inflated Gaussian-MLP",
     "lin multinomial + severity copula -> current-contr + copula", "#177"),
]
CASE = {c[0]: c for c in CASES}
ORDER = [c[0] for c in CASES]
SIM = {
    "a_vnode": "23_2g8a_contr_group_vnode_self_gnncopar1_contr_gnn_switch",
    "b_skip": "23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch",
    "c_infl": "23_2g8a_infl_self_gaussian_mlp_inflated_group_copula_contr_gnn_joint_exodus_k_onehot_switch",
    "d_kexo": "23_2g8a_kexo_self_gaussian_mlp_v2_group_copula_contr_gnn_joint_exodus_k_onehot_switch",
    "e_lin": "23_2g8a_self_gnn_contr_gnn_switch",
    "e_gnn": "23_2g8a_self_gnn_contr_gnn_switch",
}
ROWS = ["CA", "CB", "CC", "CD", "CE", "CF", "CG", "RCA", "RCB", "RCC", "RCD", "RCE",
        "SA", "SB", "SC", "RSA", "PA", "PB", "PC", "PD", "RPA", "RPB"]
ROWSLOT = {r: "contribution" if r.lstrip("R")[0] == "C" else "switch" if r.lstrip("R")[0] == "S"
           else "punisher" for r in ROWS}
ROWNAME = {"CA": "participant means", "CB": "round means", "CC": "group means", "CD": "raw contributions",
           "CE": "signed group differences", "CF": "boundary shares", "CG": "group-spread ratio",
           "SA": "switch rate", "SB": "switch timing", "SC": "size of the larger group",
           "PA": "punishment levels", "PB": "punishment by round", "PC": "share unpunished",
           "PD": "punishment spread ratio", "RCA": "change by round type",
           "RCB": "reaction to punishment (rate bins)", "RCC": "reaction at the ceiling",
           "RCD": "switching pull", "RCE": "punishment response slope",
           "RSA": "switching after punishment", "RPA": "the manager's policy",
           "RPB": "punishment by group size"}

tab = pd.read_csv(PC / "rebaseline_table.csv", index_col=0)
S = {c: {st: tab[f"{c}_{st}"] for st in ("before", "after")} for c in ORDER}
MEAN = {c: {st: float(tab.loc["mean", f"{c}_{st}"]) for st in ("before", "after")} for c in ORDER}
LE1 = {c: {st: int(tab.loc["rows <= 1", f"{c}_{st}"]) for st in ("before", "after")} for c in ORDER}
for c in ORDER:  # the CSV's summary lines must agree with the 22 rows
    for st in ("before", "after"):
        v = S[c][st].loc[ROWS]
        assert abs(v.mean() - MEAN[c][st]) < 1e-9 and int((v <= 1).sum()) == LE1[c][st]

bands = pd.read_csv(PC / "rce_bands.csv").set_index(["case", "stage"])
BANDS = ["0-4", "5-9", "10-14", "15-19"]
mech = pd.read_csv(PC / "mechanism_selfplay.csv").set_index("sim")
hum_fit = pd.read_csv(R / "plots/data_analysis/evaluation/rcb_alternative/human_rce_fit.csv", index_col=0)
HUMAN_SLOPES = [float(hum_fit.loc[b, "slope"]) for b in BANDS]
ceil = pd.read_csv(R / "plots/data_analysis/evaluation/rcb_alternative/noise_ceilings.csv", index_col=0)
CEIL = {k: float(ceil.loc[k, "ceiling"]) for k in ("RCB", "RCE")}
cmp = pd.read_csv(R / "plots/data_analysis/evaluation/rcb_alternative/comparison_table.csv")
SPEARMAN = float(cmp["RCB_score"].corr(cmp["RCE_score"], method="spearman"))
SPEAR_SIGNS = {k: float(cmp[f"{k}_score"].corr(cmp["signs_match"], method="spearman")) for k in ("RCB", "RCE")}
PR179_SELFPLAY_RCB = float(cmp.loc[cmp["model"] == "PR 179 group vnode", "RCB_d"].iloc[0])
ho = pd.read_csv(H / "plots/data_analysis/rcb_holdout_teacher_forced/group_vnode_raven_gpu.csv")
hom = ho[ho["source"] == "model"].set_index("set")
HO = {k: float(hom.loc[k, "rcb_stat"]) for k in ("full_in_sample", "pooled_held_out", "pooled_in_sample")}
HO_SLOPES = {k: [float(hom.loc[k, f"slope_{b}"]) for b in BANDS] for k in ("pooled_held_out", "pooled_in_sample")}
ho_folds = ho[(ho["set"] == "held_out")].set_index(["fold", "source"])["rcb_stat"]
HO_FOLD_RANGE = (float(ho_folds.xs("model", level="source").min()), float(ho_folds.xs("model", level="source").max()))
HO_HUM_RANGE = (float(ho_folds.xs("human", level="source").min()), float(ho_folds.xs("human", level="source").max()))
# numbers with no CSV in any worktree (from the experiment log, stage C)
RHO = {"before": 0.3508, "after": 0.4273, "ci_before": (0.2780, 0.4232), "ci_after": (0.3514, 0.5283), "se": 0.0459}
CV = {"lin": (1.3661, 1.3465), "gnn": (1.2030, 1.1756), "lin_test": (1.3031, 1.2468), "floor": 1.3561}
SYNTH = {"none": 1.42, "half": 0.82}  # reports/rcb_alternative_comparison.md, RCE synthetic responses


def band(v):
    return 0 if v <= 1 else 1 if v <= 2 else 2 if v <= 5 else 3


BANDLBL = ["&lt;= 1", "1-2", "2-5", "&gt; 5"]
BANDTXT = ["<= 1", "1-2", "2-5", "> 5"]
f3 = lambda v: f"{v:.3f}"
f2 = lambda v: f"{v:.2f}"
sgn = lambda v, d=3: f"{v:+.{d}f}"
arrow = lambda c, r: f"{f3(S[c]['before'][r])} -> {f3(S[c]['after'][r])}"
band_arrow = lambda c, r: f"{BANDTXT[band(S[c]['before'][r])]} -> {BANDTXT[band(S[c]['after'][r])]}"
pr_link = lambda n: f'<a href="{PRURL}{n}" target="_blank">#{n}</a>'


def stats(c):
    b, a = S[c]["before"].loc[ROWS], S[c]["after"].loc[ROWS]
    up = [r for r in ROWS if band(a[r]) < band(b[r])]
    down = [r for r in ROWS if band(a[r]) > band(b[r])]
    return dict(base_mean=round(MEAN[c]["before"], 4), cand_mean=round(MEAN[c]["after"], 4),
                d_mean=round(MEAN[c]["after"] - MEAN[c]["before"], 4),
                d_le1=LE1[c]["after"] - LE1[c]["before"],
                d_gt2=int((a > 2).sum()) - int((b > 2).sum()),
                upgrades=len(up), up_rows=", ".join(up), down_rows=", ".join(down))


ST = {c: stats(c) for c in ORDER}
NODE_TIP = {c: (f"{CASE[c][3]} ({CASE[c][1]} spine) | mean {f3(MEAN[c]['before'])} -> {f3(MEAN[c]['after'])}"
               f" | rows <= 1: {LE1[c]['before']} -> {LE1[c]['after']}"
               f" | RCE {arrow(c, 'RCE')} ({band_arrow(c, 'RCE')})") for c in ORDER}

# ---------------------------------------------------------------- 1. stack cards
def stack_cards():
    reruns = lambda sp: ", ".join(CASE[c][5] for c in ORDER if CASE[c][1] == sp)
    def card(sp, name, contr, switch, pun):
        return (f'<div class="scard">\n<h3 style="color:{SPINE[sp]}">{name} <small>(reruns {esc(reruns(sp))})</small></h3>\n'
                f'<div class="srow"><span class="schip" style="background:{SLOT["contribution"]}">contribution</span><span>{contr}</span></div>\n'
                f'<div class="srow"><span class="schip" style="background:{SLOT["switch"]}">switch</span><span>{switch}</span></div>\n'
                f'<div class="srow"><span class="schip" style="background:{SLOT["punisher"]}">punisher</span><span>{pun}</span></div>\n</div>')
    pun = (f"a multinomial logistic regression over 31 punishment levels, now retrained on the current round's contribution "
           f"(it used to read last round's), its group draws coupled by the severity copula, recalibrated from rho {RHO['before']:.3f} to {RHO['after']:.3f}")
    return ('<div class="stacks">\n' +
            card("gnn", "the gnn stack",
                 "a graph neural network: members exchange messages each round, each keeps a small recurrent memory; "
                 "the #179 stack adds a per-group virtual node, the #181 stack a direct punishment-to-output skip; draws coupled by a herding copula",
                 "a graph-network switch predictor; on decision rounds a joint head draws how many leave each group, then a conditional-Bernoulli step picks who",
                 pun + " -- the main-sweep reference stack runs the plain multinomial (no copula) and, as a second run, the GNN punisher, both retrained the same way") +
            card("gmlp", "the gaussian-MLP stack",
                 "a tiny 2-layer neural network predicting a bell curve (centre and spread) per member with a group copula on the draws; "
                 "the #177 stack adds probability spikes at repeat / 0 / 20",
                 "the same graph-network switch predictor and joint exodus head, with the group sizes one-hot encoded (#174)",
                 pun + " -- unchanged between the two stacks, as before") +
            '</div>')


# ---------------------------------------------------------------- 2. progress tree
def tree_svg():
    W, Hh = 1150, 560
    x0, x1, ytop, ybot = 56, 1126, 60, 500
    m_lo, m_hi = 0.95, 1.95
    y = lambda m: ybot - (m - m_lo) / (m_hi - m_lo) * (ybot - ytop)
    o = [f'<svg viewBox="0 0 {W} {Hh}" font-family="system-ui, sans-serif">']
    for g in (1.0, 1.2, 1.4, 1.6, 1.8):
        o.append(f'<line x1="{x0}" y1="{y(g):.1f}" x2="{x1}" y2="{y(g):.1f}" stroke="{GRID}"/>'
                 f'<text x="{x0-8}" y="{y(g)+3:.1f}" text-anchor="end" font-size="10" fill="{MUTED}">{g:.1f}</text>')
    o.append(f'<text x="14" y="272" font-size="11" fill="{MUTED}" transform="rotate(-90 14 272)" text-anchor="middle">stack mean score over 22 rows (lower is better)</text>')
    o.append(f'<text x="{(x0+x1)/2:.1f}" y="550" font-size="11" fill="{MUTED}" text-anchor="middle">the six reruns: hollow = before (lagged punisher), filled = after (current-contribution punisher)</text>')
    xb = {c: 130 + i * 180 for i, c in enumerate(ORDER)}
    xa = {c: xb[c] + 80 for c in ORDER}
    pb = {c: (xb[c], y(MEAN[c]["before"])) for c in ORDER}
    pa = {c: (xa[c], y(MEAN[c]["after"])) for c in ORDER}
    # lineage spines between the before states: main -> #179 -> #181 (gnn), main -> #174 -> #177 (gmlp)
    def spine(a, b, sp, dashed=False):
        (ax, ay), (bx, by) = pb[a], pb[b]
        dash = ' stroke-dasharray="6 4" stroke-opacity="0.55"' if dashed else ""
        o.append(f'<line class="t-{sp} v-success" x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{SPINE[sp]}" stroke-width="{1.1 if dashed else 2.4}"{dash}/>')
    spine("e_lin", "a_vnode", "gnn"); spine("a_vnode", "b_skip", "gnn")
    spine("e_lin", "d_kexo", "gmlp"); spine("d_kexo", "c_infl", "gmlp")
    spine("e_lin", "e_gnn", "gnn", dashed=True)
    # dotted step: best after-mean so far, in x order
    best, pts = math.inf, []
    for c in ORDER:
        best = min(best, MEAN[c]["after"])
        pts.append((xb[c], y(best))); pts.append((xa[c], y(best)))
    d = " ".join(f"{'M' if i == 0 else 'L'} {x:.1f} {yy:.1f}" for i, (x, yy) in enumerate(pts))
    o.append(f'<path d="{d}" fill="none" stroke="#8f8e89" stroke-dasharray="2 4"/>')
    # punisher-fix edges and nodes
    for c in ORDER:
        sp = CASE[c][1]; cls = "" if c == "e_lin" else f" t-{sp}"
        (bx, by), (ax, ay) = pb[c], pa[c]
        o.append(f'<line class="v-success{cls}" x1="{bx:.1f}" y1="{by:.1f}" x2="{ax:.1f}" y2="{ay:.1f}" stroke="{FIX}" stroke-width="2.4"/>')
        tip = esc(NODE_TIP[c])
        o.append(f'<a href="#story-rebaseline" class="node v-success{cls}" data-tip="before: {tip}">'
                 f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="7.5" fill="{PAPER}" stroke="{SPINE[sp]}" stroke-width="2"/>'
                 f'<text x="{bx:.1f}" y="{by-13:.1f}" text-anchor="middle" font-size="9" fill="{MUTED}">{esc(CASE[c][5])}</text></a>')
        o.append(f'<a href="#story-punisher" class="node v-success{cls}" data-tip="after the punisher fix: {tip}">'
                 f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="7.5" fill="{SPINE[sp]}" stroke="{PAPER}" stroke-width="1.5"/>'
                 f'<text x="{ax:.1f}" y="{ay+18:.1f}" text-anchor="middle" font-size="9" fill="{MUTED}">{f3(MEAN[c]["after"])}</text></a>')
    o.append("</svg>")
    return "".join(o)


# ---------------------------------------------------------------- 3. all 22 scores
def small_chart(r):
    ylog = lambda v: 98.6 - 29.89 * math.log(v)
    col = SLOT[ROWSLOT[r]]
    o = [f'<svg viewBox="0 0 210 150" font-family="system-ui, sans-serif">'
         f'<text x="116.0" y="13" text-anchor="middle" font-size="11" font-weight="600" fill="{col}">{r}</text>']
    if r == "RCE":
        o.append(f'<text x="116.0" y="22" text-anchor="middle" font-size="7" font-weight="600" fill="#4a3aa7">protected row</text>')
    for g in (1, 2, 5):
        o.append(f'<line x1="30" y1="{ylog(g):.1f}" x2="202" y2="{ylog(g):.1f}" stroke="{GRID}"/>'
                 f'<text x="26" y="{ylog(g)+3:.1f}" text-anchor="end" font-size="8" fill="{MUTED}">{g}</text>')
    for i, c in enumerate(ORDER):
        x = 30 + i * 34.4; sp = CASE[c][1]
        b, a = float(S[c]["before"][r]), float(S[c]["after"][r])
        tip = esc(f"{CASE[c][3]} ({sp} spine) - {r} {f3(b)} -> {f3(a)} ({band_arrow(c, r)})")
        o.append(f'<a href="#story-rebaseline" class="node" data-tip="{tip}">'
                 f'<line x1="{x-3:.1f}" y1="{ylog(b):.1f}" x2="{x+3:.1f}" y2="{ylog(a):.1f}" stroke="{SPINE[sp]}" stroke-width="2"/>'
                 f'<circle cx="{x-3:.1f}" cy="{ylog(b):.1f}" r="4" fill="{PAPER}" stroke="{SPINE[sp]}" stroke-width="1.5"/>'
                 f'<circle cx="{x+3:.1f}" cy="{ylog(a):.1f}" r="4" fill="{SPINE[sp]}" stroke="{PAPER}" stroke-width="1"/></a>')
        o.append(f'<text x="{x:.1f}" y="142" text-anchor="middle" font-size="6.5" fill="{MUTED}">{esc(CASE[c][5].replace("main · ", ""))}</text>')
    o.append("</svg>")
    return "".join(o)


# ---------------------------------------------------------------- 4. breakdown
def breakdown_svg(sp):
    cases = [c for c in ORDER if CASE[c][1] == sp]
    ylog = lambda v: 275.3 - 94.07 * math.log(v)
    xs = [40 + i * (460 / (len(cases) - 1)) for i in range(len(cases))]
    o = [f'<svg viewBox="0 0 560 400" font-family="system-ui, sans-serif">'
         f'<text x="270.0" y="16" text-anchor="middle" font-size="12.5" font-weight="600" fill="{INK}">{sp} spine: {" &#8594; ".join(esc(CASE[c][5]) for c in cases)}, before (dashed) and after (solid)</text>']
    for g in (1, 2, 5):
        o.append(f'<line x1="40" y1="{ylog(g):.1f}" x2="500" y2="{ylog(g):.1f}" stroke="{GRID}"/>'
                 f'<text x="35" y="{ylog(g)+3:.1f}" text-anchor="end" font-size="9" fill="{MUTED}">{g}</text>')
    for x, c in zip(xs, cases):
        o.append(f'<text x="{x:.1f}" y="392" text-anchor="middle" font-size="9" fill="{MUTED}">{esc(CASE[c][5])}</text>')
    labels = []
    def line(vals, col, cls, tip, dashed, width=1.6, op=0.7):
        pts = " ".join(f"{x:.1f},{ylog(v):.1f}" for x, v in zip(xs, vals))
        dash = ' stroke-dasharray="4 3"' if dashed else ""
        o.append(f'<polyline points="{pts}" fill="none" class="bline{cls}" stroke="{col}" stroke-width="{width}" stroke-opacity="{op}"{dash} data-tip="{esc(tip)}"/>')
    for r in ROWS:
        col = SLOT[ROWSLOT[r]]; cls = f" f-{ROWSLOT[r]}"
        bv = [float(S[c]["before"][r]) for c in cases]; av = [float(S[c]["after"][r]) for c in cases]
        line(bv, col, cls, f"{r} ({ROWSLOT[r]}) before: " + " -> ".join(f2(v) for v in bv), True, 1.2, 0.5)
        line(av, col, cls, f"{r} ({ROWSLOT[r]}) after: " + " -> ".join(f2(v) for v in av), False)
        labels.append((ylog(av[-1]), r, col, cls))
    mb = [MEAN[c]["before"] for c in cases]; ma = [MEAN[c]["after"] for c in cases]
    line(mb, INK, "", "mean of all 22 rows, before: " + " -> ".join(f3(v) for v in mb), True, 2, 0.6)
    line(ma, INK, "", "mean of all 22 rows, after: " + " -> ".join(f3(v) for v in ma), False, 3, 1)
    last = -99
    for yy, r, col, cls in sorted(labels):
        if yy - last >= 9:
            o.append(f'<text class="{cls.strip()}" x="505.0" y="{yy+3:.1f}" font-size="8.5" fill="{col}">{r}</text>'); last = yy
    o.append(f'<text x="505.0" y="{ylog(ma[-1])+3:.1f}" font-size="9" font-weight="700" fill="{INK}">mean</text></svg>')
    return "".join(o)


# ---------------------------------------------------------------- 5. before / after browser
BA_STACKS = ["b_skip", "d_kexo", "e_gnn"]
EMBEDDED, MISSING = [], []
_b64cache = {}


def b64(p):
    if p not in _b64cache:
        _b64cache[p] = "data:image/jpeg;base64," + base64.b64encode(Path(p).read_bytes()).decode()
    return _b64cache[p]


def shrink_all(width=800, quality=78):
    from PIL import Image
    for p in list(_b64cache):
        im = Image.open(p).convert("RGB"); im.thumbnail((width, width))
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=quality, optimize=True)
        _b64cache[p] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def fig_files(r):
    vis = D / "plots/simulation" / (SIM["b_skip"] + "_curpun") / "evaluation/visuals"
    return sorted(f.name for f in vis.glob(f"{r}_*.jpg"))


def ba_cards():
    rows = [r for r in ROWS if fig_files(r)]
    def stack_name(c): return CASE[c][3].split(",")[0]
    def figure(r, fname, c, st):
        d = D / "plots/simulation" / (SIM[c] + ("_curpun" if st == "after" else "")) / "evaluation/visuals" / fname
        v = float(S[c][st][r])
        who = "lagged punisher" if st == "before" else "current-contribution punisher"
        cap = f"{st}: {esc(stack_name(c))}, {who} ({r} {f3(v)}, mean {f3(MEAN[c][st])})"
        if c.startswith("e_"):
            cap += "; the sim's figure shows every manager of that run, the GNN punisher is gnn_self"
        if d.exists() and d.read_bytes()[:3] == b"\xff\xd8\xff":
            EMBEDDED.append((r, fname, c, st))
            return f'<figure><img src="{b64(d)}" loading="lazy" alt="{esc(fname)} ({c} {st})"><figcaption>{cap}</figcaption></figure>'
        MISSING.append((r, fname, c, st))
        why = ("the source sim dir carries no visuals" if c == "d_kexo" else
               "no before figure exists: the source sims were scored with the 21-row suite and only rescored, not re-plotted")
        return (f'<figure><div style="border:1px dashed #e1e0d9;border-radius:6px;aspect-ratio:14/9;display:grid;place-items:center;'
                f'color:#898781;font-size:12px;text-align:center;padding:8px">no figure &mdash; {why}</div><figcaption>{cap}</figcaption></figure>')
    btns, cards = [], []
    for i, r in enumerate(rows):
        col = SLOT[ROWSLOT[r]]; on = i == 0
        style = f"border-color:{col};color:{PAPER if on else col};background:{col if on else 'none'}"
        btns.append(f'<button data-m="{r}" data-color="{col}" class="{"on" if on else ""}" style="{style}">{r}</button>')
        body = "".join(f'<div class="barow">' + "".join(figure(r, f, c, "before") for c in BA_STACKS)
                       + "".join(figure(r, f, c, "after") for c in BA_STACKS) + "</div>" for f in fig_files(r))
        cards.append(f'<div class="bacard{" on" if on else ""}" id="ba-{r}">{body}</div>')
    return '<div class="mnav">' + "\n".join(btns) + "</div>", "\n".join(cards)


# ---------------------------------------------------------------- 6. machinery
def machinery_svg():
    o = [f'<svg viewBox="0 0 1200 620" role="img" font-family="system-ui, sans-serif" aria-label="One round of the simulation loop with the punisher input fixed, drawn as a three-bay machine.">'
         '<defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 z" fill="currentColor"/></marker></defs>']
    def bay(x, title, sub=None):
        o.append(f'<rect x="{x}" y="60" width="360" height="500" rx="10" fill="none" stroke="currentColor" stroke-opacity="0.35"/>'
                 f'<text x="{x+180}" y="84" text-anchor="middle" font-size="13" font-weight="600" fill="currentColor">{title}</text>')
        if sub: o.append(f'<text x="{x+180}" y="99" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.7">{sub}</text>')
    def box(x, y, t1, t2, tint=None, title=None, pill=None):
        cx = x + 140
        fill = f'fill="{tint}" fill-opacity="0.14" stroke="{tint}"' if tint else 'fill="none" stroke="currentColor" stroke-opacity="0.6"'
        o.append("<g>" + (f"<title>{esc(title)}</title>" if title else "") +
                 f'<rect x="{x}" y="{y}" width="280" height="62" rx="6" {fill}/>'
                 f'<text x="{cx}" y="{y+25}" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">{t1}</text>'
                 f'<text x="{cx}" y="{y+42}" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.75">{t2}</text>')
        if pill:
            text, story = pill
            w = 38 if len(text) <= 4 else 46
            o.append(f'<a href="#story-{story}"><title>the story behind {esc(text)}</title>'
                     f'<rect x="{x+280-44-(w-38)}" y="{y-8}" width="{w}" height="17" rx="8" fill="#1a1a19"/>'
                     f'<text x="{x+280-44-(w-38)+w/2}" y="{y+4.5}" fill="{PAPER}" text-anchor="middle" font-size="10">{esc(text)}</text></a>')
        o.append("</g>")
    def arrow_(cx, y1, y2, label=None, both=False):
        ms = ' marker-start="url(#arr)"' if both else ""
        o.append(f'<line x1="{cx}" y1="{y1}" x2="{cx}" y2="{y2}" stroke="currentColor" marker-end="url(#arr)"{ms}/>')
        if label: o.append(f'<text x="{cx+8}" y="{(y1+y2)/2+4}" font-size="10" fill="currentColor" opacity="0.75">{label}</text>')
    # bay 1: players
    bay(30, "PLAYERS BAY", "contributor and switch models, unchanged")
    box(70, 108, "contribution trunk", "gnn (#179 vnode, #181 skip) or gaussian-MLP (#174, #177)")
    arrow_(210, 170, 208, "per-agent marginals")
    box(70, 212, "group copula unit", "herding latent on the draws (stock, as accepted)")
    arrow_(210, 274, 312)
    box(70, 316, "switch model", "joint exodus head, fires every 4th round (stock)")
    arrow_(210, 378, 470)
    o.append(f'<text x="218" y="430" font-size="10.5" fill="currentColor">contributions c_t (0..20)</text><circle cx="210" cy="475" r="3.5" fill="currentColor"/>')
    # bay 2: punisher
    bay(420, "PUNISHER BAY", "the manager: one joint decision per group-round")
    box(460, 108, "feature intake -- fixed", "contribution c_t (new) - prev contribution - prev punishment - round",
        FIX, f"PR #184: the punisher reads the contribution it punishes. Before: prev contribution only, one round late. CV log loss lin {CV['lin'][0]:.4f} -> {CV['lin'][1]:.4f}, GNN {CV['gnn'][0]:.4f} -> {CV['gnn'][1]:.4f}.",
        ("#184", "punisher"))
    o.append(f'<text x="600" y="184" text-anchor="middle" font-size="9.5" fill="currentColor" opacity="0.6" text-decoration="line-through">was: prev contribution (round t-1) only</text>')
    arrow_(600, 192, 208)
    box(460, 212, "multinomial logistic trunk / GNN punisher", f"31 levels - test log loss {CV['lin_test'][0]:.3f} -> {CV['lin_test'][1]:.3f} (floor {CV['floor']:.3f})")
    arrow_(600, 274, 312, "per-agent multinomial CDFs")
    box(460, 316, "severity copula unit", f"shared group-round latent - rho {RHO['before']:.3f} -> {RHO['after']:.3f}",
        SLOT["contribution"], f"PR #184 re-stamps the #160 copula on the new bundle: rho {RHO['before']:.4f} -> {RHO['after']:.4f} (SE {RHO['se']:.4f}); it rose instead of dropping.",
        ("#184", "copula"))
    arrow_(600, 378, 470)
    o.append(f'<text x="608" y="430" font-size="10.5" fill="currentColor">punishments p_t (0..30)</text><circle cx="600" cy="475" r="3.5" fill="currentColor"/>')
    # bay 3: scoring
    bay(810, "SCORING BAY", "the 22-row evaluation suite, 500 repeats, seed 42")
    box(850, 108, "RCE: punishment response slope", "protected row: no sign flip, no halving, no band downgrade",
        "#4a3aa7", f"RCE: OLS slope of next-round change on punishment received, per contribution band; human {' / '.join(sgn(v) for v in HUMAN_SLOPES)}; ceiling {CEIL['RCE']:.4f}.",
        ("RCE", "rce"))
    arrow_(990, 170, 208, "beside RCB, not instead")
    box(850, 212, "held-out teacher-forced test", f"5 folds: held-out {HO['pooled_held_out']:.3f} vs in-sample {HO['pooled_in_sample']:.3f}",
        "#e87ba4", f"PR #183: the contributor's reaction to punishment is learned, not memorised (pooled held-out {HO['pooled_held_out']:.4f}, in-sample {HO['pooled_in_sample']:.4f}, self-play {PR179_SELFPLAY_RCB:.3f}).",
        ("#183", "holdout"))
    arrow_(990, 274, 312, "so the closed loop is the culprit")
    box(850, 316, "re-baseline ledger", "six runs, 22 rows, before -> after; gates suspended",
        "#6b6a66", "PR #184 stage D: every frontier stack rerun with the fixed punisher; the ledger's baselines reset.",
        ("#184", "rebaseline"))
    arrow_(990, 378, 470)
    o.append(f'<text x="998" y="430" font-size="10.5" fill="currentColor">scores.csv (22 rows)</text><circle cx="990" cy="475" r="3.5" fill="currentColor"/>')
    # round loop
    o.append('<line x1="30" y1="475" x2="1170" y2="475" stroke="currentColor" stroke-width="1.6" marker-end="url(#arr)"/>'
             '<text x="600" y="497" text-anchor="middle" font-size="11" fill="currentColor">round t: contributions c_t -&gt; the manager sees c_t and punishes -&gt; common good = 1.6 &#215; sum c_t &#8722; sum p_t -&gt; payoffs</text>'
             '<path d="M 1170 522 L 30 522" stroke="currentColor" stroke-dasharray="6 5" stroke-opacity="0.6" fill="none" marker-end="url(#arr)"/>'
             '<text x="600" y="542" text-anchor="middle" font-size="10.5" fill="currentColor" opacity="0.75">feeds round t+1 as prev contribution / prev punishment (24 rounds x 100 episodes, seed 42); every 4th round the switch bay regroups</text>'
             '<text x="600" y="600" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.6">the players\' models are the ones their PRs shipped; only the manager\'s input changed</text></svg>')
    return "".join(o)


# ---------------------------------------------------------------- 7. leaderboard
def lb_rows():
    return [dict(pr=CASE[c][2], label=CASE[c][3], slot="punisher", stack=CASE[c][1], note=CASE[c][4], **ST[c]) for c in ORDER]


# ---------------------------------------------------------------- 8. stories
def rng(vals, d=2):
    lo, hi = min(vals), max(vals)
    return f"[{lo:+.{d}f}, {hi:+.{d}f}]" if lo < 0 else f"{lo:.{d}f}-{hi:.{d}f}"


def pct(vals):
    return f"{100 * min(vals):.0f}-{100 * max(vals):.0f}%"


def story(id_, title, chip, color, meta, problem, change, maths, code, bought):
    return (f'<article class="story" id="story-{id_}">\n<h3>{title}</h3>\n'
            f'<p class="meta"><span class="chip" style="background:{color}">{chip}</span> {meta}</p>\n'
            f'<h4>The problem</h4><p>{problem}</p>\n<h4>The change</h4><p>{change}</p>\n'
            f'<h4>The maths, in plain English</h4><ol class="maths">{"".join(f"<li>{m}</li>" for m in maths)}</ol>\n'
            f'<h4>Where it lives in the code</h4><ul class="code">{"".join(f"<li>{c}</li>" for c in code)}</ul>\n'
            f'<p class="meta">(paths as changed on the PR)</p>\n<h4>What it bought</h4><p>{bought}</p>\n</article>')


def stories():
    hm = mech.loc["human"]
    mb = lambda c, k: float(mech.loc[f"{c} before", k]); ma = lambda c, k: float(mech.loc[f"{c} after", k])
    ols_b = [mb(c, "OLS c_t") for c in ORDER]; ols_a = [ma(c, "OLS c_t") for c in ORDER]
    lag_b = [mb(c, "OLS c_t-1") for c in ORDER]; lag_a = [ma(c, "OLS c_t-1") for c in ORDER]
    p20_b = [mb(c, "P(p>0|c_t=20)") for c in ORDER]; p20_a = [ma(c, "P(p>0|c_t=20)") for c in ORDER]
    rpa_b = [float(S[c]["before"]["RPA"]) for c in ORDER]; rpa_a = [float(S[c]["after"]["RPA"]) for c in ORDER]
    rcb_d = [float(S[c]["after"]["RCB"] - S[c]["before"]["RCB"]) for c in ORDER]
    rcc_d = {c: float(S[c]["after"]["RCC"] - S[c]["before"]["RCC"]) for c in ORDER}
    pd_d = {c: float(S[c]["after"]["PD"] - S[c]["before"]["PD"]) for c in ("a_vnode", "b_skip", "c_infl", "d_kexo")}
    signs = lambda c, st: bands.loc[(c, st), "signs_vs_human"]
    slopes = lambda c, st: " / ".join(sgn(float(bands.loc[(c, st), f"slope_{b}"])) for b in BANDS)
    stacks_pr = ", ".join(pr_link(n) for n in (179, 181, 177, 174))
    out = []
    out.append(story("punisher", "The current-contribution punisher: punishing this round, not last round",
        "correctness", FIX, f"installed by PR {pr_link(184)} &middot; punisher bay of both machines, linear and GNN family",
        f"In the human games the manager sees this round's contributions and punishes them in the same round: the recorded common good equals 1.6 &times; this round's contributions minus this round's punishments in every valid row. "
        f"Fed the human data, punishment loads on the current contribution ({sgn(float(hm['OLS c_t']))} per point) and barely on the previous one ({sgn(float(hm['OLS c_t-1']))}); a player who just dropped from 20 to 4 or less is punished {float(hm['P(p>0|c_t<=4,c_t-1=20)']):.0%} of the time, one who just rose from 4 or less to 20 only {float(hm['P(p>0|c_t=20,c_t-1<=4)']):.0%}. "
        f"Both artificial punishers decided round t's punishment from round t&minus;1's contribution. In self-play the weight on the current contribution was {rng(ols_b, 3)} (about zero) and the timing check ran the wrong way: for the #181 stack {mb('b_skip', 'P(p>0|c_t=20,c_t-1<=4)'):.2f} for the riser against {mb('b_skip', 'P(p>0|c_t<=4,c_t-1=20)'):.2f} for the dropper. Full contributors were punished {pct(p20_b)} of the time against the human {100 * float(hm['P(p>0|c_t=20)']):.1f}%.",
        f"Round t's contribution is admitted as an input for the punishment target in both families (the feature-legality rule used to hard-error on it); prev contribution, prev punishment, round number and is-first stay; same-round punishment, payoff and common good remain illegal because they contain the answer. Both punishers are retrained with the new feature: cross-validated log loss {CV['lin'][0]:.4f} &rarr; {CV['lin'][1]:.4f} for the multinomial (locked test {CV['lin_test'][0]:.4f} &rarr; {CV['lin_test'][1]:.4f}, floor {CV['floor']:.4f}) and {CV['gnn'][0]:.4f} &rarr; {CV['gnn'][1]:.4f} for the GNN. The env's ordering (contribute, then punish, then copy into the prev slots) was already right; only the input list changed.",
        ["The lag was not an off-by-one bug: training and simulation were consistently lagged, a faithful model of the wrong mechanism, inherited from the contributor's leak rule (where reading the previous round is correct, because a player cannot see this round's punishment before contributing).",
         "The fitted weights say where the response went: on the 'no punishment' logit the standardized coefficient on the current contribution is +1.165 against &minus;0.339 on the previous one; in the old bundle the previous contribution carried +0.411. The current contribution now carries the whole 'gave more &rarr; punished less' response and the lag flips to a small opposite-sign residual, the same pattern as the human regression.",
         f"Closed loop, over the six runs: the OLS weight on the current contribution goes from {rng(ols_b, 3)} to {rng(ols_a, 3)} (human {sgn(float(hm['OLS c_t']))}), the weight on the previous one from {rng(lag_b, 3)} to {rng(lag_a, 3)} (human {sgn(float(hm['OLS c_t-1']))}), and the timing check flips to the human ordering in every run.",
         f"What stays short: full contributors are still punished {pct(p20_a)} of the time against {100 * float(hm['P(p>0|c_t=20)']):.1f}%, and when they are, lightly ({rng([ma(c, 'E[p|p>0] 20') for c in ORDER], 1)} points against {float(hm['E[p|p>0] 20']):.1f}). A logit that is linear in the contribution cannot produce the human step at 20; that is a functional-form limit, not a timing one."],
        ["<code>scripts/baselines/handcrafted_grid.py</code> &mdash; PUNISHMENT_LEGAL_CURRENT / illegal_current_features: the legal set that now admits contribution for the punishment target",
         "<code>src/aimanager/simulation/linear_ah.py</code> &mdash; get_punishments reads round t's contribution (it was at the last index, never read); the load-time legality assert",
         "<code>src/aimanager/manager/api_manager.py</code> &mdash; create_data: the same timing on the GNN path",
         "<code>configs/training/baselines/punishment/multinomial_current_contr.yml</code>, <code>configs/training/artificial_humans/punishment/rnn_edge_50ep_doubled_current_contr.yml</code> &mdash; the two retraining configs",
         "<code>src/aimanager/tests/test_punisher_current_contribution.py</code> &mdash; round t's contribution reaches the punisher at round t on both paths"],
        f"RPA, the manager's policy row, {rng(rpa_b)} &rarr; {rng(rpa_a)}: from the 1-2 band to at or under the noise ceiling in all six runs, a row the lag had fabricated outright. RCB falls by {abs(max(rcb_d)):.2f} to {abs(min(rcb_d)):.2f} everywhere (#179 {arrow('a_vnode', 'RCB')}, #181 {arrow('b_skip', 'RCB')}: the band PR #181 missed by 4.34% is cleared by the punisher fix alone). RCC does not move ({', '.join(f'{sgn(rcc_d[c], 2)} in {CASE[c][5]}' for c in ORDER)}): the full contributors still being punished are exactly the population that row is made of."))
    out.append(story("copula", "The severity copula, recalibrated: the manager's shared mood got stronger",
        "correlated-sampling", SLOT["contribution"], f"re-stamped by PR {pr_link(184)} &middot; the unit installed by PR {pr_link(160)} &middot; punisher bay of the four PR stacks (#179, #181, #177, #174)",
        f"The severity copula gives every group-round one shared random number, mixed with a little per-member noise, and translates each member's mixed number back through their own predicted punishment distribution: individual histograms are untouched, only the togetherness of the draws changes. Its mixing weight rho = {RHO['before']:.3f} was estimated on the lagged model's residuals. When the marginal changes, the co-movement it leaves unexplained changes with it, so rho had to be re-estimated before the fixed punisher could be stamped. The prediction was that rho would drop: a shared current contribution should explain part of why a group is punished together.",
        f"The estimator (pairwise maximum likelihood on the within-cell pairs, 15,291 of them) was run on the new bundle with the new <code>--bundle</code> / <code>--out</code> flags: rho = {RHO['after']:.4f}, SE {RHO['se']:.4f}, 95% CI [{RHO['ci_after'][0]:.3f}, {RHO['ci_after'][1]:.3f}], against the old {RHO['before']:.4f} [{RHO['ci_before'][0]:.3f}, {RHO['ci_before'][1]:.3f}]. The round-trip gate passed (max |bias| 0.013, tolerance 0.03) and the stamped copy is weight-identical to the plain bundle plus the eight copula keys.",
        ["Each member's punishment is drawn by rolling one number u between 0 and 1 and walking up their 31 predicted probabilities until the running total passes u. The unit only changes where u comes from: u = the percentile of sqrt(rho) &times; g + sqrt(1 &minus; rho) &times; e, with g one bell-curve draw shared by the group-round and e the member's own.",
         f"Why rho rose rather than fell: conditioning each marginal on the current contribution sharpens it, so the co-movement left over is less attenuated by noise, and what remains is a per-round severity level the marginal cannot absorb -- a manager's mood, not a shared stimulus. The prediction that it would drop was wrong in the predicted direction, so the copy is stamped as calibrated, not as a smaller correction.",
         f"At the row level the stronger rho buys and costs nothing visible: PD, the row the copula targets, moves {', '.join(f'{sgn(pd_d[c], 2)} in {CASE[c][5]}' for c in pd_d)} -- no consistent direction."],
        ["<code>scripts/baselines/punishment_copula_rho.py</code> &mdash; the pairwise MLE with the round-trip gate; --bundle / --out are new so the stamped copy can carry its own name",
         "<code>artifacts/baselines/punishment_multinomial_current_contr_severity_copula.joblib</code> &mdash; the stamped bundle the four PR stacks run",
         "<code>src/aimanager/simulation/linear_ah.py</code> &mdash; the sampler reads copula_rho off the bundle; rho = 0 restores independent draws bit-for-bit"],
        f"A calibrated punisher for the PR stacks, and a question for a successor: whether the mood is a manager-level latent (one per episode) or a round-level one is testable on the human data with the same script and matters for PD (in the main-sweep stacks without the copula it still reads {arrow('e_lin', 'PD')} and {arrow('e_gnn', 'PD')})."))
    rce_after = "; ".join(f"{CASE[c][5]} {f3(float(S[c]['after']['RCE']))}, signs {signs(c, 'after')}" for c in ORDER)
    ex = dict(gave=3, pun=5, nxt=8)
    rate = ex["pun"] / (20 - ex["gave"])
    out.append(story("rce", "The RCE row: the punishment response slope, now protected",
        "evaluation", "#4a3aa7", f"branch rcb-alternative-response-slope, merged by PR {pr_link(184)} &middot; row 22 of the evaluation suite &middot; <code>reports/rcb_alternative_comparison.md</code>",
        f"The learning manager's only lever is punishment, so what the simulated players must get right is how they respond to it. RCB was the row scoring that: it sorts punished players by punishment <em>rate</em> (punishment divided by the shortfall from 20) and compares the average next-round change in each rate bin with the human one. The rate mixes how much a player gave with how hard they were hit: a rate above 1 is reached by a zero contributor punished 20 and by a 17 contributor punished 4, who react in opposite directions. So a stack can match the bin averages with the right mix of players and no within-level response at all. Across the 40 stacks of the sweep and the PRs, RCB and RCE rank the stacks almost independently (Spearman {SPEARMAN:.2f}); RCE tracks how many of the four human response signs a stack reproduces ({SPEAR_SIGNS['RCE']:.2f}), RCB does not ({SPEAR_SIGNS['RCB']:.2f}).",
        f"RCE takes the same punished non-full contributors, sorts them by what they gave (0-4, 5-9, 10-14, 15-19) and within each band fits a straight line of next-round change on punishment received, keeping the slope: how many more points a player gives per extra point of punishment. Humans comply at low levels and withdraw at high ones: {' / '.join(sgn(v) for v in HUMAN_SLOPES)}. The score is the human-frequency-weighted mean absolute slope gap over the four bands, divided by the human-vs-human ceiling of {CEIL['RCE']:.4f}. It sits beside RCB, not instead of it, and is the first protected row of the protocol.",
        [f"A worked example. A player gave {ex['gave']}, was punished {ex['pun']}, and gave {ex['nxt']} next round: a change of +{ex['nxt'] - ex['gave']}. RCB computes the rate {ex['pun']} / {20 - ex['gave']} = {rate:.2f}, files the +{ex['nxt'] - ex['gave']} into the (0.25, 0.5] bin, and compares that bin's average with the human one. RCE files the same player into the 0-4 band, where the +{ex['nxt'] - ex['gave']} is one point on the regression of change against punishment; only the slope of that line is scored.",
         f"Power. Two halves of the human data differ by {CEIL['RCE']:.3f} in slope, about three quarters of the human slopes themselves, so a simulation whose players ignore punishment scores {SYNTH['none']:.2f} ('minor deviation') and one with half the human response scores {SYNTH['half']:.2f}, at the ceiling. RCB's ceiling is {CEIL['RCB']:.3f}, but its sensitivity is to the level of the change among the punished, not to the dose. That is why RCE is protected on the statistic as well as on the band.",
         "The protection rule (notes/autoresearch.md &sect;2): an experiment may not band-downgrade RCE against its baseline, may not flip any of the four band slopes away from the human sign, and may not halve any band's slope magnitude. Any of the three is a FAIL whatever the gates say; punisher-slot and punishment-response experiments are judged on RCE for their band upgrade.",
         "The unpunished are left out on purpose: in every band their next-round change sits below the regression line of the punished (a step at zero dose, the extensive margin), so pooling them would blend that step into the dose response and flip the 10-14 sign. The two-band variant (0-9 / 10-19, ceiling 0.060) keeps the ranking with usable band edges if RCE is ever to gate alone."],
        ["<code>src/aimanager/evaluation_suite/metrics.py</code> &mdash; ResponseMetrics.rce, rce_weights, _rce_fit (slope, SE, n per band), RCE_EDGES / RCE_LABELS",
         "<code>src/aimanager/evaluation_suite/visuals.py</code> &mdash; the RCE_line figure (human vs sim slopes, &plusmn;1 SE)",
         "<code>notes/evaluation_metric_defs.md</code> &mdash; the row's definition; <code>notes/autoresearch.md</code> &sect;2 &mdash; the protected-row rule, 21 &rarr; 22 rows",
         "<code>src/aimanager/tests/test_eval_metrics.py</code> &mdash; the row's tests; the RCF cell variant was dropped again on the re-baseline branch"],
        f"Under the fixed punisher: {rce_after}. The #181 skip stack and both Gaussian-MLP stacks keep all four human signs with every band moving toward the human magnitude (#181's 15-19 band {sgn(float(bands.loc[('b_skip', 'before'), 'slope_15-19']))} &rarr; {sgn(float(bands.loc[('b_skip', 'after'), 'slope_15-19']))} against the human {sgn(HUMAN_SLOPES[3])}); #179 and the GNN-punisher run lose two signs each through bands whose |slope| was under 0.04 before and after."))
    out.append(story("holdout", "Learned, not memorised: the held-out teacher-forced test",
        "diagnosis", "#e87ba4", f"PR {pr_link(183)} &middot; branch rcb-holdout-teacher-forced &middot; 5 folds of 10 games, Raven A100 &middot; on the #179 group-vnode contributor",
        f"A simulated player's reaction to punishment can be read in two ways. <em>Teacher-forced</em>: feed the network the real human history round by round and read off only its predicted next contribution. <em>Self-play</em> (closed loop): let the models generate the whole game themselves. PR #181 reported that the #179 contributor reacts almost like humans teacher-forced (raw RCB gap {HO['full_in_sample']:.3f}, inside the {CEIL['RCB']:.3f} ceiling) but not in self-play ({PR179_SELFPLAY_RCB:.3f}), and read the gap as drift of the simulated game state. But the {HO['full_in_sample']:.3f} came from a model trained on all 50 games: a network with memory, tested on games it has memorised, matches averages almost by construction. The flat self-play result was equally consistent with a reaction that never generalised.",
        f"Retrain the contributor five times, each time with 10 games held out, and teacher-force every game through the copy that never saw it. Pooled over the 50 games, the held-out RCB statistic is {HO['pooled_held_out']:.3f}, the pooled in-sample one {HO['pooled_in_sample']:.3f}, the shipped model {HO['full_in_sample']:.3f}: within 0.013 of each other, all inside the {CEIL['RCB']:.3f} ceiling, eight times below self-play. The reaction is learned; it only goes flat when the players play against the simulated manager.",
        [f"Per fold the held-out statistic swings from {HO_FOLD_RANGE[0]:.3f} to {HO_FOLD_RANGE[1]:.3f}, but so does the human yardstick on the same 500-odd rows ({HO_HUM_RANGE[0]:.3f} to {HO_HUM_RANGE[1]:.3f}): ten games are noisy, and the pooled number is the honest one.",
         f"Two secondary deficits are real and are not closed-loop: the 10-14 band has the wrong sign under every condition (pooled held-out {sgn(HO_SLOPES['pooled_held_out'][2])}, in-sample {sgn(HO_SLOPES['pooled_in_sample'][2])}, human {sgn(HUMAN_SLOPES[2])}: never learned), and the 15-19 withdrawal is weaker held-out than in-sample ({sgn(HO_SLOPES['pooled_held_out'][3])} vs {sgn(HO_SLOPES['pooled_in_sample'][3])}, human {sgn(HUMAN_SLOPES[3])}: partly memorised).",
         "So the fixes worth pursuing act on the closed loop: a simulated manager whose punishments resemble the human ones (the next card), and the stimulus skip. Work on input features or regularisation can only address the two secondary bands."],
        ["<code>scripts/data_analysis/rcb_holdout_teacher_forced.py</code> &mdash; the fold harness: retrain, teacher-force the held-out games, pool",
         "<code>configs/training/artificial_humans/contribution/group_switching_contribution_50ep_group_vnode_holdout_folds.yml</code> &mdash; the five fold configs",
         "<code>plots/data_analysis/rcb_holdout_teacher_forced/group_vnode_raven_gpu.csv</code> &mdash; the per-fold and pooled statistics quoted here (CPU cross-check alongside)",
         "<code>notes/autoresearch_log/rcb-holdout-teacher-forced.md</code> &mdash; the log"],
        "It removed the wrong branch of the decision tree before any modelling effort went there: the closed loop, not the contributor's memory, was where the response died, which is what pointed the investigation at the manager's input and produced PR #184. It also left a reusable held-out-fold harness."))
    best = min(ORDER, key=lambda c: MEAN[c]["after"])
    egnn_sev = " / ".join(f"{ma('e_gnn', 'E[p|p>0] ' + b):.1f}" for b in BANDS + ["20"])
    out.append(story("rebaseline", "The re-baseline: what moved, what it cost, and how to read the lineage",
        "re-baseline", "#6b6a66", f"PR {pr_link(184)}, stage D &middot; the stacks of PRs {stacks_pr} and the main sweep's gnn x gnn stack (two punishers) &middot; ledger reset in <code>notes/autoresearch.md</code> &sect;3",
        "Every accepted stack had been scored against a manager that punished last round, and the punisher artifact sits in every stack. The fix therefore moves every row of every stack at once, so the usual acceptance rule (a band upgrade on a declared row with the mean within 10%) cannot apply: nothing is competing against the ledger, the ledger itself is wrong. The honest artifact is a re-baseline: every frontier stack rerun under the fixed punisher, every row rescored, the ledger's baselines reset to the new numbers, and the 21-row scores recorded before it declared not comparable with the 22-row scores after it.",
        f"The five source configs were copied with only the punisher path changed and run with the 23-family protocol (seed 42, 100 episodes, 24 rounds, about two minutes each on one A100); the two Gaussian-MLP cases ran from a second checkout of the gmlp code tree with the three fix commits cherry-picked, because their contributor bundles unpickle classes that exist only there. Everything was evaluated with the merged 22-row suite (500 repeats, seed 42); the before column is the source sim rescored with the same suite, so the 21 old rows reproduce the committed values exactly and RCE is added.",
        [f"What moved as it should: RPA into the noise band in all six runs; RCB down {abs(max(rcb_d)):.2f}-{abs(min(rcb_d)):.2f} everywhere; RSA down in five of six; RCD {arrow('b_skip', 'RCD')} in #181 (2-5 &rarr; 1-2); RCA down in #177 and both main-sweep runs.",
         f"What got worse is systematic, not noise: the contribution marginals CA / CB / CD rise by about 0.2 in #179, #174 and the main-lin run (#179 loses three rows from its &lt;= 1 count: {LE1['a_vnode']['before']} &rarr; {LE1['a_vnode']['after']}); PA rises {min(float(S[c]['after']['PA'] - S[c]['before']['PA']) for c in ('a_vnode', 'c_infl', 'd_kexo')):.2f}-{max(float(S[c]['after']['PA'] - S[c]['before']['PA']) for c in ('a_vnode', 'c_infl', 'd_kexo')):.2f} in #179, #177, #174; SC rises {min(float(S[c]['after']['SC'] - S[c]['before']['SC']) for c in ('a_vnode', 'b_skip', 'c_infl')):.2f}-{max(float(S[c]['after']['SC'] - S[c]['before']['SC']) for c in ('a_vnode', 'b_skip', 'c_infl')):.2f} in #179, #181, #177. The contributor and switch models were not retrained (they learn from human data, not from the punisher), but the closed-loop states they reach shift under a manager that punishes low contributors harder and full ones less, and their baselines shift with them.",
         "Means and counts: " + "; ".join(f"{CASE[c][5]} {f3(MEAN[c]['before'])} &rarr; {f3(MEAN[c]['after'])}, rows &lt;= 1 {LE1[c]['before']} &rarr; {LE1[c]['after']}" for c in ORDER) + ".",
         f"The lineage reading: the #181 skip stack now holds the best mean on record ({f3(MEAN[best]['after'])}, {LE1[best]['after']} rows at or under the ceiling) with all four RCE signs; the Gaussian-MLP line keeps all four signs with the strongest response (#174 RCE {f3(float(S['d_kexo']['after']['RCE']))}); the #179 vnode line, on which the ledger sat, loses two RCE signs and three &lt;= 1 rows; the GNN-punisher run is the largest net mean gain ({f3(MEAN['e_gnn']['before'])} &rarr; {f3(MEAN['e_gnn']['after'])}) but has the weakest marginals. Successors of #179 and of the GNN-punisher run inherit the after column as their RCE baseline.",
         "Caveats: the 32-stack sweep matrix was not re-run, so the ranking rule of &sect;3 stays defined on the pre-fix matrix until the maintainer refreshes it; the copula rho rose (the previous card); the sign flips in #179 and the GNN-punisher run are bands with |slope| under 0.04 before and after."],
        ["<code>scripts/data_analysis/curpun_rebaseline.py</code> &mdash; writes rebaseline_table.{csv,md} and rce_bands.csv from the before and after scores",
         "<code>scripts/simulation/run_curpun_reruns.sh</code> &mdash; submit / fetch / evaluate for the five _curpun configs, with the gmlp-lineage checkout for cases c and d",
         "<code>configs/simulation/manager_testing/*_curpun.yml</code> &mdash; the five rerun configs, byte-identical to their sources except for the punisher path and output dir",
         "<code>plots/data_analysis/evaluation/punisher_current_contr/</code> &mdash; the tables this page is built from, plus before/&lt;case&gt;/scores.csv and mechanism_selfplay.csv",
         "<code>plots/simulation/&lt;source&gt;_curpun/</code> &mdash; the five sim dirs with per_round.parquet and the 22-row evaluations"],
        "A ledger whose baselines are the closed-loop states of the accepted contributor and switch models under a manager that punishes what it sees; a frontier ranking under those baselines; and a first protected row from which every successor is judged."))
    out.append(story("successor", "What this leaves for a successor",
        "successor", "#898781", "from the log's successor section &middot; ordered as the log lists them &middot; none of the threads is blocked",
        "Six runs, one fix, no gate: the re-baseline answers where the frontier stands under the fixed manager, and leaves the questions it raised open on purpose, because each is an experiment of its own rather than a stage of this one.",
        "None yet. The threads below are the candidates, with the row each one targets.",
        [f"Retrain or re-evaluate the frontier contributors against the fixed punisher. The contribution marginals and SC in #179, #181, #177 moved because the closed-loop states moved; #179's vnode and #181's skip were chosen in a world with a lagged punisher. First check whether their copula recalibrations (rho_p 0.044 / 0.039) still hold, and whether the skip's CG cost ({arrow('b_skip', 'CG')}) is worth its RCB gain now that the punisher fix alone clears RCB's band.",
         f"The P(punished | gave 20) residual ({pct(p20_a)} vs {100 * float(hm['P(p>0|c_t=20)']):.1f}%) is a functional-form limit shared by both punisher families and the reason RCC does not move. A punisher that separates 'whether' (a full-contributor indicator or a hinge at 20) from 'how much' is the obvious next punisher-slot experiment; its target rows are RCC and RPA at 20.",
         "The copula rho question: rho rose with the current contribution in the marginal, against the prediction. Whether the residual mood is a manager-level latent (one per episode) or a round-level one is testable on the human data with the copula scripts, and matters for PD.",
         f"The GNN punisher is now competitive: mean {f3(MEAN['e_gnn']['after'])} vs {f3(MEAN['e_lin']['after'])} for the linear one on the same stack, but {LE1['e_gnn']['after']} vs {LE1['e_lin']['after']} rows at or under the ceiling (PA / PB / RPB worse, CG / SC / RCA / RSA better). A sweep row for it inside the frontier stacks would tell whether its steeper response is worth its worse marginals.",
         f"RCE in the GNN-punisher run reads {signs('e_gnn', 'after')} against the human ++--. Whether the contributor's withdrawal at high contribution is masked by that punisher's flat severity profile ({egnn_sev} across the bands) is a one-run question.",
         "Rerun the 32-stack sweep under the fixed manager so the ledger's deficit profiles and ranking are all post-fix."],
        ["<code>notes/autoresearch_log/punisher-current-contribution.md</code> &mdash; the successor section and the caveats",
         "<code>notes/autoresearch.md</code> &sect;2-3 &mdash; the protected-row rule and the post-fix baselines a successor is judged against",
         "Raven clean-up once PR #184 closes: the isolated dirs under <code>~/repros/ai-runs/punisher-current-contr*</code>"],
        "A short, ordered list with a target row for each item, and a protocol that now judges punishment-response work on the mechanism (RCE) rather than on a composition row (RCB)."))
    return "\n\n".join(out)


# ---------------------------------------------------------------- page
def page():
    mnav, cards = ba_cards()
    rows_json = json.dumps(lb_rows())
    lb_script = SCRIPTS[1]
    assert "const ROWS = [" in lb_script
    lb_script = re.sub(r"const ROWS = \[.*?\];", lambda m: f"const ROWS = {rows_json};", lb_script, count=1, flags=re.S)
    best = min(ORDER, key=lambda c: MEAN[c]["after"])
    tree_legend = ("<p class=\"legend\">\n"
                   f'<span style="color:{SPINE["gnn"]}">&#9473;</span> gnn spine &nbsp; <span style="color:{SPINE["gmlp"]}">&#9473;</span> gaussian-MLP spine &nbsp; '
                   f'<span style="color:{FIX}">&#9473;</span> punisher fix (before &rarr; after) &nbsp; &#9675; before (lagged punisher) &nbsp; &#9679; after (current-contribution punisher)\n'
                   "&nbsp; (solid = lineage spine from the main-sweep stack, dashed = the same stack under the GNN punisher, dotted step = best after-mean so far; there are no failed attempts on this page) "
                   "&mdash; hover a node for its mean, rows &lt;= 1 and RCE before &rarr; after, click it for its story.\n</p>")
    return f"""<meta charset="utf-8">
<title>Rebaseline Atlas</title>
{STYLES[0]}
{STYLES[1]}
<div class="wrap">
<h1>Rebaseline Atlas</h1>
<p class="sub">The companion to the Autoresearch Atlas: the frontier stacks rerun after the simulated
manager was fixed to punish the current round's contribution (PR #184), scored on 22 rows including
the new, protected RCE row -- six runs, before and after, on two spines. Hover any node for its numbers;
click it for the plain-language story.</p>
<nav>
  <button class="on" data-layer="tree">Progress tree</button>
  <button data-layer="scores">All 22 scores</button>
  <button data-layer="breakdown">Score breakdown</button>
  <button data-layer="beforeafter">Before / after</button>
  <button data-layer="machinery">Machinery</button>
  <button data-layer="lb">Leaderboard</button>
  <button data-layer="stories">Stories</button>
</nav>
<section class="layer on" id="tree">

{stack_cards()}

<div class="treehead">
  <p class="legend">Focus a spine, or select both (nothing is hidden: this page has no failed
  attempts):</p>
  <div class="focus">
    <button data-key="gnn" data-color="{SPINE['gnn']}">gnn
    spine</button>
    <button data-key="gmlp" data-color="{SPINE['gmlp']}">gaussian-MLP
    spine</button>
  </div>
</div>
<div id="treebox">
{tree_svg()}
</div>

{tree_legend}

</section>
<section class="layer" id="scores">
<p class="legend">Every evaluation row for the six reruns, grouped by spine
(<span style="color:{SPINE['gnn']}">&#9473; gnn</span>: main-lin, main-gnn, #179, #181;
<span style="color:{SPINE['gmlp']}">&#9473; gaussian-MLP</span>: #174, #177); in each column the hollow
marker is the score before the punisher fix and the filled one after, joined by a short segment
(guides at the 1 / 2 / 5 band edges, log scale; titles colored by slot:
<span style="color:{SLOT['contribution']}">contribution</span> <span style="color:{SLOT['switch']}">switch</span> <span style="color:{SLOT['punisher']}">punisher</span>;
RCE is the protected row). Hover a pair for the numbers and the band change.</p>
<div class="grid21">{"".join(small_chart(r) for r in ROWS)}</div>
</section>
<section class="layer" id="breakdown">
<div class="treehead">
<p class="legend">All 22 rows per spine, colored by the slot each row
measures, across that spine's reruns: solid = after the punisher fix, dashed = before; the bold
line is the 22-row mean (always shown) &mdash; hover a line for its values, or focus one or more slots:</p>
  <div class="focus">
    <button data-key="contribution" data-color="{SLOT['contribution']}">contribution</button>    <button data-key="switch" data-color="{SLOT['switch']}">switch</button>    <button data-key="punisher" data-color="{SLOT['punisher']}">punisher</button>
  </div>
</div>
<div class="two" id="breakbox">{breakdown_svg("gnn")}
{breakdown_svg("gmlp")}
</div>
</section>
<section class="layer" id="beforeafter">
<p class="legend">The evaluation suite's own figure for each score row, before the punisher fix
(the source sim, lagged punisher) and after it (the _curpun rerun), for three stacks: #181 stimulus
skip, #174 k-one-hot Gaussian-MLP, and the main-sweep stack with the GNN punisher. Top line = before,
bottom line = after, one column per stack; rows with two figures show both. Pick a row.
(SA has no figure; its score is a single rate. The #174 source sim carries no visuals, and no
before figure exists for RCE, which was added to the suite after those sims were plotted.)</p>
{mnav}
{cards}
</section>
<section class="layer" id="machinery">

<p class="legend">
<span style="color:{FIX}">&#9632;</span> the fixed piece (punisher input) &nbsp;
<span style="color:{SLOT['contribution']}">&#9632;</span> recalibrated correlated-sampling unit &nbsp;
<span style="color:#4a3aa7">&#9632;</span> new protected row &nbsp;
<span style="color:#e87ba4">&#9632;</span> diagnostic &nbsp;
<span style="color:#6b6a66">&#9632;</span> ledger &nbsp;
&#9633; stock part &nbsp; &#11044;<small>#PR</small> installed by
&mdash; hover a tinted part for its story, <b>click a pill</b> for
the plain-language page.
</p>
<h2>One round of the simulation loop, with the punisher's input fixed (best rerun: {esc(CASE[best][5])}, mean {f3(MEAN[best]['after'])}, rows &lt;= 1: {LE1[best]['after']}/22)</h2>
<figure>
{machinery_svg()}
</figure>

</section>
<section class="layer" id="lb">
<p class="legend">The six reruns, each scored against its own before state (the same
contributor and switch models under the lagged punisher). Pick the ranking criterion:</p>
<div class="seg" id="lb-seg"></div>
<table>
  <thead><tr>
    <th></th><th>PR</th><th>Stack</th><th>Slot</th><th>Spine</th>
    <th class="num" data-c="d_mean">&Delta; mean</th>
    <th class="num" data-c="d_le1">&Delta; rows &le; 1</th>
    <th class="num" data-c="d_gt2">&Delta; rows &gt; 2</th>
    <th class="num" data-c="upgrades">Band upgrades</th>
  </tr></thead>
  <tbody id="lb-body"></tbody>
</table>
<p class="legend">&Delta; values are after &minus; before over the
22 rows; &Delta; rows &gt; 2 ranks reversed (fewer badly-missed rows is
better). Hover a band-upgrade count for the rows. The PR column is the stack's PR; the two
main-sweep runs carry the re-baseline PR #184. Ties break on &Delta; mean.</p>
</section>
<section class="layer" id="stories">
<p class="legend">The plain-language story of the re-baseline in six cards &mdash; also
reachable by clicking tree nodes, score markers and machinery pills.</p>

{stories()}
</section>
</div>
<div id="tip"></div>
{SCRIPTS[0]}
{lb_script}
"""


text = page()
if len(text.encode()) > SIZE_LIMIT:
    shrink_all()
    text = page()
    print("figures downscaled to fit the size limit")
OUT.write_text(text)
print(OUT, f"{OUT.stat().st_size / 1e6:.2f} MB")
print(f"embedded {len(EMBEDDED)} figures, missing {len(MISSING)}")
for k, v in sorted({(c, st): sum(1 for r, f, cc, s in MISSING if cc == c and s == st) for r, f, c, st in MISSING}.items()):
    print("  missing", k, v)
for c in ORDER:
    print(f"  {c:8s} d_mean {ST[c]['d_mean']:+.4f} d_le1 {ST[c]['d_le1']:+d} d_gt2 {ST[c]['d_gt2']:+d} upgrades {ST[c]['upgrades']} up [{ST[c]['up_rows']}] down [{ST[c]['down_rows']}]")
