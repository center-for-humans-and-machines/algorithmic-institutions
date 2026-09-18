import base64, csv, json, html, math, os
from pathlib import Path

D = Path("/Users/brinkmann/repros/algorithmic-institutions/.claude/worktrees/agent-a80d622b939db4c1c")
R = Path("/Users/brinkmann/repros/algorithmic-institutions/.claude/worktrees/agent-a91a63a720dcceb19")
OUT = Path("/private/tmp/claude-502/-Users-brinkmann-repros-algorithmic-institutions/0d1455de-e561-4680-a5f1-1b8fb90f1f18/scratchpad/atlas/punisher_rebaseline_atlas.html")
PC = D / "plots/data_analysis/evaluation/punisher_current_contr"
RA = R / "plots/data_analysis/evaluation/rcb_alternative"
PR = "https://github.com/center-for-humans-and-machines/algorithmic-institutions/pull/"

def esc(s): return html.escape(str(s), quote=True)
def b64(p):
    return "data:image/jpeg;base64," + base64.b64encode(Path(p).read_bytes()).decode()

# ---------- data ----------
CASES = [
    ("a_vnode", "a", "PR #179 group vnode", "graph-network players with a per-group memory node (the vnode), plus the contribution copula", 179),
    ("b_skip", "b", "PR #181 stimulus skip", "graph-network players with a direct path from punishment received to the output (the skip)", 181),
    ("c_infl", "c", "PR #177 inflated Gaussian-MLP", "Gaussian-MLP players with an inflated output spread", 177),
    ("d_kexo", "d", "PR #174 k-one-hot Gaussian-MLP", "Gaussian-MLP v2 players with the joint-exodus, k-one-hot switch model", 174),
    ("e_lin", "e1", "main branch, linear manager", "the main sweep's reference stack with the multinomial (linear) manager", None),
    ("e_gnn", "e2", "main branch, graph-network manager", "the main sweep's reference stack with the graph-network manager", None),
]
CASELBL = {c: f"{short} · {name}" for c, short, name, _, _ in CASES}
ROWS = ["CA","CB","CC","CD","CE","CF","CG","SA","SB","SC","PA","PB","PC","PD","RCA","RCB","RCC","RCD","RCE","RSA","RPA","RPB"]
ROWNAME = {"CA":"participant means","CB":"round means","CC":"group means","CD":"raw contributions","CE":"signed group differences","CF":"boundary shares","CG":"group-spread ratio","SA":"switch rate","SB":"switch timing","SC":"segregation","PA":"punishment levels","PB":"punishment by round","PC":"punished share","PD":"punishment spread ratio","RCA":"change by round type","RCB":"reaction to punishment (bins)","RCC":"reaction at the ceiling","RCD":"switching pull","RCE":"punishment response slope","RSA":"switching after punishment","RPA":"the manager's policy","RPB":"punishment by group size"}

scores = {}
with open(PC / "rebaseline_table.csv") as f:
    rd = csv.reader(f); hdr = next(rd)
    for row in rd:
        key = row[0]
        scores[key] = {h: (float(v) if v not in ("", "nan") else None) for h, v in zip(hdr[1:], row[1:])}
means = scores.get("mean"); le1 = scores.get("rows <= 1") or scores.get("rows_le1")
if means is None or le1 is None:
    # derive
    means = {}; le1 = {}
    for c,_,_,_,_ in CASES:
        for st in ("before","after"):
            vals = [scores[r][f"{c}_{st}"] for r in ROWS]
            means[f"{c}_{st}"] = sum(vals)/len(vals); le1[f"{c}_{st}"] = sum(1 for v in vals if v <= 1)

bands = {}
with open(PC / "rce_bands.csv") as f:
    for row in csv.DictReader(f):
        bands[(row["case"], row["stage"])] = row
HUMAN_SLOPES = [0.140, 0.104, -0.077, -0.161]
BANDS = ["0-4","5-9","10-14","15-19"]

mech = {}
with open(PC / "mechanism_selfplay.csv") as f:
    for row in csv.DictReader(f):
        mech[row["sim"]] = row

cmp_rows = []
with open(RA / "comparison_table.csv") as f:
    for row in csv.DictReader(f):
        cmp_rows.append(row)

def band_of(v):
    if v is None or (isinstance(v,float) and math.isnan(v)): return "na"
    if v <= 1: return "b1"
    if v <= 2: return "b2"
    if v <= 5: return "b3"
    return "b4"
BANDLBL = {"b1":"&le; 1","b2":"1&ndash;2","b3":"2&ndash;5","b4":"&gt; 5","na":"n/a"}

def fmt(v, d=2, sign=False):
    if v is None: return "&ndash;"
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return s.replace("-", "&minus;")

IMG = {
    "rcb_vs_rce": b64(RA / "RCB_vs_RCE_scores.jpg"),
    "rce_four": b64(RA / "RCE_human_vs_four_stacks.jpg"),
    "b_rce": b64(D / "plots/simulation/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_curpun/evaluation/visuals/RCE_line.jpg"),
    "b_rpa": b64(D / "plots/simulation/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_curpun/evaluation/visuals/RPA_line.jpg"),
    "d_rce": b64(D / "plots/simulation/23_2g8a_kexo_self_gaussian_mlp_v2_group_copula_contr_gnn_joint_exodus_k_onehot_switch_curpun/evaluation/visuals/RCE_line.jpg"),
    "a_rce": b64(D / "plots/simulation/23_2g8a_contr_group_vnode_self_gnncopar1_contr_gnn_switch_curpun/evaluation/visuals/RCE_line.jpg"),
}

# ---------- SVG: slope chart per case ----------
def slope_svg(case):
    b = bands[(case,"before")]; a = bands[(case,"after")]
    bs = [float(b[f"slope_{x}"]) for x in BANDS]; as_ = [float(a[f"slope_{x}"]) for x in BANDS]
    W, H = 640, 250; L, Rm, T, B = 46, 12, 18, 40
    ymax = 0.26; ymin = -0.26
    def y(v): return T + (ymax - v) / (ymax - ymin) * (H - T - B)
    gw = (W - L - Rm) / 4
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="RCE slopes per band, human vs before vs after">']
    for tv in (-0.2,-0.1,0,0.1,0.2):
        yy = y(tv)
        out.append(f'<line x1="{L}" x2="{W-Rm}" y1="{yy:.1f}" y2="{yy:.1f}" class="grid{" zero" if tv==0 else ""}"/>')
        out.append(f'<text x="{L-6}" y="{yy+4:.1f}" class="tick" text-anchor="end">{tv:+.1f}</text>')
    bw = gw / 4.4
    for i, band in enumerate(BANDS):
        x0 = L + i*gw + gw*0.12
        for j,(v,cls) in enumerate(((HUMAN_SLOPES[i],"hum"),(bs[i],"bef"),(as_[i],"aft"))):
            x = x0 + j*bw*1.1
            top = min(y(0), y(v)); h = abs(y(v)-y(0))
            out.append(f'<rect x="{x:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{max(h,0.8):.1f}" class="bar {cls}"><title>{band}: {cls} {v:+.3f}</title></rect>')
        out.append(f'<text x="{L + i*gw + gw/2:.1f}" y="{H-14}" class="tick" text-anchor="middle">gave {band}</text>')
    out.append('</svg>')
    return "\n".join(out)

# ---------- SVG: RCB vs RCE scatter ----------
def scatter_svg():
    W, H = 640, 400; L, Rm, T, B = 48, 16, 14, 44
    xs = [float(r["RCB_score"]) for r in cmp_rows]; ys = [float(r["RCE_score"]) for r in cmp_rows]
    xmin, xmax = 1.3, 2.9; ymin, ymax = 0.6, 1.85
    def X(v): return L + (v-xmin)/(xmax-xmin)*(W-L-Rm)
    def Y(v): return T + (ymax-v)/(ymax-ymin)*(H-T-B)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="RCB score against RCE score for 40 stacks">']
    for tv in (1.5,2.0,2.5):
        out.append(f'<line x1="{X(tv):.1f}" x2="{X(tv):.1f}" y1="{T}" y2="{H-B}" class="grid"/><text x="{X(tv):.1f}" y="{H-B+16}" class="tick" text-anchor="middle">{tv:.1f}</text>')
    for tv in (0.8,1.0,1.2,1.4,1.6,1.8):
        out.append(f'<line x1="{L}" x2="{W-Rm}" y1="{Y(tv):.1f}" y2="{Y(tv):.1f}" class="grid{" zero" if tv==1.0 else ""}"/><text x="{L-6}" y="{Y(tv)+4:.1f}" class="tick" text-anchor="end">{tv:.1f}</text>')
    out.append(f'<line x1="{X(2.0):.1f}" x2="{X(2.0):.1f}" y1="{T}" y2="{H-B}" class="grid zero"/>')
    def fam(name):
        n = name.lower()
        if n.startswith("pr"): return "pr"
        if " cat " in n or n.startswith("main cat"): return "cat"
        if "gaussian x" in n or n.startswith("main gaussian"): return "gau"
        if "ridge x" in n or n.startswith("main ridge"): return "rid"
        return "gnn"
    for r in cmp_rows:
        x, yv = float(r["RCB_score"]), float(r["RCE_score"]); f = fam(r["model"])
        sig = r["signs_match"]
        out.append(f'<circle cx="{X(x):.1f}" cy="{Y(yv):.1f}" r="{6 if f=="pr" else 4.5}" class="pt {f}"><title>{esc(r["model"])}: RCB {x:.2f}, RCE {yv:.2f}, {sig}/4 signs</title></circle>')
    out.append(f'<text x="{(L+W-Rm)/2:.0f}" y="{H-4}" class="tick" text-anchor="middle">RCB score (bins of punishment rate)</text>')
    out.append(f'<text transform="translate(12 {(T+H-B)/2:.0f}) rotate(-90)" class="tick" text-anchor="middle">RCE score (response slope)</text>')
    out.append('</svg>')
    return "\n".join(out)

# ---------- HTML pieces ----------
def case_buttons():
    return "".join(f'<button type="button" class="{"on" if i==1 else ""}" data-case="{c}" id="case-{c}">{esc(short)} <small>{esc(name)}</small></button>' for i,(c,short,name,_,_) in enumerate(CASES))

def score_table(c):
    rows = []
    for r in ROWS:
        bv, av, dv = scores[r][f"{c}_before"], scores[r][f"{c}_after"], scores[r][f"{c}_delta"]
        prot = ' class="prot"' if r == "RCE" else ""
        dcls = "neg" if dv is not None and dv < -0.05 else ("pos" if dv is not None and dv > 0.05 else "")
        rows.append(f'<tr{prot}><td class="row">{r}<span class="rn">{esc(ROWNAME[r])}</span></td>'
                    f'<td class="num"><span class="chip {band_of(bv)}">{fmt(bv)}</span></td>'
                    f'<td class="num"><span class="chip {band_of(av)}">{fmt(av)}</span></td>'
                    f'<td class="num {dcls}">{fmt(dv, sign=True)}</td>'
                    f'<td class="band">{BANDLBL[band_of(bv)]} &rarr; {BANDLBL[band_of(av)]}</td></tr>')
    mb, ma = means[f"{c}_before"], means[f"{c}_after"]; lb, la = le1[f"{c}_before"], le1[f"{c}_after"]
    rows.append(f'<tr class="tot"><td>mean over 22 rows</td><td class="num">{fmt(mb,3)}</td><td class="num">{fmt(ma,3)}</td><td class="num">{fmt(ma-mb,3,True)}</td><td></td></tr>')
    rows.append(f'<tr class="tot"><td>rows at or under the ceiling</td><td class="num">{int(lb)}</td><td class="num">{int(la)}</td><td class="num">{int(la-lb):+d}</td><td></td></tr>')
    return "\n".join(rows)

def score_cards():
    out = []
    for i,(c,short,name,desc,pr) in enumerate(CASES):
        out.append(f'<div class="bacard{" on" if i==1 else ""}" data-case="{c}">'
                   f'<p class="legend">{esc(name)} &middot; {esc(desc)}' + (f' &middot; <a href="{PR}{pr}" target="_blank" rel="noopener">PR #{pr}</a>' if pr else "") +
                   '. Before: the stack as originally accepted, with the simulated manager that punished last round. After: the same player and switch models, with the manager retrained on the current round.</p>'
                   f'<div class="twocol"><div class="tablewrap"><table class="scores"><thead><tr><th>row</th><th class="num">before</th><th class="num">after</th><th class="num">&Delta;</th><th>band</th></tr></thead><tbody>{score_table(c)}</tbody></table></div>'
                   f'<div><h4>RCE: response slope per contribution band</h4>{slope_svg(c)}<p class="legend">Each bar is the slope of next-round change on punishment received, for punished players who gave 0&ndash;4, 5&ndash;9, 10&ndash;14 or 15&ndash;19 points. Humans: {" / ".join(f"{v:+.3f}" for v in HUMAN_SLOPES)}, so low contributors give more when punished and high contributors give less. Slope signs matching the humans: before {esc(bands[(c,"before")]["signs_vs_human"])}, after {esc(bands[(c,"after")]["signs_vs_human"])}.</p>'
                   f'<div class="keyline"><span class="sw hum"></span>human <span class="sw bef"></span>before <span class="sw aft"></span>after</div></div></div></div>')
    return "\n".join(out)

def mech_table():
    order = ["human"] + [f"{c} {st}" for c,_,_,_,_ in CASES for st in ("before","after")]
    rows = []
    for k in order:
        m = mech[k]
        cls = "hum" if k=="human" else ("aft" if k.endswith("after") else "")
        if k == "human":
            label = "real human managers"
        else:
            case, stage = k.rsplit(" ", 1)
            label = f"{CASELBL[case]}, {stage}"
        rows.append(f'<tr class="{cls}"><td>{esc(label)}</td>'
                    f'<td class="num">{float(m["P(p>0|c_t=20)"]):.3f}</td><td class="num">{float(m["P(p>0|c_t<=4)"]):.3f}</td>'
                    f'<td class="num">{float(m["P(p>0|c_t=20,c_t-1<=4)"]):.2f} / {float(m["P(p>0|c_t<=4,c_t-1=20)"]):.2f}</td>'
                    '<td class="num">' + " / ".join("%.1f" % float(m["E[p|p>0] " + b]) for b in ("0-4","5-9","10-14","15-19","20")) + '</td>'
                    f'<td class="num">{float(m["OLS c_t"]):+.3f} / {float(m["OLS c_t-1"]):+.3f}</td></tr>'.replace("-0.", "&minus;0.").replace("+", "+"))
    return "\n".join(rows)

def overview_tiles():
    mb = means["b_skip_after"]
    return f'''
<div class="tiles">
  <div class="tile"><span class="k">RPA, how the manager punishes</span><span class="v">1.23&ndash;1.56 &rarr; 0.68&ndash;0.89</span><span class="s">all six stacks move from the 1&ndash;2 band to at or under the ceiling</span></div>
  <div class="tile"><span class="k">RCB, reaction to punishment</span><span class="v">&minus;0.5 to &minus;0.9</span><span class="s">better in every stack; two move from 2&ndash;5 into 1&ndash;2, one goes under 1</span></div>
  <div class="tile"><span class="k">best mean over the 22 rows</span><span class="v">{mb:.3f}</span><span class="s">PR #181 stimulus-skip stack, 13 rows at or under the ceiling</span></div>
  <div class="tile"><span class="k">RCE, response slope (protected)</span><span class="v">0.70&ndash;0.89</span><span class="s">the skip stack and both Gaussian-MLP stacks keep all four human response signs</span></div>
</div>'''

ABOUT = '''
<h2>What this is about</h2>
<div class="twocol">
<div>
<p class="legend" style="color:var(--ink)"><b>The game.</b> Eight people play 24 rounds in two groups of four. Each round every player receives 20 points and puts any share into the group's common pot; the pot is multiplied by 1.6 and split equally, so the group does best when everyone contributes and a free-rider does best of all. Each group has a manager who can punish individual players by up to 30 points, which costs the pot as well as the player. Every fourth round, players may switch to the other group. Fifty such games were played by real people.</p>
<p class="legend" style="color:var(--ink)"><b>The models.</b> The project trains three models on those games: simulated players that decide how much to contribute, a switch model that decides who changes groups, and a simulated manager that decides punishments. One combination of the three is a <em>stack</em>, and each model's place in it is a <em>slot</em>. The stacks matter because a learning manager will later be trained by playing against the simulated players. If they react to punishment differently from real people, it learns the wrong lessons.</p>
</div>
<div>
<p class="legend" style="color:var(--ink)"><b>What a score means.</b> Twenty-two rows each compare one statistic of the simulated games with the same statistic in the real games: how much people give, how often they switch, how they react to punishment, how the manager punishes. A score is a ratio: the simulation's distance from the human data, divided by how far two halves of the human data are from each other (the <em>noise ceiling</em>). At or under 1 means the simulation is as close to the humans as humans are to themselves; 1&ndash;2 is a minor deviation, 2&ndash;5 a clear one, above 5 the behaviour is not reproduced.</p>
<p class="legend" style="color:var(--ink)"><b>What happened here.</b> In one working day, four investigations traced a stubborn deficit in the reaction-to-punishment row to the simulated manager: on every branch of the code it punished the previous round's contribution, whereas real managers punish the current one. The manager was retrained, a sharper measure of the players' response (RCE) was added, and every stack was rescored under the fix.</p>
</div>
</div>'''

STEPS = [
 ("Two questions", "First: the <em>copula</em>, a shared random number that makes the members of one group act alike, fixes the group-spread rows far better than its small fitted strength predicts. Is that a real mechanism or a patch? Second: the row scoring how simulated players react to punishment (RCB) had been stuck above 2 in every stack. Why?", "the starting point"),
 ("Four investigations", "Read-only surveys of the code, the experiment record and the data: how the copula works and what justifies it; what RCB actually measures; whether the simulated manager punishes at the right time; and whether the compute cluster had a usable checkout (it did not).", "no models changed"),
 ("Held-out test", "Is the players' reaction to punishment learned, or memorised from the training games? Retrained five times with 10 games held out each time, the reaction on unseen games (0.095) matches the in-sample one (0.082). It is learned; it only goes flat when the players play against the simulated manager.", "PR #183"),
 ("A better instrument", "RCB compares bin averages and can be matched without the right cause and effect. A new row, RCE, measures how much more a player gives per extra point of punishment. Over 40 stacks the two rank the stacks almost independently (rank correlation 0.28); RCE tracks whether the human response signs are reproduced (&minus;0.88), RCB does not (&minus;0.08).", "branch rcb-alternative-response-slope"),
 ("Fix and retrain", "The simulated manager punished last round's contribution on every branch of the code. Both manager models were retrained on the current round, and their fit to the human data (cross-validated log loss) improves.", "branch auto/punisher-current-contribution"),
 ("Re-baseline", "Five stacks rerun under the fixed manager and all 22 rows rescored. The experiment record (the <em>ledger</em>) was reset to the new numbers, and RCE became the first <em>protected row</em>: one no future experiment may worsen.", "PR #184"),
]

def steps_html():
    return "".join(f'<li><span class="n">{i+1}</span><div><h3>{t}</h3><p>{d}</p><p class="meta">{m}</p></div></li>' for i,(t,d,m) in enumerate(STEPS))

STORIES = [
 dict(id="copula", chip="correlated-sampling", color="var(--c-blue)", title="The copula question: a variance source wearing a correlation's clothes",
  meta=f'PRs <a href="{PR}160">#160</a>, <a href="{PR}165">#165</a>, <a href="{PR}170">#170</a>, <a href="{PR}179">#179</a> &middot; the copula is used in both the manager and the player models',
  problem="Members of a real group act alike: they see the same situation and read it the same way. Simulated players drawn one at a time do not, so the rows that measure how far groups drift apart (CG, the group-spread ratio, and SC, segregation) scored badly in early stacks. The fix in use is the copula: one shared random number per group (the <em>latent</em>) is mixed into every member's draw, so members move together while each member's own probability distribution over choices (the <em>marginal</em>) stays exactly as fitted. The maintainer's comment on PR #140 set the standard for when this is legitimate. Only the co-movement the model cannot explain from what it sees is a sampling problem, and for contributions that part is small: a within-group correlation of 0.07, one sixth of the raw co-movement.",
  finding="With a fitted strength (rho) of 0.04 to 0.07 the copula should barely move a group-spread row. Instead CG went from 9.81 to 4.16 (PR #165) to 0.90 (PR #179). The effect compounds: a shared number held fixed for a whole game and fed through 50 rounds of the models reacting to each other grows to roughly 15 times its one-step prediction. The ablation on PR #179 reads it the same way: the copula supplies free-running variation that the deterministic network cannot generate on its own. That is the same disease the RCB work found, a network that behaves when fed the real human history and goes flat when it plays against the other models.",
  maths=["A Bayesian or ensemble treatment (several trained copies of the model, one drawn at random per game) addresses a different term. One draw per run shifts every player in every group the same way, which does nothing for a within-group spread ratio. A draw per group per game moves CG, but is as mechanistically wrong as the copula.",
         "The principled version, a per-group random effect fitted jointly with the model, is PR #159: the strength the likelihood allowed reached only 38 percent of the required move.",
         "So the copula stays defensible as a descriptive group-heterogeneity term under three conditions the protocol already enforces or nearly enforces: each player's marginal preserved per draw, strength set by likelihood rather than by the score it improves, and no distortion of individual responses. The third is the one to watch; PR #168 and PR #179 both report the shared number partly deciding <em>who</em> moves, not just how much the group moves."],
  code=["<code>src/aimanager/generic/copula.py</code> sample_correlated_levels, a per-(game, group) latent with round-to-round persistence", "<code>src/aimanager/simulation/linear_ah.py</code> _sample_levels_copula and _sample_levels_gaussian_copula", "<code>scripts/baselines/punishment_copula_rho.py</code> pairwise maximum-likelihood fit of rho, now with --bundle/--out"],
  bought="A cheap test settles the open question: train five copies of the network with different random seeds, measure how much they disagree per situation when fed the real history, and compare with the fitted rho. If the disagreement sits well below 0.04, model uncertainty is negligible and the copula stays. One surprise here points the same way: after the manager fix, the manager's own copula strength rose from 0.35 to 0.43, which fits a better-specified model leaving residuals that are more purely the manager's shared mood."),
 dict(id="rcb", chip="evaluation", color="var(--c-amber)", title="RCB is a weak instrument for the thing the manager needs",
  meta='reports/rcb_alternative_comparison.md &middot; 40 stacks &middot; both rows are defined side by side on the Response instrument tab',
  problem="The learning manager's only lever is punishment, so what the simulated players must get right is how they respond to it. RCB was the row scoring that. It takes punished players, sorts them by punishment rate (punishment divided by the shortfall from 20) and compares the average next-round change in each rate bin. The rate mixes how much a player gave with how hard they were hit, and in a regression on the human data the rate has the wrong sign once level and dose are controlled.",
  finding="Bin averages can be matched without the mechanism. The stacks with categorical players hold three of the five best RCB scores, yet their high contributors give more when punished, matching zero or one of the four human slope signs. The Gaussian-MLP line is mid-pack on RCB and at the noise ceiling on the slope row. Across 40 stacks the two rows rank the stacks almost independently (Spearman rank correlation 0.28); RCE correlates at &minus;0.88 with the number of human-signed slopes, RCB at &minus;0.08.",
  maths=["RCE: within each contribution band (0&ndash;4, 5&ndash;9, 10&ndash;14, 15&ndash;19) the slope of a straight-line fit of next-round change on punishment received, over the same punished players RCB uses. Human slopes +0.140, +0.104, &minus;0.077, &minus;0.161: comply when punished at low contribution, withdraw at high.",
         "The score is the average absolute slope difference over the four bands, weighted by how many humans fall in each band, divided by a human-versus-human resampling ceiling of 0.086.",
         "That ceiling is large relative to the effect, three quarters of the mean human slope, so a model with no response at all scores 1.42 and one with half the response 0.82. A two-band variant (0&ndash;9 versus 10&ndash;19) keeps the ranking with usable band edges if RCE is ever to decide acceptance alone."],
  code=["<code>src/aimanager/evaluation_suite/metrics.py</code> ResponseMetrics.rce, rce_weights, _rce_fit", "<code>src/aimanager/evaluation_suite/visuals.py</code> RCE_line figure", "<code>notes/autoresearch.md</code> section 2: RCE is the first protected row, 22 rows"],
  bought="RCE sits beside RCB as the first protected row: no accepted experiment may worsen its score band, flip a human sign, or halve a band's slope. Under that rule PRs #171, #172 and #179 would not have passed as written; each traded away the punishment response for a better mean."),
 dict(id="holdout", chip="diagnosis", color="var(--c-teal)", title="Learned, not memorised: the held-out teacher-forced test",
  meta=f'<a href="{PR}183">PR #183</a> &middot; branch rcb-holdout-teacher-forced &middot; 5 folds, Raven A100, 2.5 minutes each',
  problem="A simulated player's reaction to punishment can be read in two ways. <em>Teacher-forced</em>: feed the network the real human history round by round and read off only its predicted next contribution. <em>Self-play</em> (closed loop): let the models generate the whole game themselves, each round's inputs being the models' own earlier outputs. PR #181 reported that the group-vnode players react almost like humans teacher-forced (raw RCB gap 0.093, well inside the 0.348 noise ceiling) but not in self-play (0.797), and read the gap as drift of the simulated game state. But the 0.093 came from a model trained on all 50 games: a network with memory, tested on games it has memorised, matches averages almost by construction. The flat self-play result was equally consistent with a reaction that never generalised beyond the training games.",
  finding="Retrained five times, each time with 10 games held out, and teacher-forced through the copy that never saw each game, the pooled held-out statistic is 0.095, the pooled in-sample 0.082, the shipped model 0.093: within 0.013 of each other, all inside the 0.348 ceiling, eight times below self-play. In the two low contribution bands, 71 percent of the punished population, the held-out reaction is 83 to 94 percent of human strength.",
  maths=["Two secondary deficits are real and are not closed-loop: the 10&ndash;14 band has the wrong sign under every condition (never learned) and the 15&ndash;19 withdrawal is about 40 percent as strong held-out as in-sample (partly memorised).",
         "So the fixes worth pursuing act on the closed loop: the stimulus skip (a direct path from punishment received to the output) and a simulated manager whose punishments resemble the human ones. Work on input features or regularisation can only address the two secondary bands."],
  code=["<code>scripts/data_analysis/rcb_holdout_teacher_forced.py</code>", "<code>configs/training/artificial_humans/contribution/group_switching_contribution_50ep_group_vnode_holdout_folds.yml</code>", "<code>notes/autoresearch_log/rcb-holdout-teacher-forced.md</code>"],
  bought="It removed the wrong branch of the decision tree before any modelling effort went there, and it left a reusable held-out-fold harness."),
 dict(id="lag", chip="correctness", color="var(--c-red)", title="The lagged manager: punishing last round on every branch",
  meta=f'<a href="{PR}184">PR #184</a> &middot; commits 7ad1ddd, c74bc0e, ce70a09 &middot; both manager models, linear and graph-network',
  problem="In the real game the manager sees this round's contributions and punishes them in the same round. The data proves it: the recorded pot equals 1.6 times this round's contributions minus this round's punishments in every valid row; punishment correlates with the current contribution (&minus;0.28) more than with the previous one (&minus;0.19); and a player who just dropped from 20 to 4 or less is punished 57 percent of the time, one who just rose from 4 or less to 20 only 18 percent. The simulated manager, on every branch of the code, decided round t's punishment from round t&minus;1's contribution. Every simulation flattened or flipped that contrast.",
  finding="Training and simulation agreed with each other, so this was not a bug in one of them but a consistent model of the wrong mechanism. The one-round lag was inherited from the player model, where it is correct (a player cannot see this round's punishment before contributing), as a 'GNN convention'. A scan of all 60-plus branches found no manager config ever trained on the current contribution, and the feature-legality check hard-errored on any attempt, so the automated experiment loop (<em>autoresearch</em>: AI agents propose, run and score one change per PR) could not have fixed it by itself. The repo's own model-config report had recommended the fix; it was never implemented.",
  maths=["The fix allows the current contribution and its group means as inputs for the manager, while same-round punishment, payoff and pot stay forbidden because they already contain the answer. The code that prepares the inputs already had the current value in place and simply never read it.",
         "Both manager models retrained: cross-validated log loss (lower is better) 1.366 to 1.347 for the linear model, 1.203 to 1.176 for the graph network. Fed the human data, the regression weight of punishment on the current contribution goes from about zero to about &minus;0.13 (human &minus;0.24), the rose-versus-dropped contrast flips to the human ordering, and punishment again falls with contribution.",
         "Residual: full contributors are still punished 10 to 14 percent of the time when fed the human data, and 12 to 16 percent in the closed-loop simulations, against 4 percent for real managers; the weight on the current contribution is half the human one. That is a limit of the manager's functional form, not of timing."],
  code=["<code>scripts/baselines/handcrafted_grid.py</code> PUNISHMENT_LEGAL_CURRENT, illegal_current_features", "<code>src/aimanager/simulation/linear_ah.py</code>, <code>src/aimanager/manager/api_manager.py</code> input adapters", "<code>configs/training/baselines/punishment/multinomial_current_contr.yml</code>, <code>configs/training/artificial_humans/punishment/rnn_edge_50ep_doubled_current_contr.yml</code>", "<code>src/aimanager/tests/test_punisher_current_contribution.py</code>, 128 tests pass on Raven"],
  bought="RPA, the row scoring how the manager punishes, drops from 1.2&ndash;1.6 into 0.7&ndash;0.9 in all six stacks, and RCB falls by 0.5 to 0.9 everywhere. RCC, the reaction at the ceiling, does not move, because the full contributors still being punished are exactly the population RCC is made of."),
 dict(id="rebase", chip="re-baseline", color="var(--c-ink)", title="The re-baseline: what the fixed manager did to the frontier",
  meta=f'<a href="{PR}184">PR #184</a> &middot; five stacks, six runs &middot; ledger reset in notes/autoresearch.md',
  problem="Every accepted stack had been scored against a manager that punished last round. The fix changes every simulated game, so all 22 rows move at once and the usual rule for accepting an experiment (a score-band upgrade on a declared row, with the mean within 10 percent) does not apply. The honest artifact is a <em>re-baseline</em>: every stack rerun under the fixed manager, every row rescored, and the experiment record (the ledger in notes/autoresearch.md) reset to the new numbers.",
  finding="The means barely moved, because the RPA and RCB gains are offset elsewhere: the contribution distribution rows and the segregation row SC worsen by 0.15 to 0.4 in several stacks, and the group-vnode stack loses three rows from its count at or under 1. The player models were not retrained and did not need to be, since they learn from human data; but the states they reach in self-play shift under a different manager, and that is what these rows show.",
  maths=["Stimulus-skip stack (b, PR #181): mean 1.096 to 1.036, 13 rows at or under the ceiling, RCD (switching pull) 2.21 to 1.31, all four RCE signs right. The best mean on record.",
         "Gaussian-MLP stacks (c and d, PRs #177 and #174): RCE 0.85 and 0.70 with all signs right; c pushes RCB under 1.",
         "Group-vnode stack (a, PR #179): loses the human sign in two RCE bands, RCE 1.10 to 1.27. Under the new protected-row rule that is a violation.",
         "Main-branch reference with the graph-network manager (e2): mean 1.866 to 1.709, the largest single improvement, but two RCE signs lost through near-zero bands."],
  code=["<code>scripts/data_analysis/curpun_rebaseline.py</code>, <code>scripts/simulation/run_curpun_reruns.sh</code>", "<code>plots/data_analysis/evaluation/punisher_current_contr/</code> tables and mechanism_selfplay.csv", "<code>plots/simulation/*_curpun/</code> five simulation folders with per_round.parquet and the 22-row evaluations"],
  bought="Evidence for which model line to build on next (the <em>lineage</em>, a chain of experiments each starting from the last): the stimulus-skip stack or the Gaussian-MLP line, not the group-vnode line the ledger sat on. Caveats: the 32-stack sweep was not rerun; cases c and d ran on the Gaussian-MLP code tree with the three fix commits copied over; and the manager's copula strength rose rather than fell."),
 dict(id="next", chip="successor", color="var(--c-muted)", title="What this leaves for whoever continues",
  meta="ordered by expected value per hour",
  problem="Six threads are open. None is blocked.",
  finding="",
  maths=["The manager's functional form: a full contributor is still punished three to four times as often as by a human manager, and the weight on the current contribution is half the human one. A manager model with a term that knows 20 is the ceiling is the natural next experiment; it is what RCC needs.",
         "A gated stimulus skip on the player model: the follow-up PR #181 proposed, now to be judged on the protected RCE row rather than on RCB.",
         "Lineage: once the fixed-manager numbers are accepted, combine the stimulus-skip line and the Gaussian-MLP line into one code tree; they differ in 17 files.",
         "The seed ensemble for the copula question: five seeds, per-situation disagreement versus a rho of 0.04.",
         "Where the learning manager will act: humans rarely punished above 10 points or punished high contributors, about 300 rows in total. Either keep the learning manager's punishments near the shortfall, or audit where a trained policy actually operates.",
         "Rerun the 32-stack sweep under the fixed manager so the ledger's deficit profiles are all post-fix."],
  code=["<code>notes/autoresearch_log/punisher-current-contribution.md</code>, successor section", "Raven clean-up once PR #184 closes: four isolated folders under ~/repros/ai-runs/"],
  bought=""),
]

def story_html(s):
    parts = [f'<article class="story" id="story-{s["id"]}"><h3>{s["title"]}</h3><p class="meta"><span class="chip" style="background:{s["color"]}">{s["chip"]}</span>{s["meta"]}</p>']
    parts.append(f'<h4>The problem</h4><p>{s["problem"]}</p>')
    if s["finding"]: parts.append(f'<h4>What was found</h4><p>{s["finding"]}</p>')
    if s["maths"]: parts.append('<h4>The details, in plain English</h4><ol class="maths">' + "".join(f"<li>{m}</li>" for m in s["maths"]) + "</ol>")
    if s["code"]: parts.append('<h4>Where it lives in the code</h4><ul class="code">' + "".join(f"<li>{c}</li>" for c in s["code"]) + "</ul>")
    if s["bought"]: parts.append(f'<h4>What it bought</h4><p>{s["bought"]}</p>')
    parts.append("</article>")
    return "".join(parts)

ledger = [
 ("PR #182", f"{PR}182", "open", "Bookkeeping for the main branch: ignore the local config file, cluster account lines, the evaluation metric notes."),
 ("PR #183", f"{PR}183", "open", "The held-out teacher-forced test of the players' reaction to punishment, built on top of the PR #181 branch."),
 ("PR #184", f"{PR}184", "open", "The re-baseline: manager retrained on the current contribution, RCE made a protected row, ledger reset."),
 ("rcb-alternative-response-slope", "", "merged into PR #184", "The RCE row, its tests, and the comparison report over 40 stacks."),
 ("auto/punisher-current-contribution-sims", "", "documentation", "The rerun configs, the runner script, and the 'before' column of the score tables."),
 ("auto/punisher-current-contribution-gmlp", "", "documentation", "The Gaussian-MLP code tree plus the three fix commits copied over; used for cases c and d."),
]
def ledger_html():
    out=[]
    for n,u,st,d in ledger:
        link = '<a href="%s" target="_blank" rel="noopener">%s</a>' % (u, n) if u else n
        out.append('<tr><td class="pr">%s</td><td><span class="pill">%s</span></td><td>%s</td></tr>' % (link, st, d))
    return "".join(out)

CSS = r"""
:root{--bg:#f6f4ef;--panel:#ffffff;--ink:#1f1d1a;--muted:#615c53;--faint:#8f897d;--line:#e3ded3;--line2:#cfc8b9;--acc:#a8531c;--acc-soft:#f3e3d6;--sel:#f4ece2;
 --b1:#2f7d5b;--b2:#b3841c;--b3:#c2452d;--b4:#6b2323;--c-blue:#2f6fb4;--c-amber:#b3841c;--c-teal:#1f7a74;--c-red:#b23a2a;--c-ink:#3b3733;--c-muted:#7a746a;
 --hum:#3b3733;--bef:#c9b79a;--aft:#a8531c;--grid:#e3ded3;--tip:#1f1d1a;--tipfg:#f6f4ef;}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#17161a;--panel:#211f24;--ink:#ece8e0;--muted:#b3ada2;--faint:#7f7a70;--line:#34313a;--line2:#4a4650;--acc:#e28a4e;--acc-soft:#3a2a1f;--sel:#2b2620;
 --b1:#5fbf90;--b2:#d9ab4a;--b3:#e0705a;--b4:#c96a6a;--c-blue:#6fa3e0;--c-amber:#d9ab4a;--c-teal:#5cb8b1;--c-red:#e0705a;--c-ink:#9a948b;--c-muted:#7f7a70;--hum:#ece8e0;--bef:#6c6558;--aft:#e28a4e;--grid:#34313a;--tip:#ece8e0;--tipfg:#17161a;}}
:root[data-theme="dark"]{--bg:#17161a;--panel:#211f24;--ink:#ece8e0;--muted:#b3ada2;--faint:#7f7a70;--line:#34313a;--line2:#4a4650;--acc:#e28a4e;--acc-soft:#3a2a1f;--sel:#2b2620;
 --b1:#5fbf90;--b2:#d9ab4a;--b3:#e0705a;--b4:#c96a6a;--c-blue:#6fa3e0;--c-amber:#d9ab4a;--c-teal:#5cb8b1;--c-red:#e0705a;--c-ink:#9a948b;--c-muted:#7f7a70;--hum:#ece8e0;--bef:#6c6558;--aft:#e28a4e;--grid:#34313a;--tip:#ece8e0;--tipfg:#17161a;}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Source Sans 3","Segoe UI",system-ui,sans-serif;font-size:15px;line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding-block:20px 48px;padding-inline:20px}
h1{font-family:"Fraunces",Georgia,serif;font-weight:600;font-size:30px;margin:4px 0 2px;letter-spacing:-0.01em;text-wrap:balance}
h2{font-family:"Fraunces",Georgia,serif;font-weight:600;font-size:20px;margin:22px 0 8px;text-wrap:balance}
h3{font-size:16px;margin:0 0 4px;text-wrap:balance}
h4{font-size:11.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--faint);margin:14px 0 4px}
.sub{color:var(--muted);font-size:14px;margin:0 0 14px;max-width:70ch}
nav{display:flex;gap:2px;border-bottom:1px solid var(--line);margin-bottom:20px;flex-wrap:wrap}
nav button{border:0;background:none;color:var(--muted);padding:9px 14px;font:inherit;font-size:14px;cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1px}
nav button.on{color:var(--ink);font-weight:600;border-bottom-color:var(--acc)}
nav button:hover,nav button:focus-visible{color:var(--ink);outline:none}
nav button:focus-visible{box-shadow:0 0 0 2px var(--acc) inset}
.layer{display:none}.layer.on{display:block}
.legend{color:var(--muted);font-size:13px;margin:6px 0 10px;max-width:80ch}
.legend a,.meta a,td a{color:var(--acc);text-decoration:none}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:10px 0 18px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px;display:flex;flex-direction:column;gap:2px}
.tile .k{font-size:11.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--faint)}
.tile .v{font-family:"Fraunces",Georgia,serif;font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}
.tile .s{font-size:12.5px;color:var(--muted)}
ol.steps{list-style:none;padding:0;margin:8px 0 0;display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
ol.steps li{display:flex;gap:12px;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
ol.steps .n{flex:none;width:26px;height:26px;border-radius:50%;background:var(--acc);color:#fff;display:grid;place-items:center;font-weight:700;font-size:12.5px}
ol.steps p{margin:4px 0 0;font-size:13.5px;color:var(--muted)}
ol.steps p.meta{font-size:12px;color:var(--faint)}
.mnav{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0 14px}
.mnav button{border:1.5px solid var(--line2);border-radius:999px;background:none;color:var(--ink);font:inherit;font-size:13px;padding:4px 12px;cursor:pointer}
.mnav button small{color:var(--faint);font-size:11.5px;margin-left:4px}
.mnav button.on{border-color:var(--acc);background:var(--acc-soft);font-weight:600}
.mnav button:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
.bacard{display:none}.bacard.on{display:block}
.twocol{display:grid;grid-template-columns:1.15fr 1fr;gap:20px;align-items:start}
.tablewrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:13px}
thead th{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--faint);font-weight:600;padding:8px 10px;border-bottom:1px solid var(--line)}
th.num,td.num{text-align:right;font-variant-numeric:tabular-nums}
tbody td{padding:5px 10px;border-bottom:1px solid var(--line)}
tbody tr:last-child td{border-bottom:0}
tr.prot td{background:var(--sel)}
td.row .rn{display:block;font-size:11px;color:var(--faint);line-height:1.2}
tr.tot td{font-weight:600;border-top:1px solid var(--line2)}
td.neg{color:var(--b1);font-weight:600}td.pos{color:var(--b3)}
td.band{color:var(--muted);font-size:12px;white-space:nowrap}
.chip{display:inline-block;min-width:44px;text-align:right;padding:1px 7px;border-radius:6px;font-size:12.5px;font-weight:600;color:#fff}
.chip.b1{background:var(--b1)}.chip.b2{background:var(--b2)}.chip.b3{background:var(--b3)}.chip.b4{background:var(--b4)}.chip.na{background:var(--faint)}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11.5px;font-weight:600;background:var(--acc-soft);color:var(--acc)}
svg{max-width:100%;height:auto;display:block}
svg .grid{stroke:var(--grid);stroke-width:1}svg .grid.zero{stroke:var(--line2);stroke-width:1.2}
svg .tick{fill:var(--muted);font-size:11px;font-family:inherit}
svg .bar.hum{fill:var(--hum)}svg .bar.bef{fill:var(--bef)}svg .bar.aft{fill:var(--aft)}
svg .pt{fill-opacity:.85;stroke:var(--bg);stroke-width:1}
svg .pt.gnn{fill:var(--c-teal)}svg .pt.cat{fill:var(--c-red)}svg .pt.gau{fill:var(--c-amber)}svg .pt.rid{fill:var(--c-muted)}svg .pt.pr{fill:var(--c-blue)}
.keyline{font-size:12px;color:var(--muted);display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-top:4px}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;margin-left:8px}
.sw.hum{background:var(--hum)}.sw.bef{background:var(--bef)}.sw.aft{background:var(--aft)}
.sw.gnn{background:var(--c-teal)}.sw.cat{background:var(--c-red)}.sw.gau{background:var(--c-amber)}.sw.rid{background:var(--c-muted)}.sw.pr{background:var(--c-blue)}
.figrow{display:grid;grid-template-columns:repeat(2,1fr);gap:14px;margin:10px 0 16px}
.figrow figure{margin:0}.figrow img{width:100%;border:1px solid var(--line);border-radius:8px;background:#fff}
.figrow figcaption{font-size:12px;color:var(--muted);margin-top:4px}
.story{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 22px;margin:0 0 16px;max-width:760px}
.story p{margin:3px 0;line-height:1.55;font-size:14px}
.story .meta{color:var(--muted);font-size:12.5px}
.story .chip{min-width:0;text-align:left;margin-right:8px;border-radius:999px;font-size:11px}
ol.maths{padding-left:1.3rem;margin:4px 0}ol.maths li{margin:6px 0;font-size:14px;line-height:1.5}
ul.code{list-style:none;padding:0;margin:4px 0}ul.code li{margin:4px 0;font-size:12.5px;color:var(--muted)}
code{background:var(--sel);border-radius:4px;padding:1px 5px;font-size:12px;color:var(--ink);font-family:"JetBrains Mono",ui-monospace,Menlo,monospace}
table.mech tr.hum td{font-weight:600}table.mech tr.aft td{background:var(--sel)}
#tip{position:fixed;display:none;background:var(--tip);color:var(--tipfg);font-size:12px;padding:6px 9px;border-radius:6px;max-width:360px;pointer-events:none;z-index:10;line-height:1.4}
@media (max-width:900px){.tiles{grid-template-columns:repeat(2,1fr)}ol.steps{grid-template-columns:1fr}.twocol{grid-template-columns:1fr}.figrow{grid-template-columns:1fr}}
@media (max-width:520px){.tiles{grid-template-columns:1fr}h1{font-size:24px}}
@media (prefers-reduced-motion:no-preference){nav button,.mnav button{transition:color .15s,border-color .15s}}
"""

page = f'''<title>Punisher Rebaseline Atlas</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400&display=swap">
<style>{CSS}</style>
<div class="wrap">
<h1>Punisher Rebaseline Atlas</h1>
<p class="sub">Real people played a public-goods game with a punishing manager; this project trains simulated players and a simulated manager to behave like them. This page reports one working day, 18 September 2026, on those simulations: the copula question, why the simulated players seemed not to react to punishment, the discovery that the simulated manager punished last round's contribution on every branch of the code, and the five stacks rerun under the fix. Companion to the Autoresearch Atlas of experiments #146&ndash;#181.</p>
<nav>
  <button type="button" class="on" data-layer="overview">Overview</button>
  <button type="button" data-layer="scores">Before / after</button>
  <button type="button" data-layer="response">Response instrument</button>
  <button type="button" data-layer="mechanism">Punisher mechanism</button>
  <button type="button" data-layer="stories">Stories</button>
  <button type="button" data-layer="ledger">Ledger</button>
</nav>

<section class="layer on" id="overview">
{ABOUT}
<h2>The headline numbers</h2>
<p class="legend">Each tile is one row of the 22-row score card, before and after the manager fix. RPA scores how the manager punishes, RCB and RCE score how the players react to punishment. A row is <em>protected</em> when no future experiment may make it worse.</p>
{overview_tiles()}
<h2>How the work unfolded</h2>
<ol class="steps">{steps_html()}</ol>
<h2>The one-paragraph reading</h2>
<p class="legend" style="max-width:76ch;font-size:14px;color:var(--ink)">Where the fix applies directly it worked: the row scoring how the manager punishes (RPA) and the reaction-to-punishment row (RCB) improve by a full score band in every stack, and the simulated manager now punishes low contributors more than full ones, in the same order as the real managers do. It did not move the reaction at the ceiling (RCC), because the retrained manager still punishes full contributors three to four times too often; that is a limit of the manager's functional form, not of timing. The gains were offset elsewhere: the contribution distributions and the segregation row got slightly worse, so the 22-row means stayed roughly flat. For the choice of which model line to build on, the evidence now favours the stimulus-skip stack (PR #181) or the Gaussian-MLP line (PRs #174, #177), both of which keep all four human response signs under the fixed manager, over the group-vnode line (PR #179) on which the experiment record had been built.</p>
</section>

<section class="layer" id="scores">
<p class="legend">Every row of the score card, for each of the six stacks that were rerun. Pick a stack: one combination of player, switch and manager model. Each row compares one statistic of the simulated games with the same statistic in the real games. Scores are multiples of the human-versus-human noise ceiling: at or under 1 is indistinguishable from real data, 1&ndash;2 a minor deviation, 2&ndash;5 a clear one, above 5 not reproduced. A negative &Delta; is an improvement. The shaded row is RCE, the protected response row. Two player-model families appear: graph-network players pass messages between all eight players and keep a memory across rounds; Gaussian-MLP players draw each contribution from a bell curve whose centre and width a small neural network predicts.</p>
<div class="mnav" id="case-nav">{case_buttons()}</div>
{score_cards()}
</section>

<section class="layer" id="response">
<h2>Two ways to measure the reaction to punishment</h2>
<p class="legend">The learning manager's only lever is punishment, so the one thing the simulated players must get right is how they react to it. Two rows score that reaction. To tell them apart, take a player who gave 3 points, was punished 5 points, and gave 8 the next round: a change of +5.</p>
<div class="twocol">
<div>
<h4>RCB, reaction to punishment (the old row)</h4>
<p class="legend" style="color:var(--ink)">RCB works with the punishment <em>rate</em>: punishment divided by the shortfall from 20. Our player's rate is 5 / 17 = 0.29, which lands in the 0.25&ndash;0.5 bin. RCB records the +5 as one entry in that bin's average, then compares the four bin averages (rates up to 0.25, 0.25&ndash;0.5, 0.5&ndash;1, above 1) with the human ones. Real players give more after harder punishment: on average +0.9, +1.3, +1.7 and +2.0 across the four bins.</p>
<p class="legend" style="color:var(--ink)"><b>Why it can be fooled.</b> The rate mixes how much someone gave with how hard they were hit. A rate above 1 is reached both by a low contributor punished heavily and by a high contributor punished lightly, and those two react in opposite directions. A simulation can match the bin averages by having the right mix of players in each bin, even if none of them responds to the dose. In the human data, once level and dose are held fixed, a higher rate goes with a smaller change, not a larger one.</p>
</div>
<div>
<h4>RCE, punishment response slope (the new row)</h4>
<p class="legend" style="color:var(--ink)">RCE first sorts punished players by what they gave: 0&ndash;4, 5&ndash;9, 10&ndash;14 or 15&ndash;19 points. Our player is in the 0&ndash;4 band. Within each band it fits a straight line of next-round change against punishment received and keeps the slope: how many more points a player gives per extra point of punishment. Real players comply at low levels and withdraw at high ones: slopes +0.140, +0.104, &minus;0.077, &minus;0.161. The score is the weighted gap between the simulated and the human slopes over the four bands.</p>
<p class="legend" style="color:var(--ink)"><b>Why it was added but does not decide alone.</b> Its noise ceiling is large. Two halves of the human data differ by 0.086 in slope, about three quarters of the human slopes themselves. So a simulation whose players ignore punishment entirely scores 1.42, a 'minor deviation', and one with half the human response scores 0.82, at the ceiling. RCE therefore sits beside RCB as a protected row: no experiment may worsen its score band, flip one of the four human signs, or halve a band's slope.</p>
</div>
</div>
<h2>RCB and RCE rank the stacks almost independently</h2>
<p class="legend">Forty stacks, the 32 combinations of the main sweep plus eight from accepted experiments, each scored on both rows. Spearman rank correlation 0.28 overall, 0.11 within the sweep: a good RCB score says little about RCE. Hover a point for the stack and how many of its four slope signs match the human ones. Vertical guide at RCB 2, horizontal at RCE 1.</p>
<div class="twocol">
<div>{scatter_svg()}<div class="keyline"><span class="sw gnn"></span>graph-network players <span class="sw cat"></span>categorical players <span class="sw gau"></span>gaussian players <span class="sw rid"></span>ridge players <span class="sw pr"></span>experiment (PR) stacks</div></div>
<div>
<h4>Why they disagree</h4>
<p class="legend" style="color:var(--ink)">RCB tracks who gets punished at which level; RCE tracks how they respond to the dose. The stacks with categorical players (red) hold three of the five best RCB scores, yet their high contributors give more when punished, matching zero or one of the four human signs. The Gaussian-MLP line (blue, lower left) is mid-pack on RCB and at the ceiling on RCE. Over the 40 stacks RCE correlates at &minus;0.88 with the number of human-signed slopes, RCB at &minus;0.08.</p>
<h4>Noise ceilings</h4>
<div class="tablewrap"><table><thead><tr><th>row</th><th class="num">ceiling</th><th class="num">no response scores</th><th class="num">half response scores</th></tr></thead>
<tbody><tr><td>RCB</td><td class="num">0.348</td><td class="num">3.64</td><td class="num">1.83</td></tr><tr><td>RCE, four bands</td><td class="num">0.086</td><td class="num">1.42</td><td class="num">0.82</td></tr><tr><td>RCE, two bands</td><td class="num">0.060</td><td class="num">2.15</td><td class="num">&ndash;</td></tr></tbody></table></div>
<p class="legend">Ceilings from 500 human-versus-human resampling repeats (seed 42): each repeat splits the 50 real games in half and measures the distance between the halves. 'No response' is a simulation whose players ignore punishment; 'half response' one whose slopes are half the human ones. The two-band variant (0&ndash;9 versus 10&ndash;19) has a tighter ceiling and is the candidate if RCE is ever to decide acceptance alone.</p>
</div></div>
<h2>Figures from the evaluation suite</h2>
<div class="figrow">
<figure><img src="{IMG['rcb_vs_rce']}" alt="RCB score against RCE score across stacks"><figcaption>RCB versus RCE scores over the 40 stacks, from reports/rcb_alternative_comparison.md.</figcaption></figure>
<figure><img src="{IMG['rce_four']}" alt="RCE slopes, human versus four stacks"><figcaption>RCE slopes per contribution band, humans against four representative stacks, before the manager fix.</figcaption></figure>
<figure><img src="{IMG['b_rce']}" alt="RCE figure for the stimulus-skip stack after the fix"><figcaption>The RCE figure for the PR #181 stimulus-skip stack under the fixed manager: all four human signs, RCE 0.89.</figcaption></figure>
<figure><img src="{IMG['a_rce']}" alt="RCE figure for the vnode stack after the fix"><figcaption>The RCE figure for the PR #179 group-vnode stack under the fixed manager: two signs lost through near-zero bands, RCE 1.27.</figcaption></figure>
</div>
</section>

<section class="layer" id="mechanism">
<h2>Does the simulated manager punish what it sees?</h2>
<p class="legend">This tab checks the simulated manager directly, before and after the fix. Each row is one stack's self-play simulation (the models playing against each other, 19,200 player-rounds), compared with the real managers over 8,914 valid rows. The columns: how often a player who gave the full 20 is punished; how often one who gave 4 or less is; a timing check, the punishment probability when a player just rose to 20 from 4 or less against when they just dropped to 4 or less from 20 (a manager reacting to this round punishes the drop, one reacting to last round punishes the rise); the average punishment when punished, by contribution band; and the regression weights of punishment on this round's and last round's contribution (ordinary least squares). Shaded rows are after the fix.</p>
<div class="tablewrap"><table class="mech"><thead><tr><th>simulation</th><th class="num">P(punished | gave 20)</th><th class="num">P(punished | gave &le; 4)</th><th class="num">timing check: rose / dropped</th><th class="num">mean punishment if punished, by band 0-4 / 5-9 / 10-14 / 15-19 / 20</th><th class="num">regression weight, current / previous</th></tr></thead>
<tbody>{mech_table()}</tbody></table></div>
<p class="legend">Before the fix every simulated manager had a near-zero weight on the current contribution and about &minus;0.08 on the previous one, the reverse of the human pattern, and punished full contributors five to nine times too often. After the fix the current-round weight is &minus;0.12 to &minus;0.17 against the human &minus;0.24, the timing check has the human ordering, and punishment again falls as contribution rises. What remains is the ceiling: full contributors are still punished three to four times too often, and when they are punished the amount is too small. That residual is why RCC, the reaction at the ceiling, did not move.</p>
<div class="figrow">
<figure><img src="{IMG['b_rpa']}" alt="RPA figure for the stimulus-skip stack after the fix"><figcaption>RPA, how the manager punishes at each contribution level, for the PR #181 stimulus-skip stack under the fixed manager: 1.23 to 0.69.</figcaption></figure>
<figure><img src="{IMG['d_rce']}" alt="RCE figure for the k-one-hot gmlp stack after the fix"><figcaption>The RCE figure for the PR #174 Gaussian-MLP stack under the fixed manager: the strongest response on record, RCE 0.70.</figcaption></figure>
</div>
</section>

<section class="layer" id="stories">
<p class="legend">The plain-language story of each finding, in the order they were reached. Each card starts from the game, says what was found, gives the details, and points to where it lives in the code.</p>
{"".join(story_html(s) for s in STORIES)}
</section>

<section class="layer" id="ledger">
<h2>Pull requests and branches</h2>
<p class="legend">Where the work lives. A branch is one line of code changes; a pull request (PR) proposes merging it into the main branch.</p>
<div class="tablewrap"><table><thead><tr><th>item</th><th>state</th><th>what it holds</th></tr></thead><tbody>{ledger_html()}</tbody></table></div>
<h2>Numbers that changed hands</h2>
<p class="legend">The quantities the work moved, for anyone who needs to quote or check them.</p>
<div class="tablewrap"><table><thead><tr><th>quantity</th><th class="num">before</th><th class="num">after</th><th>source</th></tr></thead><tbody>
<tr><td>Linear manager model, cross-validated log loss (lower is better)</td><td class="num">1.366</td><td class="num">1.347</td><td>same data split and seed</td></tr>
<tr><td>Graph-network manager model, cross-validated log loss</td><td class="num">1.203</td><td class="num">1.176</td><td>Raven job, 7 min 47 s on one A100 GPU</td></tr>
<tr><td>Manager copula strength (rho)</td><td class="num">0.351</td><td class="num">0.427</td><td>pairwise maximum likelihood, confidence interval 0.351&ndash;0.528</td></tr>
<tr><td>Regression weight of punishment on the current contribution, linear manager fed human data</td><td class="num">+0.054</td><td class="num">&minus;0.125</td><td>human &minus;0.242</td></tr>
<tr><td>Raw RCB gap of the group-vnode players fed human data</td><td class="num">0.093 shipped model, in-sample</td><td class="num">0.095 held-out</td><td>PR #183</td></tr>
<tr><td>Rows in the score card</td><td class="num">21</td><td class="num">22</td><td>RCE added, RCF dropped</td></tr>
</tbody></table></div>
<h2>Clean-up once PR #184 closes</h2>
<ul class="code">
<li>On the Raven cluster: <code>~/repros/ai-runs/punisher-current-contr</code>, <code>…-gmlp</code>, <code>…-train</code>, <code>…-tests</code></li>
<li>Local worktree <code>.claude/worktrees/curpun-gmlp</code>; the documentation branches for the simulations and the Gaussian-MLP setup</li>
<li>The Raven checkout itself, <code>~/repros/algorithmic-institutions</code> with <code>~/algorithmic-institutions</code> symlinked, stays</li>
</ul>
</section>
</div>
<div id="tip" role="tooltip"></div>
<script>
(function(){{
  var tabs=document.querySelectorAll("nav button");
  function show(id){{tabs.forEach(function(b){{b.classList.toggle("on",b.dataset.layer===id)}});
    document.querySelectorAll(".layer").forEach(function(l){{l.classList.toggle("on",l.id===id)}});
    try{{localStorage.setItem("pra-tab",id)}}catch(e){{}}}}
  tabs.forEach(function(b){{b.addEventListener("click",function(){{show(b.dataset.layer)}})}});
  var cbs=document.querySelectorAll("#case-nav button");
  function showCase(c){{cbs.forEach(function(b){{b.classList.toggle("on",b.dataset.case===c)}});
    document.querySelectorAll(".bacard").forEach(function(x){{x.classList.toggle("on",x.dataset.case===c)}});
    try{{localStorage.setItem("pra-case",c)}}catch(e){{}}}}
  cbs.forEach(function(b){{b.addEventListener("click",function(){{showCase(b.dataset.case)}})}});
  try{{var t=localStorage.getItem("pra-tab");if(t&&document.getElementById(t))show(t);
      var c=localStorage.getItem("pra-case");if(c&&document.getElementById("case-"+c))showCase(c)}}catch(e){{}}
  var tip=document.getElementById("tip");
  document.querySelectorAll("svg .pt, svg .bar").forEach(function(el){{
    var t=el.querySelector("title");if(!t)return;var txt=t.textContent;t.remove();
    el.addEventListener("mousemove",function(ev){{tip.textContent=txt;tip.style.display="block";tip.style.left=(ev.clientX+12)+"px";tip.style.top=(ev.clientY+12)+"px"}});
    el.addEventListener("mouseleave",function(){{tip.style.display="none"}});
  }});
}})();
</script>
'''
OUT.write_text(page)
print(OUT, OUT.stat().st_size)
