import ast, base64, csv, json, html, math, os, re, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path("/Users/brinkmann/repros/algorithmic-institutions")
CACHE = HERE / "data"
D = Path("/Users/brinkmann/repros/algorithmic-institutions/.claude/worktrees/agent-a80d622b939db4c1c")
R = Path("/Users/brinkmann/repros/algorithmic-institutions/.claude/worktrees/agent-a91a63a720dcceb19")
OUT = HERE / "punisher_rebaseline_atlas.html"
PC = D / "plots/data_analysis/evaluation/punisher_current_contr"
RA = R / "plots/data_analysis/evaluation/rcb_alternative"
PR = "https://github.com/center-for-humans-and-machines/algorithmic-institutions/pull/"
# The four September measurement PRs this report reads (#186, #187, #188, #191)
# were consolidated onto main; SEPT_BRANCH overrides that while the PR is open.
SEPT = os.environ.get("SEPT_BRANCH", "main")

# ---------- committed files on other branches (LFS-smudged, cached beside this script) ----------
def gitfile(branch, path):
    # the cache keeps the full path: one branch now ships round_blocks.csv under
    # two analysis dirs, and a basename key would serve the wrong one.
    dst = CACHE / branch.replace("/", "-") / path
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        raw = subprocess.run(["git", "-C", str(REPO), "show", f"origin/{branch}:{path}"],
                             capture_output=True, check=True).stdout
        if raw.startswith(b"version https://git-lfs"):
            raw = subprocess.run(["git", "-C", str(REPO), "lfs", "smudge"], input=raw,
                                 capture_output=True, check=True).stdout
        dst.write_bytes(raw)
    return dst

def drows(branch, path):
    with open(gitfile(branch, path)) as f:
        return list(csv.DictReader(f))

def dkey(branch, path, col):
    return {r[col]: r for r in drows(branch, path)}

def djson(branch, path):
    return json.loads(gitfile(branch, path).read_text())

def md_table(branch, path, marker):
    """Rows of the first markdown table whose header line contains `marker`."""
    lines = gitfile(branch, path).read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if marker in l and l.startswith("|"))
    out = []
    for l in lines[i + 2:]:
        if not l.startswith("|"):
            break
        out.append([c.strip().strip("*").strip() for c in l.strip().strip("|").split("|")])
    return {r[0]: r for r in out}

def md_tables(branch, path, marker):
    """Every markdown table in `path` whose header line contains `marker`, keyed by first cell."""
    lines = gitfile(branch, path).read_text().splitlines()
    out = []
    for i, l in enumerate(lines):
        if not (l.startswith("|") and marker in l):
            continue
        rows = {}
        for l2 in lines[i + 2:]:
            if not l2.startswith("|"):
                break
            cells = [c.strip().strip("*").strip() for c in l2.strip().strip("|").split("|")]
            rows[cells[0]] = cells
        out.append(rows)
    return out

def lead(cell):
    """The leading signed number of a cell like '+0.013 +- 0.019 (n 1375)'."""
    return float(re.match(r"\s*([+-]?[0-9.]+)", cell).group(1))

def F(v):
    return float(v)

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
BANDS = ["0-4","5-9","10-14","15-19"]

mech = {}
with open(PC / "mechanism_selfplay.csv") as f:
    for row in csv.DictReader(f):
        mech[row["sim"]] = row

cmp_rows = []
with open(RA / "comparison_table.csv") as f:
    for row in csv.DictReader(f):
        cmp_rows.append(row)

# ---------- the seven pull requests that followed the re-baseline ----------
B_CL, B_MS, B_SE = SEPT, SEPT, SEPT
B_HS, B_CE, B_SK = SEPT, "auto/punisher-ceiling-fix", "auto/switch-kexo-port"
P_CL = "plots/data_analysis/evaluation/copula_closed_loop/"
P_MS = "plots/data_analysis/evaluation/copula_missing_state/"
P_SE = "plots/data_analysis/copula_seed_ensemble/"
P_HS = "plots/data_analysis/evaluation/head_state_spread/"
P_CE = "plots/data_analysis/evaluation/punisher_ceiling_fix/"
P_SK = "plots/data_analysis/evaluation/switch_kexo_port/"

# PR #186: the three-arm ablation of the shared draw
cl_dec = dkey(B_CL, P_CL + "cg_decomposition.csv", "arm")
cl_blk = {(r["arm"], r["rounds"]): r for r in drows(B_CL, P_CL + "round_blocks.csv")}
cl_sc = dkey(B_CL, P_CL + "scores_22.csv", "row")
cl_rce = dkey(B_CL, P_CL + "rce_bands.csv", "arm")
with open(gitfile(B_CL, P_CL + "latent_regression.csv")) as f:
    cl_lat = dict(csv.reader(f))

# PR #187: what the shared error is made of, on the human data
ms_base = dkey(B_MS, P_MS + "baseline.csv", "quantity")
ms_joint = dkey(B_MS, P_MS + "joint_model.csv", "scale")
ms_cand = dkey(B_MS, P_MS + "candidates.csv", "candidate")
ms_ref = dkey(B_MS, P_MS + "reference_sets.csv", "set")
ms_pers = dkey(B_MS, P_MS + "persistence_boot.csv", "stage")
ms_fwd = drows(B_MS, P_MS + "forward_selection.csv")

# PR #188: five copies of the model, trained with different random seeds
se_sum = djson(B_SE, P_SE + "train40_summary.json")
se_dis = dkey(B_SE, P_SE + "train40_disagreement.csv", "quantity")
se_sim = {r["metric"]: F(r["score"]) for r in drows(
    B_SE, "plots/simulation/23_2g8a_contr_stimulus_skip_seed_ensemble_self_gnncopar1_contr_gnn_switch_curpun/evaluation/scores.csv")}
se_mean = sum(se_sim.values()) / len(se_sim)
se_le1 = sum(1 for v in se_sim.values() if v <= 1)

# PR #191 (step 1): emission head and the spread of the states the players reach
hs_head = dkey(B_HS, P_HS + "headline.csv", "arm")
hs_boot = dkey(B_HS, P_HS + "retention_bootstrap.csv", "arm")
hs_blk = {(r["arm"], r["rounds"]): r for r in drows(B_HS, P_HS + "round_blocks.csv")}
hs_gain = {(r["model"], r["delta"]): r for r in drows(B_HS, P_HS + "gain_curves.csv") if r["set"] == "common_6_14"}
hs_recon = md_tables(B_HS, "notes/autoresearch_log/head-state-spread-diagnostic.md", "mean predictive variance")[0]
GAIN_D = ["-6", "-4", "-2", "2", "4", "6"]

# PR #192 (step 2): the manager at the contribution ceiling
ce_ba = dkey(B_CE, P_CE + "before_after.csv", "metric")
ce_mech = dkey(B_CE, P_CE + "mechanism_selfplay.csv", "")
ce_tf = dkey(B_CE, P_CE + "mechanism_teacher_forced.csv", "")
ce_logit = dkey(B_CE, P_CE + "human_ceiling_logit.csv", "")
ce_cv = drows(B_CE, "data/baselines/punishment_cv_multinomial_ceiling.csv")[0]
ce_cv0 = drows(B_CE, "data/baselines/punishment_cv_multinomial_current_contr.csv")[0]
ce_rcc = md_table(B_CE, "notes/autoresearch_log/punisher-ceiling-fix.md", "dc, punished")

def ce_mean(col): return sum(F(ce_ba[r][col]) for r in ROWS) / len(ROWS)
def ce_le1(col): return sum(1 for r in ROWS if F(ce_ba[r][col]) <= 1)

ce_sl = md_tables(B_CE, P_CE + "before_after.md", "| RCE slopes |")   # frontier, ref_lin, ref_gnn
ce_se = [ast.literal_eval(m) for m in re.findall(r"protected-row checks: (\{.*\})", gitfile(B_CE, P_CE + "before_after.md").read_text())]

# PR #190 (step 3): the other lineage's group-switching component
sk_cmp = dkey(B_SK, P_SK + "compare.csv", "metric")
sk_rce = dkey(B_SK, P_SK + "rce_bands.csv", "stage")

HUMAN_SLOPES = [F(sk_rce["human"]["slope_" + b]) for b in BANDS]

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

NUM = ' class="num"'
def tbl(heads, rows, cls=""):
    th = "".join("<th%s>%s</th>" % (NUM if h.startswith("~") else "", h.lstrip("~")) for h in heads)
    body = []
    for r in rows:
        tds = "".join("<td%s>%s</td>" % (NUM if h.startswith("~") else "", c) for h, c in zip(heads, r))
        body.append("<tr>%s</tr>" % tds)
    return '<div class="tablewrap"><table class="%s"><thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>' % (cls, th, "".join(body))

def ci(lo, hi, d=3):
    sg = F(lo) < 0 or F(hi) < 0
    return f"[{fmt(F(lo), d, sign=sg)}, {fmt(F(hi), d, sign=sg)}]"

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

# ---------- SVG: grouped bars, shared helper ----------
def grouped_bars(groups, series, ymin, ymax, ticks, label, tickfmt="{:.0f}", W=640, H=270,
                 labels_last=False, base=None, vfmt="{:.2f}"):
    """groups: [(group label, [v1, v2, ...])]; series: [(name, css class)].
    Bars run from `base` (default ymin, i.e. a true zero baseline) to the value."""
    if base is None: base = ymin
    L, Rm, T, B = 40, 12, 22, 40
    def y(v): return T + (ymax - v) / (ymax - ymin) * (H - T - B)
    gw = (W - L - Rm) / len(groups)
    n = len(series)
    bw = (gw * 0.80) / n - 2
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(label)}">']
    for tv in ticks:
        yy = y(tv)
        out.append(f'<line x1="{L}" x2="{W-Rm}" y1="{yy:.1f}" y2="{yy:.1f}" class="grid{" zero" if tv == base else ""}"/>')
        out.append(f'<text x="{L-6}" y="{yy+4:.1f}" class="tick" text-anchor="end">{tickfmt.format(tv)}</text>')
    for i, (glab, vals) in enumerate(groups):
        x0 = L + i * gw + gw * 0.10
        for j, v in enumerate(vals):
            if v is None: continue
            x = x0 + j * (bw + 2)
            top = min(y(base), y(v)); h = abs(y(v) - y(base))
            out.append(f'<rect x="{x:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{max(h,1.0):.1f}" rx="2" class="bar {series[j][1]}">'
                       f'<title>{esc(glab)} &middot; {esc(series[j][0])}: {vfmt.format(v)}</title></rect>')
            if labels_last and i == len(groups) - 1:
                out.append(f'<text x="{x+bw/2:.1f}" y="{top-4:.1f}" class="tick" text-anchor="middle">{v:.1f}</text>')
        out.append(f'<text x="{L + i*gw + gw/2:.1f}" y="{H-14}" class="tick" text-anchor="middle">{glab}</text>')
    out.append('</svg>')
    return "\n".join(out)

def keyline(series):
    return '<div class="keyline">' + " ".join(f'<span class="sw {c}"></span>{esc(nm)}' for nm, c in series) + '</div>'

SPREAD_SERIES = [("real people", "hum"), ("categorical players", "s1"), ("inflated Gaussian players", "s2"), ("plain Gaussian players", "s3")]
def spread_svg():
    arms = ["e_skip_kexo_rho0", "c_infl_rho0", "d_kexo_rho0"]
    groups = []
    for blk, lab in (("1-8", "rounds 1–8"), ("9-16", "rounds 9–16"), ("17-24", "rounds 17–24")):
        vals = [F(cl_blk[("human", blk)]["sd_group_mean"])] + [F(hs_blk[(a, blk)]["sd_group_mean"]) for a in arms]
        groups.append((lab, vals))
    return grouped_bars(groups, SPREAD_SERIES, 0, 7, [0, 2, 4, 6],
                        "How far apart the groups drift, by third of the game", labels_last=True)

GAIN_SERIES = [("categorical players", "s1"), ("inflated Gaussian players", "s2"), ("plain Gaussian players", "s3")]
def gain_svg():
    models = ["skip_categorical", "infl", "v2"]
    groups = [(("+" if d[0] != "-" else "−") + d.lstrip("-+"),
               [F(hs_gain[(m, d)]["gain_e"]) for m in models]) for d in GAIN_D]
    return grouped_bars(groups, GAIN_SERIES, 0.82, 1.06, [0.85, 0.90, 0.95, 1.00, 1.05],
                        "How much each model still moves when pushed away from real situations",
                        tickfmt="{:.2f}", H=250, base=1.0)

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
    spread_sim = F(cl_dec["B"]["var_cond_mean"]); spread_hum = F(cl_dec["human"]["var_cond_mean"])
    ceil_b = F(ce_mech["frontier before"]["P(p>0|c_t=20)"]); ceil_a = F(ce_mech["frontier after"]["P(p>0|c_t=20)"])
    ceil_h = F(ce_mech["human"]["P(p>0|c_t=20)"])
    dc_sim = abs(F(ce_rcc["frontier after"][2])); dc_hum = abs(F(ce_rcc["human"][2]))
    best = min(ce_mean("frontier_after"), F(sk_cmp["mean"]["after"]), means["b_skip_after"])
    return f'''
<div class="tiles">
  <div class="tile"><span class="k">how varied the situations get</span><span class="v">{spread_sim:.1f} vs {spread_hum:.1f}</span><span class="s">with the shared-noise machinery switched off, the simulated games reach two thirds of the variety of real ones. This is the one defect left standing.</span></div>
  <div class="tile"><span class="k">punished after giving everything</span><span class="v">{ceil_b*100:.1f}% &rarr; {ceil_a*100:.1f}%</span><span class="s">real managers {ceil_h*100:.1f}%. An indicator for &ldquo;gave the maximum&rdquo; made the simulated manager behave almost exactly like a real one here (PR #192).</span></div>
  <div class="tile"><span class="k">how hard players take that punishment</span><span class="v">&minus;{dc_sim:.2f} vs &minus;{dc_hum:.2f}</span><span class="s">points a punished full contributor gives up next round. The simulated players under-react by about {dc_hum/dc_sim:.1f} times, and only the players can fix it.</span></div>
  <div class="tile"><span class="k">the best 22-row mean on record</span><span class="v">{best:.3f}</span><span class="s">from PR #190, which failed both the row it had declared and the protected response row. Two of the seven experiments since the re-baseline failed their gate; a third falsified its own hypothesis.</span></div>
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
<p class="legend" style="color:var(--ink)"><b>What was wrong.</b> A stubborn deficit in the reaction-to-punishment row turned out to come from the simulated manager: on every branch of the code it punished the <em>previous</em> round's contribution, whereas real managers punish the current one. The manager was retrained, a sharper measure of the players' response (RCE) was added, and every stack was rescored under the fix.</p>
<p class="legend" style="color:var(--ink)"><b>Where it stands now.</b> Seven further experiments narrowed what is left. The simulated players' round-by-round randomness is the right size and their one-step reaction to punishment is learned rather than memorised, so neither is the problem. What is wrong is that the simulated games stay too much alike: real groups keep pulling apart as a game runs and the simulated ones stop. Three smaller defects are cleanly isolated. Two of the four most recent steps failed the bar they had set themselves and a third came back against the hypothesis its own author had proposed. Those three results are the most useful things on this page.</p>
</div>
</div>'''

STEPS = [
 ("Two questions", "First: the <em>copula</em>, a shared random number that makes the members of one group act alike, fixes the group-spread rows far better than its small fitted strength predicts. Is that a real mechanism or a patch? Second: the row scoring how simulated players react to punishment (RCB) had been stuck above 2 in every stack. Why?", "the starting point"),
 ("Four investigations", "Read-only surveys of the code, the experiment record and the data: how the copula works and what justifies it; what RCB actually measures; whether the simulated manager punishes at the right time; and whether the compute cluster had a usable checkout (it did not).", "no models changed"),
 ("Held-out test", "Is the players' reaction to punishment learned, or memorised from the training games? Retrained five times with 10 games held out each time, the reaction on unseen games (0.095) matches the in-sample one (0.082). It is learned; it only goes flat when the players play against the simulated manager.", "PR #183"),
 ("A better instrument", "RCB compares bin averages and can be matched without the right cause and effect. A new row, RCE, measures how much more a player gives per extra point of punishment. Over 40 stacks the two rank the stacks almost independently (rank correlation 0.28); RCE tracks whether the human response signs are reproduced (&minus;0.88), RCB does not (&minus;0.08).", "branch rcb-alternative-response-slope"),
 ("Fix and retrain", "The simulated manager punished last round's contribution on every branch of the code. Both manager models were retrained on the current round, and their fit to the human data (cross-validated log loss) improves.", "branch auto/punisher-current-contribution"),
 ("Re-baseline", "Five stacks rerun under the fixed manager and all 22 rows rescored. The experiment record (the <em>ledger</em>) was reset to the new numbers, and RCE became the first <em>protected row</em>: one no future experiment may worsen.", "PR #184"),
 ("Three answers on the shared error", "Why do the members of one simulated group get things wrong together? Three experiments, run side by side. The models' own randomness is already the right size; observable group facts explain a seventh of what is left; the models' uncertainty about themselves is far too small to be the cause. What the patch really supplies is variety, not correlation.", "PRs #186, #187, #188"),
 ("A programme, and a freeze", "Four steps declared in advance: test a different output design, fix the manager at the contribution ceiling, borrow the other model line's group-switching component, and change the rules so that the noise settings stop moving underneath every experiment.", "PR #189"),
 ("Two failed, one was falsified", "The output design test came back the opposite way round from its own hypothesis. The ceiling fix worked as a mechanism and still missed its target row, which turned out to be measuring the players and not the manager. The borrowed switching component improved every pure switching measure and damaged the response ones.", "PRs #190, #191, #192"),
]

def steps_html():
    return "".join(f'<li><span class="n">{i+1}</span><div><h3>{t}</h3><p>{d}</p><p class="meta">{m}</p></div></li>' for i,(t,d,m) in enumerate(STEPS))

STORIES = [
 dict(id="copula", chip="correlated-sampling", color="var(--c-blue)", title="The copula question: a variance source wearing a correlation's clothes",
  meta=f'PRs <a href="{PR}160">#160</a>, <a href="{PR}165">#165</a>, <a href="{PR}170">#170</a>, <a href="{PR}179">#179</a> &middot; the copula is used in both the manager and the player models',
  problem=f"Members of a real group act alike: they see the same situation and read it the same way. Simulated players drawn one at a time do not, so the rows that measure how far groups drift apart (CG, the group-spread ratio, and SC, segregation) scored badly in early stacks. The fix in use is the copula: one shared random number per group (the <em>latent</em>) is mixed into every member's draw, so members move together while each member's own probability distribution over choices (the <em>marginal</em>) stays exactly as fitted. The maintainer's comment on PR #140 set the standard for when this is legitimate. Only the co-movement the model cannot explain from what it sees is a sampling problem, and that part is small. PR #140 put it at 0.07 from a straight-line fit of the situation; measured against the graph network's own expectation it is smaller still, {F(ms_base['level residual, plain Pearson']['value']):.3f} out of a raw {F(ms_base['raw contribution, plain Pearson']['value']):.2f} (PR #187), so the network already accounts for {(1-F(ms_base['level residual, plain Pearson']['value'])/F(ms_base['raw contribution, plain Pearson']['value']))*100:.0f} per cent of why group members move together.",
  finding="With a fitted strength (rho) of 0.04 to 0.07 the copula should barely move a group-spread row. Instead CG went from 9.81 to 4.16 (PR #165) to 0.90 (PR #179). The effect compounds: a shared number held fixed for a whole game and fed through 50 rounds of the models reacting to each other grows to roughly 15 times its one-step prediction. The ablation on PR #179 reads it the same way: the copula supplies free-running variation that the deterministic network cannot generate on its own. That is the same disease the RCB work found, a network that behaves when fed the real human history and goes flat when it plays against the other models.",
  maths=[f"A Bayesian or ensemble treatment (several trained copies of the model, one drawn at random per game) was the standing alternative. It has since been measured and it does not work: five copies disagree by {F(se_dis['sd_E_between_seeds']['mean']):.2f} contribution points per player-round, which translates to a shared-draw strength of {se_sum['implied_rho']:.4f} against the fitted {se_sum['rho_copula_json']:.4f}, and the disagreement halves in about a round rather than lasting a game (PR #188).",
         "The principled version, a per-group random effect fitted jointly with the model, is PR #159: the strength the likelihood allowed reached only 38 percent of the required move.",
         "So the copula stays defensible as a descriptive group-heterogeneity term under three conditions the protocol already enforces or nearly enforces: each player's marginal preserved per draw, strength set by likelihood rather than by the score it improves, and no distortion of individual responses. The third is the one to watch; PR #168 and PR #179 both report the shared number partly deciding <em>who</em> moves, not just how much the group moves.",
         f"What the three follow-ups changed is the reading of what it is <em>for</em>. Its strength is right and is worth about a fifth of the group-spread gap; its persistence, which was never fitted, carries the rest and does so by compounding &mdash; a push of {F(cl_lat['resid_on_z_slope']):.2f} points per round becomes a shift of {F(cl_lat['cell_slope']):.2f} in a group's level (PR #186). On the real games the shared deviation has no lasting part at all: pooled over every pair of rounds two or more apart it is {fmt(F(ms_pers['before partialling']['lag>=2_moment']),3)}, interval {ci(ms_pers['before partialling']['lag>=2_lo'], ms_pers['before partialling']['lag>=2_hi'])} (PR #187)."],
  code=["<code>src/aimanager/generic/copula.py</code> sample_correlated_levels, a per-(game, group) latent with round-to-round persistence", "<code>src/aimanager/simulation/linear_ah.py</code> _sample_levels_copula and _sample_levels_gaussian_copula", "<code>scripts/baselines/punishment_copula_rho.py</code> pairwise maximum-likelihood fit of rho, now with --bundle/--out"],
  bought="The open question at the time was a cheap seed-ensemble test. It has since been run, twice over: the Shared mistakes tab lays out the three experiments that settled it. The short answer is that the copula's persistence is doing a job nothing has yet replaced, that the shape it uses has no counterpart in the human data, and that both settings are now frozen so no future experiment can move them as a side effect. One detail from the re-baseline still fits that reading: the manager's own copula strength rose from 0.35 to 0.43 after the timing fix, which is what a better-specified model leaving residuals that are more purely the manager's shared mood would do."),
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
 dict(id="sharederr", chip="shared error", color="var(--c-teal)", title="Why the simulated players get things wrong together",
  meta=f'<a href="{PR}186">PR #186</a>, <a href="{PR}187">PR #187</a>, <a href="{PR}188">PR #188</a> &middot; three experiments run side by side &middot; all the numbers are on the Shared mistakes tab',
  problem="Two people in the same group face the same situation and then decide for themselves. So once the situation is known, their two choices should be independent of each other, and if the model's <em>mistakes</em> still move together inside a group, the model is missing part of the situation. The models are fitted one player at a time, so nothing in them produces that togetherness; the patch is a shared random number per group per game, mixed into every member's choice. It is calibrated small, about 0.04, and yet it is worth several score bands on the rows that measure how far groups drift apart. That mismatch is what the three experiments were for.",
  finding=f"The shared number is doing two jobs and only one of them is the job it is named for. Redraw it every round and it reproduces the human within-group co-movement almost exactly ({F(cl_dec['C']['resid_corr_all_rounds']):.3f} against {F(cl_dec['human']['resid_corr_all_rounds']):.3f}) while buying about a fifth of the group-spread gap. Hold it for the whole game, which is what ships, and it buys the rest by compounding: a push of {F(cl_lat['resid_on_z_slope']):.2f} contribution points in the round it is drawn becomes a shift of {F(cl_lat['cell_slope']):.2f} in the group's level, {F(cl_lat['compounding_factor']):.1f} times over. Meanwhile the players' own round-by-round randomness is already right &mdash; leftover variance {F(cl_dec['B']['var_resid']):.1f} against the human {F(cl_dec['human']['var_resid']):.1f} &mdash; and what is short is the variety of situations they reach: {F(cl_dec['B']['var_cond_mean']):.1f} against {F(cl_dec['human']['var_cond_mean']):.1f}.",
  maths=[f"Can the missing part of the situation simply be handed to the model? Only a seventh of it. The best three observable group facts &mdash; how many of your group were punished last round, which way the group is drifting, how far apart its members are &mdash; explain {F(ms_joint['mle']['share'])*100:.0f} per cent of the leftover co-movement, interval {F(ms_joint['mle']['share_lo'])*100:.0f} to {F(ms_joint['mle']['share_hi'])*100:.0f} per cent; everything legal together explains {F(ms_ref['all legal candidates']['share_mle'])*100:.0f} per cent.",
         f"Is it a lasting group trait? No. Between two different members of a group the shared deviation is {F(ms_pers['before partialling']['lag0_moment']):.3f} within a round, {F(ms_pers['before partialling']['lag1_moment']):.3f} one round later, and {fmt(F(ms_pers['before partialling']['lag>=2_moment']),3)} pooled over every pair two or more rounds apart, interval {ci(ms_pers['before partialling']['lag>=2_lo'], ms_pers['before partialling']['lag>=2_hi'])}. It is a shock in one round with a two-thirds echo into the next. The shipped setting holds one number for all 24 rounds, which has no counterpart in the data, and is kept only because nothing replaces the variety it supplies.",
         f"Is it the model's own uncertainty? Also no, which closes the Bayesian route. Five copies of the model trained with different random seeds disagree by {F(se_dis['sd_E_between_seeds']['mean']):.2f} contribution points per player-round, an eighth of one model's own spread. That is worth a shared-draw strength of {se_sum['implied_rho']:.4f} against the fitted {se_sum['rho_copula_json']:.4f}, whose interval starts at {se_sum['rho_copula_json_ci'][0]:.3f}, and it decays by half in about a round. Run as a simulation, one copy per game scores like having no machinery at all: group spread {se_sim['CG']:.2f} against {F(cl_sc['CG']['A']):.2f} with it and {F(cl_sc['CG']['B']):.2f} without.",
         "Caveat: five copies trained on the same data is the narrowest kind of ensemble, so that last number is a lower bound. Resampling the games themselves would be wider and has not been tried."],
  code=["<code>scripts/data_analysis/copula_closed_loop_variance.py</code>, the three-arm ablation and the variety diagnostic", "<code>scripts/data_analysis/copula_missing_state.py</code> and its analysis companion", "<code>src/aimanager/simulation/ensemble_ah.py</code> SeedEnsembleAH, one trained copy drawn per game"],
  bought="Three routes closed and one defect named, in a day of cluster time and no new models worth keeping. The two shared-noise settings are now frozen, so no experiment can move them as a side effect of changing something else, and a change to the players is judged with the machinery switched off, on the variety measure, because with it on the row that was being used cannot tell you whether the change helped."),
 dict(id="head", chip="step 1, falsified", color="var(--c-blue)", title="The output design that was supposed to hold the line, and did the opposite",
  meta=f'<a href="{PR}191">PR #191</a> &middot; three short simulations, nothing trained &middot; the hypothesis was the author\'s own',
  problem="A simulated player picks a number from 0 to 20. One family of models scores all 21 possibilities separately; another predicts a centre and a width and draws from a bell curve. The argument for building a combined model around the bell curve went like this: when a simulated game wanders somewhere no real game went, 21 unconnected scores have nothing tying them together and should sag back toward the average, while a single centre keeps tracking. If that were true it would explain the drift, and it would decide which of the two model lines to build on.",
  finding=f"It is wrong in its mechanism and wrong in its consequence. Pushing the recent group level 2, 4 and 6 points away from anything real, the 21-score design tracks the shift most closely of the three and is the only one that does not sag at the extremes ({F(hs_gain[('skip_categorical','-6')]['gain_e']):.2f} to {F(hs_gain[('skip_categorical','6')]['gain_e']):.2f}, against {F(hs_gain[('infl','6')]['gain_e']):.2f} and {F(hs_gain[('v2','6')]['gain_e']):.2f} at the far end). With the shared-noise machinery off it also holds the most variety in the situations it reaches, {F(hs_head['e_skip_kexo_rho0']['var_cond_mean']):.2f} against {F(hs_head['c_infl_rho0']['var_cond_mean']):.2f} and {F(hs_head['d_kexo_rho0']['var_cond_mean']):.2f}. Scored against each model's own fit to real games &mdash; the reading most favourable to the bell curve &mdash; one ties at {F(hs_head['c_infl_rho0']['retention']):.3f} against {F(hs_head['e_skip_kexo_rho0']['retention']):.3f} and the other is clearly worse at {F(hs_head['d_kexo_rho0']['retention']):.3f}.",
  maths=[f"The sharper finding is not the one it was aimed at. Real groups keep drifting further apart as the game runs: the spread of group averages goes {F(cl_blk[('human','1-8')]['sd_group_mean']):.2f}, {F(cl_blk[('human','9-16')]['sd_group_mean']):.2f}, {F(cl_blk[('human','17-24')]['sd_group_mean']):.2f} across the three thirds. With the machinery off, both bell-curve models stall or reverse in the last third ({F(hs_blk[('c_infl_rho0','9-16')]['sd_group_mean']):.2f} to {F(hs_blk[('c_infl_rho0','17-24')]['sd_group_mean']):.2f}; {F(hs_blk[('d_kexo_rho0','9-16')]['sd_group_mean']):.2f} to {F(hs_blk[('d_kexo_rho0','17-24')]['sd_group_mean']):.2f}) where the 21-score design keeps climbing ({F(hs_blk[('e_skip_kexo_rho0','9-16')]['sd_group_mean']):.2f} to {F(hs_blk[('e_skip_kexo_rho0','17-24')]['sd_group_mean']):.2f}). All three fall well short. The defect is a failure of late divergence, not a level offset.",
         f"Within every model, the shared-noise machinery is worth about twice what the choice of output design is worth: the variety measure goes {F(hs_head['c_infl_rho0']['retention']):.3f} to {F(hs_head['c_infl']['retention']):.3f} and {F(hs_head['d_kexo_rho0']['retention']):.3f} to {F(hs_head['d_kexo']['retention']):.3f} when it is switched on, a bigger move than any head difference. So the lever with headroom is something that carries a group's state across rounds, not the shape of one round's output.",
         f"The one confound was closed rather than argued about. The two model lines also used different group-switching components, so a matched pair was run. It moves the 21-score design's variety measure by {F(hs_head['e_skip_kexo_rho0']['retention'])-F(hs_head['skip B (no copula)']['retention']):.3f}, two orders of magnitude less than the gap it was supposed to explain away, and the ordering is unchanged.",
         f"One component of the bell-curve family does earn its place, inside that family: the extra weight it puts on the corners and on repeating last round's number. Without it the plain bell curve emits {(F(hs_recon['human (v2)'][1])/F(hs_recon['human (v2)'][2])-1)*100:.0f} per cent less randomness than its own errors, on real games, turn out to need. A design that already gets the corners for free has nothing to take from it."],
  code=["<code>scripts/data_analysis/head_state_spread.py</code>, five stages, everything regenerable from committed inputs", "<code>plots/data_analysis/evaluation/head_state_spread/</code> headline, gain curves, bootstrap and round blocks"],
  bought="The combined design is dead as argued for, at the cost of three two-minute simulations and no training at all. A negative result that arrives before the build rather than after it is the cheapest thing in this whole record."),
 dict(id="ceiling", chip="step 2, failed its row", color="var(--c-red)", title="Fixing the manager at the ceiling found the fault was in the players",
  meta=f'<a href="{PR}192">PR #192</a> &middot; one flag added to both manager models &middot; two short retrainings and two simulations',
  problem=f"Real managers almost never punish someone who gave the full 20 &mdash; {F(ce_mech['human']['P(p>0|c_t=20)'])*100:.1f} per cent of the time &mdash; and when they do they punish hard, {F(ce_mech['human']['E[p|p>0] 20']):.1f} points on average. Both simulated managers read the amount given as a single number on a scale, so neither can make a sharp break at exactly 20; they read the ceiling off the 15-to-19 band just below it. The result was that the simulation punished full contributors {F(ce_mech['frontier before']['P(p>0|c_t=20)'])*100:.1f} per cent of the time and too lightly. One row of the score card is built entirely out of punished full contributors, so the simulation was inventing the very population that row measures, and that row was the one thing the manager timing fix had not moved.",
  finding=f"The flag works and the experiment still fails. In the simulated games the punish rate at the ceiling goes to {F(ce_mech['frontier after']['P(p>0|c_t=20)']):.3f} against the real {F(ce_mech['human']['P(p>0|c_t=20)']):.3f} &mdash; right to a thousandth &mdash; and the severity there from {F(ce_mech['frontier before']['E[p|p>0] 20']):.2f} to {F(ce_mech['frontier after']['E[p|p>0] 20']):.2f} against the real {F(ce_mech['human']['E[p|p>0] 20']):.2f}. The target row moved {fmt(F(ce_ba['RCC']['frontier_after'])-F(ce_ba['RCC']['frontier_before']),3,sign=True)}, from {F(ce_ba['RCC']['frontier_before']):.4f} to {F(ce_ba['RCC']['frontier_after']):.4f}, the largest move it has ever had, and did not cross a score band. The declared target was a band, so the verdict is a failure.",
  maths=[f"Splitting the row into its parts says where the remaining distance is, and it is not the manager's. The invented population is gone: {ce_rcc['frontier before'][6]} of full contributors punished before, {ce_rcc['frontier after'][6]} after, {ce_rcc['human'][6]} in the real games. What is left is that a punished full contributor in the simulation gives up {abs(F(ce_rcc['frontier after'][2])):.2f} points the next round where a real person gives up {abs(F(ce_rcc['human'][2])):.2f}. That is the players under-reacting by about {abs(F(ce_rcc['human'][2]))/abs(F(ce_rcc['frontier after'][2])):.1f} times, and no change to the manager can touch it.",
         f"A second defect is untouched and now separately live: the simulated manager's punishment falls with contribution at {fmt(F(ce_mech['frontier after']['OLS c_t']),3)} per point against the real {fmt(F(ce_mech['human']['OLS c_t']),3)}, a little over half the human strength. The flag changes this by at most {max(abs(F(t[a]['OLS c_t'])-F(t[b]['OLS c_t'])) for t,a,b in ((ce_tf,'lin new (c_t)','lin ceiling'),(ce_tf,'gnn new (c_t)','gnn ceiling'),(ce_mech,'frontier before','frontier after'),(ce_mech,'ref_lin before','ref_lin after'),(ce_mech,'ref_gnn before','ref_gnn after'))):.3f} in any condition and was never meant to. Nobody should read &ldquo;the manager was fixed at the ceiling&rdquo; as &ldquo;the manager's response to contribution was fixed&rdquo;.",
         f"What went the wrong way: the older reaction-to-punishment row got worse, {F(ce_ba['RCB']['frontier_before']):.3f} to {F(ce_ba['RCB']['frontier_after']):.3f}, precisely because removing the punishments at the ceiling removed the rows where its denominator was smallest, leaving it shaped entirely by the slope defect above. Group spread also rose on this stack while falling on the reference one.",
         f"On the real games the flag is worth {fmt(F(ce_logit['+max']['contribution_max_coef']))} on the log-odds scale and takes the fitted rate at the ceiling from {F(ce_logit['linear']['fit_P(p>0|c=20)']):.3f} onto the observed {F(ce_logit['+max']['fit_P(p>0|c=20)']):.3f}; the fit over all punishment levels improves from {F(ce_cv0['log_loss']):.4f} to {F(ce_cv['log_loss']):.4f}. A companion flag for &ldquo;gave nothing&rdquo; was tested and dropped."],
  code=["<code>scripts/data_analysis/punisher_ceiling_check.py</code>, the diagnosis on the real games before anything was built", "<code>scripts/baselines/handcrafted_grid.py</code> the derived indicator and its legality, <code>src/aimanager/generic/data.py</code> and <code>manager/api_manager.py</code> the same on the graph path"],
  bought=f"The manager's half of the row is finished and the other half is named, with a decomposition table ready to serve as the baseline for whoever declares against the players. The graph-network reference stack improved broadly on the way past: mean {ce_mean('ref_gnn_before'):.4f} to {ce_mean('ref_gnn_after'):.4f}, rows at or under the ceiling {ce_le1('ref_gnn_before')} to {ce_le1('ref_gnn_after')}. Whether artifacts from a failed experiment are worth merging anyway is a maintainer's call; the mechanism evidence says yes and the gate says no."),
 dict(id="switchport", chip="step 3, failed", color="var(--c-amber)", title="The right number of switches, the wrong people leaving",
  meta=f'<a href="{PR}190">PR #190</a> &middot; one component swapped, one two-minute simulation &middot; no retraining',
  problem="Every fourth round, players may move to the other group. Real people never empty a group of five or more and often abandon a singleton, so group size matters in a lumpy way that one smooth curve cannot follow. The other model line has a component that gives each possible group size its own free setting, and the stack built around it posts the best switching numbers in the whole set. The question was whether those numbers belong to that component or to the players it was paired with. Swapping only the component across is the cheapest way to find out.",
  finding=f"Every measure that is purely about who ends up in which group improved, and every measure of how players respond got worse. Switch timing gained a score band, {F(sk_cmp['SB']['before']):.3f} to {F(sk_cmp['SB']['after']):.3f}; group spread fell sharply, {F(sk_cmp['CG']['before']):.3f} to {F(sk_cmp['CG']['after']):.3f}; segregation improved without crossing a band, {F(sk_cmp['SC']['before']):.3f} to {F(sk_cmp['SC']['after']):.3f}. The switching-pull row went the wrong way, {F(sk_cmp['RCD']['before']):.3f} to {F(sk_cmp['RCD']['after']):.3f}, and the protected response row was violated: in the 10-to-14 band the slope fell from {fmt(F(sk_rce['before']['slope_10-14']),3)} to {fmt(F(sk_rce['after']['slope_10-14']),3)}, about a third of what it was. The 22-row mean is the lowest on record at {F(sk_cmp['mean']['after']):.4f} and the experiment still fails, because the mean is not what it declared.",
  maths=["The row split is the useful result. Only segregation, switch timing and group spread are decided by the switching component alone. The switching-pull row is the slope of a <em>switcher's contribution change</em> on the gap to the group they join: the component picks who goes, the players decide what they then give. Declaring that row as a target was a mis-specification in the plan itself, independent of how the run came out.",
         f"Undeclared and the largest single regression: switching after being punished, {F(sk_cmp['RSA']['before']):.3f} to {F(sk_cmp['RSA']['after']):.3f}. Right number of switches, right group sizes, wrong people leaving after a punishment. That is a concrete mismatch and the most informative follow-up in the set.",
         f"One seed, one run, no repeats, so the small movements carry no measured spread. The gate miss is safe &mdash; the switching-pull row moves by {abs(F(sk_cmp['RCD']['delta'])):.2f} and neither target comes near a band edge &mdash; but the protected-row violation rests on a single run's band slope over {int(F(sk_rce['after']['n_10-14'])):,} observations, and anyone wanting to overturn it should refit that band across seeds rather than argue about it.",
         "What it cannot settle: whether the component needs the other line's players or clashes with these ones specifically. Both fit this run and they imply different successors. Pairing it with a third set of players separates them, at one simulation each."],
  code=["<code>src/aimanager/generic/joint_exodus.py</code>, the size encoding ported across with a default so older components keep loading", "<code>plots/data_analysis/evaluation/switch_kexo_port/</code> the 22-row comparison and the band slopes"],
  bought="A rule for every future switching experiment: declare only the rows the component actually decides. And one clear counter-example to the premise the whole lineage merge rested on, that the two lines' strengths are separable and additive."),
 dict(id="protocol", chip="step 4, repaired", color="var(--c-ink)", title="A safety rule that failed two experiments it should not have",
  meta=f'<a href="{PR}189">PR #189</a> &middot; not an experiment &middot; the plan and the rules it is judged by',
  problem="One row of the score card is protected, because it is the one thing a learning manager will depend on: how much more a player gives per extra point of punishment. No experiment may worsen its score band, flip any of the four human signs, or halve a band's slope. The last clause is a magnitude test on a signed quantity, and the bands it guards are thin &mdash; one of them, the middle range, has been known since the held-out test to have the wrong sign in every condition, learned or not.",
  finding=f"It fired twice on its first outing, and both firings were wrong. Once on a change of {ce_se[1]['change_in_se']['15-19']:.2f} standard errors on the thinnest band in the suite, which on a single run is not distinguishable from noise. Once on a slope moving from {fmt(lead(ce_sl[2]['before'][3]),3,sign=True)}, the wrong sign, to {fmt(lead(ce_sl[2]['after'][3]),3,sign=True)}, toward the human {fmt(HUMAN_SLOPES[2],3,sign=True)} &mdash; a change of {ce_se[2]['change_in_se']['10-14']:.2f} standard errors, and an improvement read as an erosion, because a magnitude test cannot tell a slope passing through zero in the right direction from one wasting away.",
  maths=["Two qualifications were added. The clause does not fire when the new slope is closer to the human value than the old one was, and it fires only when the change exceeds one pooled standard error of the two slopes. Neither changes a verdict already recorded: both failing experiments failed their declared row independently, so the repair cannot be read as rescuing anything.",
         "Every experiment that touches the row now reports each band's slope with its standard error, its row count and the change in pooled standard errors, so a reader can tell erosion from noise without re-running anything.",
         "The other half of this step is the freeze. The two shared-noise settings are added to the surface no experiment may modify, and a change to the players is judged with the machinery switched off, on the variety measure. Before the freeze, a recalibration rode along with every change to the players, so a player experiment and a noise experiment moved at once and could not be told apart."],
  code=["<code>notes/autoresearch.md</code> sections 2 and 8, the protected row, the freeze and the frozen surface", "<code>doc/plans/post-rebaseline-program.md</code>, the four steps as declared and then as they came out"],
  bought="A rule that fails experiments over differences too small to be real, and once over an improvement, is worse than no rule, because it teaches the people it governs to argue with it instead of respecting it. It is now stated with the arithmetic that makes a firing readable."),
 dict(id="next", chip="successor", color="var(--c-muted)", title="What this leaves for whoever continues",
  meta="one large defect and three small ones, each with a baseline ready",
  problem=f"The target is the late-divergence failure. Real groups keep pulling apart as a game runs and the models stop: with the shared-noise machinery switched off, the variety of situations the simulation reaches is {F(cl_dec['B']['var_cond_mean']):.1f} against the human {F(cl_dec['human']['var_cond_mean']):.1f}, while the randomness inside each round is already correct. Neither the output design nor the noise model is the lever; both have been tested and neither is. What is wanted is something that carries a group's state across rounds and survives the models playing against each other.",
  finding="",
  maths=[f"<b>The players under-react to a heavy punishment at the ceiling.</b> A punished full contributor gives up {abs(F(ce_rcc['frontier after'][2])):.2f} points next round where a real person gives up {abs(F(ce_rcc['human'][2])):.2f}. The manager's side of that row is finished, the population is now the right size, and the decomposition table is the baseline. Only one row in the suite measures it, so it has nowhere else to show up.",
         f"<b>The manager's response to contribution is about half the human strength</b>, {fmt(F(ce_mech['frontier after']['OLS c_t']),3)} per point against {fmt(F(ce_mech['human']['OLS c_t']),3)}. Nothing so far has moved it. It wants a bent response rather than another flag, and it is the row the older reaction-to-punishment measure would most plausibly follow.",
         f"<b>The wrong people leave after being punished.</b> The borrowed switching component gets the number of switches and the group sizes right and regressed that row hardest, {F(sk_cmp['RSA']['before']):.3f} to {F(sk_cmp['RSA']['after']):.3f}. It is diagnosable and cheap.",
         "Deliberately out of scope: the three group facts from the missing-state experiment are worth folding into the next retrain of the players but account for a seventh of a small quantity and do not justify a cycle of their own. Rollout training is the wrong tool for this defect, for a reason that is now understood.",
         "Still open and not closable by anything above: the middle contribution band has the wrong sign in every condition, including when the model is fed real games and when it is held out, so no closed-loop fix will supply it. And the manager's room to act &mdash; real managers rarely punished above 10 points or punished high contributors, about 300 rows of evidence in total &mdash; must either be bounded or audited before a learning manager explores there.",
         "Bookkeeping that would otherwise mislead: the 32-stack sweep has not been rerun under the fixed manager, so the deficit profiles in the experiment record are all from before it."],
  code=["<code>doc/plans/post-rebaseline-program.md</code>, the four steps and their results in one place", "<code>notes/autoresearch_log/</code>, one log per experiment, each with its own successor section", "Raven clean-up as each pull request closes: one isolated folder per experiment under ~/repros/ai-runs/"],
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

# ---------- the shared-error tab ----------
def shared_error_section():
    A, Bm, C, H = cl_dec["A"], cl_dec["B"], cl_dec["C"], cl_dec["human"]
    arms = tbl(["what was measured", "~real people", "~machinery on", "~machinery off", "~number redrawn each round"], [
        ["how varied the situations the players reach", fmt(F(H["var_cond_mean"]),1), fmt(F(A["var_cond_mean"]),1), f'<b>{fmt(F(Bm["var_cond_mean"]),1)}</b>', fmt(F(C["var_cond_mean"]),1)],
        ["how much randomness is left in one round", fmt(F(H["var_resid"]),1), fmt(F(A["var_resid"]),1), fmt(F(Bm["var_resid"]),1), fmt(F(C["var_resid"]),1)],
        ["how far apart the group averages sit", fmt(F(H["sd_group_mean"])), fmt(F(A["sd_group_mean"])), fmt(F(Bm["sd_group_mean"])), fmt(F(C["sd_group_mean"]))],
        ["leftover co-movement inside a group", fmt(F(H["resid_corr_all_rounds"]),3), fmt(F(A["resid_corr_all_rounds"]),3), fmt(F(Bm["resid_corr_all_rounds"]),3), f'<b>{fmt(F(C["resid_corr_all_rounds"]),3)}</b>'],
        ["the group-spread score CG", "&ndash;", fmt(F(cl_sc["CG"]["A"])), fmt(F(cl_sc["CG"]["B"])), fmt(F(cl_sc["CG"]["C"]))],
        ["mean over all 22 rows", "&ndash;", fmt(F(cl_sc["mean"]["A"]),3), fmt(F(cl_sc["mean"]["B"]),3), fmt(F(cl_sc["mean"]["C"]),3)],
    ])
    pb = ms_pers["before partialling"]
    echo = tbl(["how far apart the two rounds are", "~shared co-movement", "~95% interval"], [
        ["the same round", fmt(F(pb["lag0_moment"]),3), ci(pb["lag0_lo"], pb["lag0_hi"])],
        ["one round apart", fmt(F(pb["lag1_moment"]),3), ci(pb["lag1_lo"], pb["lag1_hi"])],
        ["two rounds apart", fmt(F(pb["lag2_moment"]),3), ci(pb["lag2_lo"], pb["lag2_hi"])],
        ["<b>two or more, pooled over every pair</b>", f'<b>{fmt(F(pb["lag>=2_moment"]),3)}</b>', ci(pb["lag>=2_lo"], pb["lag>=2_hi"])],
        ["share of it that is a lasting group trait", fmt(F(pb["static_share"])), ci(pb["static_share_lo"], pb["static_share_hi"], 2)],
    ])
    lag = se_sum["persistence_lag_corr"]
    seeds = tbl(["what was measured", "~value"], [
        ["how far apart the five copies' predictions sit, per player and round", f'{F(se_dis["sd_E_between_seeds"]["mean"]):.2f} points'],
        ["how wide one copy's own prediction already is, same rounds", f'{F(se_dis["sd_pred_ensemble_mean"]["mean"]):.2f} points'],
        ["so the disagreement is, as variance", f'{se_sum["ratio_var_E_to_var_pred"]*100:.0f}% of what one model already spreads'],
        ["is the disagreement shared inside a group?", f'yes: {se_sum["corr_E_mean"]:.2f} on the choice scale, {se_sum["corr_latent_mean"]:.2f} on the fitted scale'],
        ["<b>what that is worth as a shared-draw strength</b>", f'<b>{se_sum["implied_rho"]:.4f}</b> by arithmetic, {se_sum["rho_synthetic_mean"]:.4f} when measured the way the real one is'],
        ["the strength actually fitted, for comparison", f'{se_sum["rho_copula_json"]:.4f}, interval {se_sum["rho_copula_json_ci"][0]:.3f} to {se_sum["rho_copula_json_ci"][1]:.3f}'],
        ["how long the disagreement lasts, 1 / 2 / 3 / 5 / 10 rounds on", " / ".join(f'{lag[k]:.2f}' for k in ("1","2","3","5","10"))],
        ["how much of it belongs to the group and game at all", f'{se_sum["persistence_icc_episode"]*100:.0f}%'],
    ])
    return f'''
<p class="legend">Three experiments, run side by side, asking the same question three ways: when the simulated players get something wrong, why do the members of one group get it wrong <em>together</em>? Each one is laid out below as an answer, with what it measured and what it rules out. This is the clearest thing on this page, and it needs no background beyond the game.</p>

<h2>The principle</h2>
<div class="twocol">
<div>
<p class="legend" style="color:var(--ink)">Two people in the same group face the same situation and then decide for themselves. So once you know the situation &mdash; what everyone gave last round, who was punished, how big the group is &mdash; their two choices should be independent of each other. Turn that around and it becomes a test. If the model's <em>mistakes</em> still move together within a group after the situation has been accounted for, then the model is missing part of the situation. Naming the missing part is the whole question.</p>
</div>
<div>
<p class="legend" style="color:var(--ink)">The models are fitted one player at a time, so nothing in them makes group members move together. The patch in use is a <em>shared draw</em>: one random number per group per game, mixed into every member's choice so that they lean the same way. Two settings control it. Its <b>strength</b> is fitted from how much of the model's leftover error is genuinely shared, and comes out small, about {F(ms_base["latent, pairwise MLE (40 train games)"]["value"]):.3f}. Its <b>persistence</b> is separate and was never fitted: the shipped version holds the same number for all 24 rounds.</p>
</div>
</div>

<h2>Answer one: it is mostly not a correlation at all</h2>
<p class="legend">{esc("PR #186")} ran the same simulation three times &mdash; with the machinery on, with it off, and with the shared number redrawn every round instead of held for the whole game &mdash; and then asked the model what it would have expected at each point it actually reached.</p>
{arms}
<p class="legend">Read across the bottom two rows first: switching the machinery off costs a whole score band on group spread. Now read the top two. The players' round-by-round randomness is <b>already correct</b> &mdash; the leftover variance is {fmt(F(Bm["var_resid"]),1)} against the human {fmt(F(H["var_resid"]),1)}, and it barely moves between arms. What is short is the variety of situations the simulated games reach: {fmt(F(Bm["var_cond_mean"]),1)} against {fmt(F(H["var_cond_mean"]),1)}, about two thirds. Redrawing the number every round reproduces the human within-group co-movement almost exactly ({fmt(F(C["resid_corr_all_rounds"]),3)} against {fmt(F(H["resid_corr_all_rounds"]),3)}) and yet buys only about a fifth of the group-spread gap. Holding it for the whole game buys the rest, and it does so by <em>compounding</em>: the shared number pushes a player by {F(cl_lat["resid_on_z_slope"]):.2f} points in the round it is drawn, but because the players react to each other round after round, a group's level ends up shifted by {F(cl_lat["cell_slope"]):.2f} points &mdash; {F(cl_lat["compounding_factor"]):.1f} times the push.</p>
<p class="legend"><b>What this rules out.</b> The reading that the machinery is simply supplying a missing correlation. It supplies a correlation of about the right size, and that part is worth a fifth of one row. The rest of its value is refilling variety that the players fail to generate when they play against each other &mdash; which is a defect in the players, not a missing sampler.</p>

<h2>Answer two: the missing part is a one-round echo, not a group trait</h2>
<p class="legend">{esc("PR #187")} stayed on the real games. It asked the model what it expected round by round, took what was left over, and tried to explain the shared part of it with group facts the model cannot currently see: how many of your group were punished last round, which way the group is drifting, how far apart its members are.</p>
<div class="twocol">
<div>
<p class="legend" style="color:var(--ink)">In the raw data, two members of a group move together strongly: {fmt(F(ms_base["raw contribution, plain Pearson"]["value"]),3)}. Once the model's own expectation is subtracted, {fmt(F(ms_base["level residual, plain Pearson"]["value"]),3)} is left &mdash; the model already explains {(1-F(ms_base["level residual, plain Pearson"]["value"])/F(ms_base["raw contribution, plain Pearson"]["value"]))*100:.0f} per cent of it. Of that remainder, the best three observable group facts explain {F(ms_joint["mle"]["share"])*100:.0f} per cent, interval {F(ms_joint["mle"]["share_lo"])*100:.0f} to {F(ms_joint["mle"]["share_hi"])*100:.0f} per cent; every legal fact together explains {F(ms_ref["all legal candidates"]["share_mle"])*100:.0f} per cent. The two that carry it are the share of the group punished last round ({F(ms_cand["grp_share_pun_last"]["share_mle"])*100:.0f} per cent) and the group's contribution trend ({F(ms_cand["own_trend"]["share_mle"])*100:.0f} per cent). Adding all three to the model would move the strength, measured here over all fifty games rather than the forty it was fitted on, from {F(ms_joint["mle"]["rho_before"]):.4f} to about {F(ms_joint["mle"]["rho_after"]):.3f} &mdash; well inside its own interval. They are worth having for what they say about behaviour, not for the dose.</p>
</div>
<div>
{echo}
<p class="legend">Shared co-movement between two <em>different</em> players of the same group, at increasing distances in time. A lasting group trait would give the same value at every distance.</p>
</div>
</div>
<p class="legend"><b>What this rules out.</b> Two things. First, that the fix is simply to feed the model the group facts it is missing: they account for a seventh of a quantity that is already small. Second, and more important, the <em>shape</em> the machinery uses. In the real games the shared deviation is a shock in one round with about a two-thirds echo into the next ({F(pb["phi_lag1"]):.2f}) and nothing at all beyond that. The shipped setting holds one number fixed for 24 rounds, and there is no counterpart to that in the human data. It is kept anyway, because nothing yet replaces the variety it supplies &mdash; a caveat, not an endorsement.</p>

<h2>Answer three: the model's uncertainty about itself is far too small</h2>
<p class="legend">The standing proposal was to replace the hand-set number with the model's own uncertainty: train several copies of the model, draw one at random per game, and let the size of the shared error come from how much the copies disagree. {esc("PR #188")} trained five copies, identical but for the random seed, and measured it.</p>
{seeds}
<p class="legend">The copies do disagree in a shared way &mdash; members of one group are pushed together, which is the right shape. But the size is about a sixth of the fitted strength and sits below the bottom of its confidence interval, and the time structure is wrong: the disagreement halves in roughly a round, because the same fixed weights meet a different situation each round, where the machinery needs something that holds for the whole game. Run as an actual simulation, one copy per game scores like having no machinery at all: group-spread score {se_sim["CG"]:.2f}, against {F(cl_sc["CG"]["A"]):.2f} with the machinery and {F(cl_sc["CG"]["B"]):.2f} with it switched off, and the 22-row mean rises to {se_mean:.3f} with {se_le1} rows at or under the ceiling instead of {int(F(cl_sc["rows <= 1"]["A"]))}.</p>
<p class="legend"><b>What this rules out.</b> The Bayesian route, at least as a five-seed ensemble measures it. For this to supply what the machinery supplies, the spread over trained copies would have to be six to eight times wider <em>and</em> persistent across a whole game, which a fixed set of weights meeting drifting situations is not. Caveat: five seeds on the same data is the narrowest kind of ensemble, so this is a lower bound; resampling the games themselves would be wider, and has not been tried.</p>

<h2>What all three leave standing</h2>
<p class="legend" style="max-width:76ch;font-size:14px;color:var(--ink)">One defect. The simulated games do not reach the variety of situations the real ones do &mdash; {fmt(F(Bm["var_cond_mean"]),1)} against {fmt(F(H["var_cond_mean"]),1)} &mdash; while the randomness inside each round is already right. Everything the shared draw was doing beyond its small honest job was covering that up. Three of the 22 rows are symptoms of it: how far the groups drift apart, how strongly a switcher is pulled toward the group they join, and the flattened reaction to punishment when the models play each other. The next tab shows what that defect actually looks like: real groups keep pulling apart through the last third of a game, and the models stop.</p>
'''

# ---------- the four-step tab ----------
def four_steps_section():
    steps = tbl(["step", "what was tried", "verdict"], [
        ["1 &middot; the output design", "whether a different way of producing a number resists the drift better", '<span class="pill">hypothesis falsified</span>'],
        ["2 &middot; the manager at the ceiling", "an indicator telling the manager a player gave the maximum", '<span class="pill">failed its declared row</span>'],
        ["3 &middot; the borrowed switch component", "the other model line's group-switching component, dropped into this one", '<span class="pill">failed, and failed the protected row</span>'],
        ["4 &middot; the rules", "freeze the noise settings; judge players with the machinery off", '<span class="pill">done, then repaired</span>'],
    ])
    b, hd = hs_boot, hs_head
    ret = tbl(["players", "~variety of situations reached", "~as a share of what the same model manages on real games", "~95% interval"], [
        ["categorical, the current design", fmt(F(hd["e_skip_kexo_rho0"]["var_cond_mean"]),2), f'<b>{F(hd["e_skip_kexo_rho0"]["retention"]):.3f}</b>', ci(b["e_skip_kexo_rho0"]["ret_lo"], b["e_skip_kexo_rho0"]["ret_hi"])],
        ["inflated Gaussian", fmt(F(hd["c_infl_rho0"]["var_cond_mean"]),2), f'{F(hd["c_infl_rho0"]["retention"]):.3f}', ci(b["c_infl_rho0"]["ret_lo"], b["c_infl_rho0"]["ret_hi"])],
        ["plain Gaussian", fmt(F(hd["d_kexo_rho0"]["var_cond_mean"]),2), f'{F(hd["d_kexo_rho0"]["retention"]):.3f}', ci(b["d_kexo_rho0"]["ret_lo"], b["d_kexo_rho0"]["ret_hi"])],
    ])
    mech = tbl(["", "~punished after giving everything", "~how hard, when punished at the ceiling", "~punishment falls with contribution by"], [
        [lab, fmt(F(ce_mech[k]["P(p>0|c_t=20)"]),3), fmt(F(ce_mech[k]["E[p|p>0] 20"]),2), fmt(F(ce_mech[k]["OLS c_t"]),3, sign=True)]
        for k, lab in (("human", "<b>real managers</b>"), ("frontier before", "the simulated manager, before"),
                       ("frontier after", "the simulated manager, with the indicator"),
                       ("ref_gnn before", "the graph-network manager, before"), ("ref_gnn after", "the graph-network manager, after"))
    ], cls="mech")
    d = ce_rcc
    dec = tbl(["", "~next-round change if punished", "~how many such players", "~next-round change if not punished", "~share punished"], [
        ["<b>real people</b>", fmt(F(d["human"][2])), d["human"][3], fmt(F(d["human"][4])), d["human"][6]],
        ["the simulation, before", fmt(F(d["frontier before"][2])), d["frontier before"][3], fmt(F(d["frontier before"][4])), d["frontier before"][6]],
        ["the simulation, with the indicator", f'<b>{fmt(F(d["frontier after"][2]))}</b>', d["frontier after"][3], fmt(F(d["frontier after"][4])), f'<b>{d["frontier after"][6]}</b>'],
    ], cls="mech")
    att = tbl(["row", "~the component in its own model line", "~the component moved over here", "~this line as it was"], [
        [f'{r} <span class="rn">{esc(ROWNAME[r])}</span>', fmt(scores[r]["d_kexo_after"]), fmt(F(sk_cmp[r]["after"])), fmt(F(sk_cmp[r]["before"]))]
        for r in ("SC", "SB", "CG", "RCD", "RCE", "RSA")
    ])
    return f'''
<p class="legend">After the three answers above, four things were done next, all declared in writing before any of them ran: three experiments and one change to the rules by which experiments are judged. Each block says what was tried, what happened and what it means. Two of the three experiments failed the bar they had set themselves. The third set no bar, because it was a measurement rather than a candidate, and it came back against the hypothesis its own author had proposed. Those are the results worth reading.</p>
{steps}

<h2>Step 1 &mdash; the output design, and what it found instead</h2>
<div class="twocol">
<div>
<p class="legend" style="color:var(--ink)"><b>What was tried.</b> The players pick a number from 0 to 20. One family of models scores all 21 possibilities separately; another predicts a centre and a width and draws from a bell curve. The argument for the second was that when a simulated game wanders somewhere no real game went, a model with 21 unconnected scores has nothing holding them together and should sag back toward its average, while one that shifts a single centre keeps tracking. If true, that would explain the drift, and the two model lines should be merged around the bell-curve design. No training was needed &mdash; three short simulations and a probe over the real games.</p>
<p class="legend" style="color:var(--ink)"><b>What happened.</b> The opposite, on every measurement. Pushing the recent group level 2, 4 and 6 points away from anything real, the 21-score design tracks the shift most closely of the three and is the only one that does not sag at the extremes. With the shared-noise machinery off, it also holds the most variety in the situations it reaches. Reading it the way most favourable to the bell-curve models &mdash; scoring each against its own fit to real games &mdash; one ties and one is clearly worse.</p>
<p class="legend" style="color:var(--ink)"><b>What it means.</b> The combined design is dead as argued for. One piece of the bell-curve family still earns its place inside that family: the extra weight it puts on the corners and on repeating last round's number, which is what keeps its randomness honest. But it has nothing to offer a design that gets the corners for free.</p>
</div>
<div>
<h4>How much each model still moves when pushed away from real situations</h4>
{gain_svg()}
<p class="legend">Bars hang from the line at 1.00, which is where a model shifts its prediction one-for-one with the push; the shorter the bar, the better the model tracks. The six groups are the six pushes, from 6 points down to 6 points up, over the same set of player-rounds for all three models. The 21-score design tracks most closely everywhere and is the only one that rises above the line at the extremes. Neither bell-curve model decays with distance either, so nothing here fails to extrapolate.</p>
{keyline(GAIN_SERIES)}
</div>
</div>
<h4>Variety of situations reached, with the shared-noise machinery off</h4>
{ret}
<p class="legend">All three are short of the human {fmt(F(cl_dec["human"]["var_cond_mean"]),1)}. The share column divides each model by what that same model manages when it is fed real games, so a model is not penalised for being a worse fit in the first place; on that reading the inflated bell-curve model ties with the current design and the plain one is worse.</p>

<div class="twocol">
<div>
<h4>The sharper finding: the groups stop pulling apart</h4>
<p class="legend" style="color:var(--ink)">The same measurements produced a better description of the defect than the one they were aimed at. Real groups keep drifting further apart as a game runs, all the way to the last round. With the shared-noise machinery off, every model starts in about the right place and then stalls or reverses in the last third. So the drift is not a model that is uniformly too tame; it is a model that stops accumulating differences after about round 16. That is a missing slow process, not a missing output design &mdash; and it is why the next round of work is aimed at something that can carry a group's state across rounds.</p>
<p class="legend">Caveat kept from the experiment: within every model, the shared-noise machinery is worth about twice what the choice of output design is worth, and the earlier work showed all of that value sits in its persistence rather than its correlation.</p>
</div>
<div>
<h4>How far apart the groups drift, by third of the game</h4>
{spread_svg()}
<p class="legend">Standard deviation of group averages, real people against three sets of simulated players with the shared-noise machinery switched off. Labels on the last third.</p>
{keyline(SPREAD_SERIES)}
</div>
</div>

<h2>Step 2 &mdash; the manager at the contribution ceiling</h2>
<p class="legend"><b>What was tried.</b> Real managers almost never punish someone who gave the full 20, and when they do they punish hard. Both simulated managers treat the amount given as one number on a scale, so neither can make a sharp break at exactly 20; they read the ceiling off the 15-to-19 band and punish full contributors three to four times too often, too lightly. The row that scores the reaction at the ceiling was the one thing the manager fix had not moved. So both managers got one extra input: a yes-or-no flag saying the player gave the maximum. On the real games the flag is worth {fmt(F(ce_logit["+max"]["contribution_max_coef"]))} on the log-odds scale ({esc("p = ")}{F(ce_logit["+max"]["contribution_max_p"]):.3f}) and takes the fitted rate at the ceiling from {fmt(F(ce_logit["linear"]["fit_P(p>0|c=20)"]),3)} onto the observed {fmt(F(ce_logit["+max"]["fit_P(p>0|c=20)"]),3)}; the model's fit over all punishment levels improves from {F(ce_cv0["log_loss"]):.4f} to {F(ce_cv["log_loss"]):.4f}. A companion flag for &ldquo;gave nothing&rdquo; was tested and dropped.</p>
{mech}
<p class="legend"><b>What happened.</b> The mechanism is now essentially exact. In the simulated games the manager punishes a full contributor {fmt(F(ce_mech["frontier after"]["P(p>0|c_t=20)"]),3)} of the time against the real {fmt(F(ce_mech["human"]["P(p>0|c_t=20)"]),3)}, where before it was {fmt(F(ce_mech["frontier before"]["P(p>0|c_t=20)"]),3)}; the severity there goes from {fmt(F(ce_mech["frontier before"]["E[p|p>0] 20"]),2)} to {fmt(F(ce_mech["frontier after"]["E[p|p>0] 20"]),2)} against the real {fmt(F(ce_mech["human"]["E[p|p>0] 20"]),2)}. The target row moved {fmt(F(ce_ba["RCC"]["frontier_after"])-F(ce_ba["RCC"]["frontier_before"]),3, sign=True)}, from {fmt(F(ce_ba["RCC"]["frontier_before"]),3)} to {fmt(F(ce_ba["RCC"]["frontier_after"]),3)} &mdash; the largest move that row has ever had &mdash; and still did not cross a score band, so the experiment is recorded as a failure. The 22-row mean is flat at {ce_mean("frontier_after"):.4f} against {ce_mean("frontier_before"):.4f} and the protected response row holds. The graph-network reference stack improved more broadly: mean {ce_mean("ref_gnn_before"):.4f} to {ce_mean("ref_gnn_after"):.4f}, rows at or under the ceiling {ce_le1("ref_gnn_before")} to {ce_le1("ref_gnn_after")}.</p>
{dec}
<p class="legend"><b>What it means.</b> Splitting the target row into its parts says exactly where the remaining distance sits, and it is not the manager's. The manager's half is finished: the invented population of punished full contributors is gone, {d["frontier before"][6]} before against {d["frontier after"][6]} after and {d["human"][6]} in the real games. What is left is the players. A punished full contributor in the simulation gives up {fmt(abs(F(d["frontier after"][2])),2)} points the next round where a real person gives up {fmt(abs(F(d["human"][2])),2)} &mdash; an under-reaction of about {abs(F(d["human"][2]))/abs(F(d["frontier after"][2])):.1f} times, which no change to the manager can touch. A second defect is untouched and stays live: the simulated manager's punishment falls with contribution at {fmt(F(ce_mech["frontier after"]["OLS c_t"]),3)} per point against the real {fmt(F(ce_mech["human"]["OLS c_t"]),3)}, a little over half the human strength, and the indicator does not move it.</p>

<h2>Step 3 &mdash; borrowing the other line's group-switching component</h2>
<p class="legend"><b>What was tried.</b> Two model lines have been developed in parallel. The other one posts the best switching numbers in the whole set, and the piece most likely to be responsible is its group-switching component, which gives each possible group size its own free setting instead of forcing one smooth curve through them. The test was cheap: change nothing but that one piece and run the simulation once.</p>
{att}
<p class="legend"><b>What happened.</b> Every measure that is purely about who ends up in which group improved, and every measure of how players respond got worse. Switch timing gained a score band, segregation improved without crossing one, the group-spread row improved sharply, and the switching-pull row moved the wrong way. The protected response row was violated: in the 10-to-14 band the slope fell from {fmt(F(sk_rce["before"]["slope_10-14"]),3)} to {fmt(F(sk_rce["after"]["slope_10-14"]),3)}, about a third of what it was. The 22-row mean is the lowest on record at {F(sk_cmp["mean"]["after"]):.4f} and it still fails, because the mean is not what it declared.</p>
<p class="legend"><b>What it means.</b> The split in the table is the useful result. Only the first three rows are decided by the switching component alone; the last three depend on what the players do once the groups have been set, so they were never the right targets. Declaring the switching-pull row as a target was a mistake in the plan itself, independent of how the run came out. One further thing went wrong that nobody had declared: switching after being punished regressed more than any other row, {fmt(F(sk_cmp["RSA"]["before"]),3)} to {fmt(F(sk_cmp["RSA"]["after"]),3)}. Right number of switches, right group sizes, wrong people leaving. One run on one seed cannot separate whether the component needs the other line's players or clashes with these ones specifically.</p>

<h2>Step 4 &mdash; changing the rules, and then repairing them</h2>
<div class="twocol">
<div>
<p class="legend" style="color:var(--ink)"><b>What was tried.</b> Two rule changes, so that the results above mean what they say. First, the shared-noise settings are frozen: until now a recalibration rode along with every change to the players, so a player experiment and a noise experiment moved at once and could not be told apart. Changing either setting is now its own declared experiment. Second, a change to the players is judged with the machinery switched off, on the variety measure against the human {fmt(F(cl_dec["human"]["var_cond_mean"]),1)}, alongside the usual checks &mdash; because while the persistence is supplying most of the group-spread row by compounding, that row cannot tell you whether a change to the players helped.</p>
</div>
<div>
<p class="legend" style="color:var(--ink)"><b>What happened.</b> The rule protecting the response row misfired on its first outing and had to be amended. It says a band's slope may not fall to half its previous value or less. It fired twice on reference stacks: once on a change of {ce_se[1]["change_in_se"]["15-19"]:.2f} standard errors on the thinnest band in the suite, and once on a slope going from {fmt(lead(ce_sl[2]["before"][3]),3,sign=True)} to {fmt(lead(ce_sl[2]["after"][3]),3,sign=True)}, toward the human {fmt(HUMAN_SLOPES[2],3,sign=True)} &mdash; an improvement read as an erosion, because a magnitude test on a signed quantity cannot tell the difference. Two qualifications were added: the rule does not fire when the new slope is closer to the human value than the old one, and it fires only when the change exceeds one pooled standard error. Neither changes a verdict already recorded, because both failing experiments failed their declared row independently.</p>
<p class="legend"><b>What it means.</b> A safety rule that fails experiments over differences too small to be real, and once over an improvement, is worse than no rule, because it teaches people to argue with it. It is now stated with the arithmetic that makes a firing readable, and every experiment touching that row reports its slopes with standard errors.</p>
</div>
</div>
'''

ledger = [
 ("PR #182", f"{PR}182", "open", "Bookkeeping for the main branch: ignore the local config file, cluster account lines, the evaluation metric notes."),
 ("PR #183", f"{PR}183", "open", "The held-out teacher-forced test of the players' reaction to punishment, built on top of the PR #181 branch."),
 ("PR #184", f"{PR}184", "open", "The re-baseline: manager retrained on the current contribution, RCE made a protected row, ledger reset."),
 ("rcb-alternative-response-slope", "", "merged into PR #184", "The RCE row, its tests, and the comparison report over 40 stacks."),
 ("auto/punisher-current-contribution-sims", "", "documentation", "The rerun configs, the runner script, and the 'before' column of the score tables."),
 ("auto/punisher-current-contribution-gmlp", "", "documentation", "The Gaussian-MLP code tree plus the three fix commits copied over; used for cases c and d."),
 ("PR #186", f"{PR}186", "open &middot; result", "The shared draw run three ways: on, off, and redrawn every round. Establishes that the players' per-round randomness is right and the variety of situations is two thirds of the human value. No model proposed."),
 ("PR #187", f"{PR}187", "open &middot; result", "What the shared mistake is made of, measured on the real games: observable group facts explain a seventh of it, and the rest is a one-round echo with no lasting part. No model trained, no simulation run."),
 ("PR #188", f"{PR}188", "open &middot; result", "Five copies of the players trained with different random seeds. Their disagreement is about a sixth of the shared draw's strength and decays in a round, which closes the Bayesian route."),
 ("PR #189", f"{PR}189", "open &middot; protocol", "The four-step plan, the freeze on the shared-noise settings, and the rule that a change to the players is judged with the machinery off. Later amended: the protected-row magnitude clause gained two qualifications after firing wrongly twice."),
 ("PR #190", f"{PR}190", "open &middot; fail", "Step 3. The other line's group-switching component moved across. Every pure switching measure improved and every response measure worsened; the protected row was violated. Lowest 22-row mean on record and still a failure."),
 ("PR #191", f"{PR}191", "open &middot; result", "Step 1. The emission-head comparison, which falsified its own hypothesis: the current design extrapolates best and holds the most variety. Also the sharpest description of the defect, as a failure of late divergence."),
 ("PR #192", f"{PR}192", "open &middot; fail", "Step 2. A &lsquo;gave the maximum&rsquo; flag for both managers. The mechanism at the ceiling is now essentially exact, the declared row moved more than it ever has and still did not cross a band, and the remainder decomposes onto the players."),
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
.twocol>*{min-width:0}
.tablewrap{overflow-x:auto;min-width:0;max-width:100%}
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
svg .bar.s1{fill:var(--c-blue)}svg .bar.s2{fill:var(--c-amber)}svg .bar.s3{fill:var(--c-red)}
.sw.s1{background:var(--c-blue)}.sw.s2{background:var(--c-amber)}.sw.s3{background:var(--c-red)}
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
<p class="sub">Real people played a public-goods game with a punishing manager; this project trains simulated players and a simulated manager to behave like them. This page reports two working days, 18 and 19 September 2026, on those simulations: the discovery that the simulated manager punished last round's contribution on every branch of the code and the stacks rerun under the fix; three experiments on why the simulated players' mistakes come in groups; and a four-step programme of which two steps failed the bar they had set themselves. Companion to the Autoresearch Atlas of experiments #146&ndash;#181.</p>
<nav>
  <button type="button" class="on" data-layer="overview">Overview</button>
  <button type="button" data-layer="scores">Before / after</button>
  <button type="button" data-layer="response">Response instrument</button>
  <button type="button" data-layer="mechanism">Punisher mechanism</button>
  <button type="button" data-layer="shared">Shared mistakes</button>
  <button type="button" data-layer="steps4">Four steps</button>
  <button type="button" data-layer="stories">Stories</button>
  <button type="button" data-layer="ledger">Ledger</button>
</nav>

<section class="layer on" id="overview">
{ABOUT}
<h2>The headline numbers</h2>
<p class="legend">Where things stand after the manager fix and the seven experiments that followed it. The first tile is the one defect still open; the next two are the piece of it that has been closed and the piece that has been isolated; the last is the best overall score on record and what it cost to get there.</p>
{overview_tiles()}
<h2>How the work unfolded</h2>
<ol class="steps">{steps_html()}</ol>
<h2>The one-paragraph reading</h2>
<p class="legend" style="max-width:76ch;font-size:14px;color:var(--ink)">The manager fix worked where it applies: the row scoring how the manager punishes and the reaction-to-punishment row improve by a full score band in every stack, and a later indicator for &ldquo;gave the maximum&rdquo; made the simulated manager's behaviour at the ceiling essentially exact. What is left is one defect and three smaller ones. The defect is that the simulated games do not become as varied as the real ones: real groups keep pulling apart through the last third of a game and the models flatten out, and the shared-noise machinery has been covering that up by supplying variety rather than the correlation it is named for. The three smaller ones are all now cleanly isolated with a baseline ready: the simulated players under-react to heavy punishment at the ceiling by about {abs(F(ce_rcc["human"][2]))/abs(F(ce_rcc["frontier after"][2])):.1f} times, the simulated manager's punishment falls with contribution at about half the human strength, and the borrowed group-switching component sends the wrong people away after being punished. Two of the four most recent steps failed their declared gate, and a third falsified the hypothesis its own author proposed &mdash; that a bell-curve output would resist the drift better than the design already in use. It is the design already in use that resists it best.</p>
</section>

<section class="layer" id="shared">
{shared_error_section()}
</section>

<section class="layer" id="steps4">
{four_steps_section()}
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
<p class="legend" style="color:var(--ink)"><b>Why it was added but does not decide alone.</b> Its noise ceiling is large. Two halves of the human data differ by 0.086 in slope, about three quarters of the human slopes themselves. So a simulation whose players ignore punishment entirely scores 1.42, a 'minor deviation', and one with half the human response scores 0.82, at the ceiling. RCE therefore sits beside RCB as a protected row: no experiment may worsen its score band, flip one of the four human signs, or halve a band's slope. The last of those three had to be qualified twice after it fired on two experiments it should not have &mdash; see the Four steps tab.</p>
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
<p class="legend">Before the fix every simulated manager had a near-zero weight on the current contribution and about &minus;0.08 on the previous one, the reverse of the human pattern, and punished full contributors five to nine times too often. After the fix the current-round weight is &minus;0.12 to &minus;0.17 against the human &minus;0.24, the timing check has the human ordering, and punishment again falls as contribution rises. What remains is the ceiling: full contributors are still punished three to four times too often, and when they are punished the amount is too small. That residual is why RCC, the reaction at the ceiling, did not move. It was closed later, by giving both managers a flag for &ldquo;gave the maximum&rdquo;: the punish rate at the ceiling goes to {fmt(F(ce_mech["frontier after"]["P(p>0|c_t=20)"]),3)} against the real {fmt(F(ce_mech["human"]["P(p>0|c_t=20)"]),3)} and the severity to {fmt(F(ce_mech["frontier after"]["E[p|p>0] 20"]),2)} against {fmt(F(ce_mech["human"]["E[p|p>0] 20"]),2)}. RCC still did not cross a band, for a reason that turned out to be about the players rather than the manager &mdash; the Four steps tab has it.</p>
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
<tr><td>Chance a full contributor is punished, in the simulation</td><td class="num">{fmt(F(ce_mech["frontier before"]["P(p>0|c_t=20)"]),3)}</td><td class="num">{fmt(F(ce_mech["frontier after"]["P(p>0|c_t=20)"]),3)}</td><td>real managers {fmt(F(ce_mech["human"]["P(p>0|c_t=20)"]),3)}; PR #192</td></tr>
<tr><td>How hard, when they are punished at the ceiling</td><td class="num">{fmt(F(ce_mech["frontier before"]["E[p|p>0] 20"]),2)}</td><td class="num">{fmt(F(ce_mech["frontier after"]["E[p|p>0] 20"]),2)}</td><td>real managers {fmt(F(ce_mech["human"]["E[p|p>0] 20"]),2)}; now a slight overshoot</td></tr>
<tr><td>RCC, the reaction at the ceiling, on the frontier stack</td><td class="num">{fmt(F(ce_ba["RCC"]["frontier_before"]),4)}</td><td class="num">{fmt(F(ce_ba["RCC"]["frontier_after"]),4)}</td><td>largest move that row has had; still band 1&ndash;2, so the experiment failed</td></tr>
<tr><td>Next-round drop of a punished full contributor</td><td class="num">{fmt(F(ce_rcc["frontier before"][2]))}</td><td class="num">{fmt(F(ce_rcc["frontier after"][2]))}</td><td>real people {fmt(F(ce_rcc["human"][2]))}; the players' defect, not the manager's</td></tr>
<tr><td>Linear manager model, cross-validated log loss, with the ceiling flag</td><td class="num">{F(ce_cv0["log_loss"]):.4f}</td><td class="num">{F(ce_cv["log_loss"]):.4f}</td><td>same split and seed; 31 classes, so the margin is small by construction</td></tr>
<tr><td>Variety of situations reached, machinery off</td><td class="num">{fmt(F(cl_dec["A"]["var_cond_mean"]),1)} with it on</td><td class="num">{fmt(F(cl_dec["B"]["var_cond_mean"]),1)}</td><td>real games {fmt(F(cl_dec["human"]["var_cond_mean"]),1)}; the open defect</td></tr>
<tr><td>Within-group co-movement left after the model's own expectation</td><td class="num">{fmt(F(ms_base["raw contribution, plain Pearson"]["value"]),3)} raw</td><td class="num">{fmt(F(ms_base["level residual, plain Pearson"]["value"]),3)}</td><td>the network already explains {(1-F(ms_base["level residual, plain Pearson"]["value"])/F(ms_base["raw contribution, plain Pearson"]["value"]))*100:.0f} per cent of it</td></tr>
<tr><td>Shared-draw strength: what five trained copies imply, against what is fitted</td><td class="num">{se_sum["implied_rho"]:.4f}</td><td class="num">{se_sum["rho_copula_json"]:.4f}</td><td>fitted interval {se_sum["rho_copula_json_ci"][0]:.3f} to {se_sum["rho_copula_json_ci"][1]:.3f}; PR #188</td></tr>
<tr><td>Lowest 22-row mean on record</td><td class="num">{means["b_skip_after"]:.4f}</td><td class="num">{F(sk_cmp["mean"]["after"]):.4f}</td><td>PR #190, which failed its declared row and the protected one</td></tr>
</tbody></table></div>
<h2>Clean-up as the pull requests close</h2>
<ul class="code">
<li>On the Raven cluster, one isolated folder per experiment: <code>~/repros/ai-runs/punisher-current-contr</code>, <code>…-gmlp</code>, <code>…-train</code>, <code>…-tests</code>, <code>copula-cl-variance</code>, <code>copula-missing-state</code>, <code>copula-ensemble</code>, <code>head-diagnostic</code>, <code>head-diag-cat</code>, <code>punisher-ceiling</code>, <code>switch-kexo-port</code></li>
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
