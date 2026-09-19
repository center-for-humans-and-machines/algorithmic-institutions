"""Rebaseline Atlas: sibling of the Autoresearch Atlas (report_bundle.html), new data.

CSS and JS are copied verbatim from the template; the sections are re-rendered
from the CSVs of the punisher re-baseline (worktree D), the RCE comparison
(worktree R) and the held-out teacher-forced test (worktree H), plus the
committed analysis files of the seven PRs that landed after the re-baseline,
read straight off their branches with gitfile() and cached beside this script.
"""
import ast, base64, csv, html, io, json, math, re, statistics, subprocess
from pathlib import Path

import pandas as pd

D = Path("/Users/brinkmann/repros/algorithmic-institutions/.claude/worktrees/agent-a80d622b939db4c1c")
R = Path("/Users/brinkmann/repros/algorithmic-institutions/.claude/worktrees/agent-a91a63a720dcceb19")
H = Path("/Users/brinkmann/repros/algorithmic-institutions/.claude/worktrees/agent-a73127aa5d3033a41")
REPO = Path("/Users/brinkmann/repros/algorithmic-institutions")
TPL = Path("/Users/brinkmann/Downloads/report_bundle.html")
HERE = Path(__file__).resolve().parent
CACHE = HERE / "data"
OUT = HERE / "rebaseline_atlas.html"
PC = D / "plots/data_analysis/evaluation/punisher_current_contr"
PRURL = "https://github.com/center-for-humans-and-machines/algorithmic-institutions/pull/"
SIZE_LIMIT = 14 * 1024 * 1024

esc = lambda s: html.escape(str(s), quote=True)


# ------------------------------------------- committed files on other branches
# same machinery as ../atlas/build_atlas.py, but the cache keeps the full path
# (two branches ship visuals/CA_hist.jpg under different sim dirs).
def gitfile(branch, path):
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


def _md_rows(lines, i):
    out = {}
    for l in lines[i + 2:]:
        if not l.startswith("|"):
            break
        cells = [c.strip().strip("*").strip() for c in l.strip().strip("|").split("|")]
        out[cells[0]] = cells
    return out


def md_table(branch, path, marker):
    """Rows of the first markdown table in `path` whose header contains `marker`."""
    lines = gitfile(branch, path).read_text().splitlines()
    return _md_rows(lines, next(k for k, l in enumerate(lines) if marker in l and l.startswith("|")))


def md_tables(branch, path, marker):
    """Every markdown table whose header contains `marker`, in file order."""
    lines = gitfile(branch, path).read_text().splitlines()
    return [_md_rows(lines, i) for i, l in enumerate(lines) if l.startswith("|") and marker in l]


def lead(cell):
    """The leading signed number of a cell like '+0.013 +- 0.019 (n 1375)'."""
    return float(re.match(r"\s*([+-]?[0-9.]+)", cell).group(1))


def _kv(body):
    """'a=1, b=[x], c={...}' -> dict, values literal-eval'd where they parse."""
    out = {}
    for k, v in re.findall(r"(\w+)=(\{[^}]*\}|\[[^\]]*\]|[^,]+)", body):
        try:
            out[k] = ast.literal_eval(v.strip())
        except (ValueError, SyntaxError):
            out[k] = v.strip()
    return out


def pchecks(branch, path):
    """Every 'protected-row checks[ (amended rule)]:' record in a before_after.md."""
    out = []
    for line in gitfile(branch, path).read_text().splitlines():
        if line.startswith("protected-row checks"):
            body = line.split(":", 1)[1].strip()
            out.append(ast.literal_eval(body) if body.startswith("{") else _kv(body))
    return out


def verdicts(branch, path):
    """Every 'verdict: ...' line of a before_after.md, as dicts."""
    return [_kv(m) for m in
            re.findall(r"^verdict: (.*)$", gitfile(branch, path).read_text(), re.M)]


F = float

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
NOISE = "#8a6d1e"   # one new accent: measurement noise (PR #195's seed floor)
ALERT = "#d03b3b"   # already the failed-gate ring; also marks the open escalation

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

# ------------------------------------- the seven PRs that landed after the re-baseline
B_CL, B_MS, B_SE = "auto/copula-closed-loop-variance", "auto/copula-missing-state", "auto/copula-seed-ensemble"
B_HS, B_CE, B_SK = "auto/head-state-spread-diagnostic", "auto/punisher-ceiling-fix", "auto/switch-kexo-port"
B_PG = "docs/post-rebaseline-program"
P_CL = "plots/data_analysis/evaluation/copula_closed_loop/"
P_MS = "plots/data_analysis/evaluation/copula_missing_state/"
P_SE = "plots/data_analysis/copula_seed_ensemble/"
P_HS = "plots/data_analysis/evaluation/head_state_spread/"
P_CE = "plots/data_analysis/evaluation/punisher_ceiling_fix/"
P_SK = "plots/data_analysis/evaluation/switch_kexo_port/"

# PR #186: the shared draw on, off and with its persistence removed
cl_dec = dkey(B_CL, P_CL + "cg_decomposition.csv", "arm")
cl_blk = {(r["arm"], r["rounds"]): r for r in drows(B_CL, P_CL + "round_blocks.csv")}
cl_sc = dkey(B_CL, P_CL + "scores_22.csv", "row")
cl_rce = dkey(B_CL, P_CL + "rce_bands.csv", "arm")
with open(gitfile(B_CL, P_CL + "latent_regression.csv")) as f:
    cl_lat = dict(csv.reader(f))

# PR #187: what the shared error is made of, on the human games
ms_base = dkey(B_MS, P_MS + "baseline.csv", "quantity")
ms_joint = dkey(B_MS, P_MS + "joint_model.csv", "scale")
ms_cand = dkey(B_MS, P_MS + "candidates.csv", "candidate")
ms_ref = dkey(B_MS, P_MS + "reference_sets.csv", "set")
ms_pers = dkey(B_MS, P_MS + "persistence_boot.csv", "stage")
ms_fwd = drows(B_MS, P_MS + "forward_selection.csv")

# PR #188: five copies of the contributor, trained with different seeds
se_sum = djson(B_SE, P_SE + "train40_summary.json")
se_dis = dkey(B_SE, P_SE + "train40_disagreement.csv", "quantity")
SE_SIM = "23_2g8a_contr_stimulus_skip_seed_ensemble_self_gnncopar1_contr_gnn_switch_curpun"
se_sim = {r["metric"]: F(r["score"]) for r in drows(B_SE, f"plots/simulation/{SE_SIM}/evaluation/scores.csv")}

# PR #191: emission head and the spread of the states the players reach
hs_head = dkey(B_HS, P_HS + "headline.csv", "arm")
hs_boot = dkey(B_HS, P_HS + "retention_bootstrap.csv", "arm")
hs_blk = {(r["arm"], r["rounds"]): r for r in drows(B_HS, P_HS + "round_blocks.csv")}
hs_gain = {(r["model"], r["delta"]): r for r in drows(B_HS, P_HS + "gain_curves.csv") if r["set"] == "common_6_14"}
hs_recon = md_tables(B_HS, "notes/autoresearch_log/head-state-spread-diagnostic.md", "mean predictive variance")[0]
GAIN_D = ["-6", "-4", "-2", "2", "4", "6"]

# PR #192: the manager at the contribution ceiling
ce_ba = dkey(B_CE, P_CE + "before_after.csv", "metric")
ce_mech = dkey(B_CE, P_CE + "mechanism_selfplay.csv", "")
ce_tf = dkey(B_CE, P_CE + "mechanism_teacher_forced.csv", "")
ce_logit = dkey(B_CE, P_CE + "human_ceiling_logit.csv", "")
ce_cv = drows(B_CE, "data/baselines/punishment_cv_multinomial_ceiling.csv")[0]
ce_cv0 = drows(B_CE, "data/baselines/punishment_cv_multinomial_current_contr.csv")[0]
ce_rcc = md_table(B_CE, "notes/autoresearch_log/punisher-ceiling-fix.md", "dc, punished")
ce_sl = md_tables(B_CE, P_CE + "before_after.md", "| RCE slopes |")          # frontier, ref_lin, ref_gnn
ce_se = [ast.literal_eval(m) for m in
         re.findall(r"protected-row checks: (\{.*\})", gitfile(B_CE, P_CE + "before_after.md").read_text())]
ce_verdict = dict(re.findall(r"(\w+)=([^,\n]+)", re.search(
    r"verdict: (.*)", gitfile(B_CE, P_CE + "before_after.md").read_text()).group(1)))
CE_SIM = {"b_skip": "23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_ceiling",
          "e_gnn": "23_2g8a_self_gnn_contr_gnn_switch_ceiling"}
# the largest move the ceiling flag makes to the punisher's slope on c_t, in any condition
CE_SLOPE_MAX = max(abs(F(t[a]["OLS c_t"]) - F(t[b]["OLS c_t"])) for t, a, b in (
    (ce_tf, "lin new (c_t)", "lin ceiling"), (ce_tf, "gnn new (c_t)", "gnn ceiling"),
    (ce_mech, "frontier before", "frontier after"), (ce_mech, "ref_lin before", "ref_lin after"),
    (ce_mech, "ref_gnn before", "ref_gnn after")))

# PR #190: the other lineage's group-switching component on the frontier trunk
sk_cmp = dkey(B_SK, P_SK + "compare.csv", "metric")
sk_rce = dkey(B_SK, P_SK + "rce_bands.csv", "stage")

# ------------------------------- the four PRs that followed the four-step programme
B_SS, B_TO = "auto/seed-spread-noise-floor", "auto/punisher-timeout-feature"
B_ST, B_CR = "auto/sim-timeout-imputation", "auto/contribution-copula-recalibrated"
P_SS = "plots/data_analysis/evaluation/seed_spread_noise_floor/"
P_TO = "plots/data_analysis/evaluation/punisher_timeout_feature/"
P_ST = "plots/data_analysis/evaluation/sim_timeout_imputation/"
P_CR = "plots/data_analysis/evaluation/contribution_copula_recalibrated/"
L_SS = "notes/autoresearch_log/seed-spread-noise-floor.md"
L_TO = "notes/autoresearch_log/punisher-timeout-feature.md"
L_ST = "notes/autoresearch_log/sim-timeout-imputation.md"
L_CR = "notes/autoresearch_log/contribution-copula-recalibrated.md"

# PR #195: the same contributor architecture trained six ways, one stack, one draw
ARMS = ["seed_1", "seed_2", "seed_3", "seed_4", "seed_5", "shipped"]
ss_row = dkey(B_SS, P_SS + "per_row.csv", "row")
ss_agg = dkey(B_SS, P_SS + "aggregates.csv", "quantity")
ss_rank = dkey(B_SS, P_SS + "arm_ranks.csv", "arm")
ss_stab = dkey(B_SS, P_SS + "ceiling_stability.csv", "row")
ss_lev = dkey(B_SS, P_SS + "levels.csv", "arm")
ss_levsd = dkey(B_SS, P_SS + "levels_spread.csv", "")
ss_verd = drows(B_SS, P_SS + "verdicts.csv")
ss_se = dkey(B_SS, P_SS + "rce_band_se.csv", "")
# the six members' training fit, recorded in the log (PR #188's cross-validated log losses)
_cv = re.search(r"cross-validated log losses as ([0-9.–/ ]+) against the shipped artifact's ([0-9.]+)",
                gitfile(B_SS, L_SS).read_text())
SS_CV = ([F(v) for v in _cv.group(1).replace("/", " ").split()], F(_cv.group(2)))
# how far apart the six arms' draws actually are, pairwise
SS_COINCIDE = [F(v) for v in re.search(
    r"contribution entries, ([\d.]+)% to ([\d.]+)% coincide", gitfile(B_SS, L_SS).read_text()).groups()]

# PR #196: can the manager tell a timeout from a chosen zero?
to_ba = dkey(B_TO, P_TO + "before_after.csv", "metric")
to_tf = dkey(B_TO, P_TO + "mechanism_teacher_forced.csv", "")
to_sl = md_tables(B_TO, P_TO + "before_after.md", "| RCE slopes |")     # frontier, ref_lin, ref_gnn
to_pc = pchecks(B_TO, P_TO + "before_after.md")
to_vd = verdicts(B_TO, P_TO + "before_after.md")[0]
_to_log = gitfile(B_TO, L_TO).read_text()
# the human manager at a timeout and at a chosen zero, and the accounting identity
TO_HUM = dict(zip(("timeout", "zero_all", "zero_mask"),
                  [F(v) for v in re.search(
                      r"punished \*\*([\d.]+)%\*\* of them; players who chose to contribute 0 were punished "
                      r"\*\*([\d.]+)%\*\* of the time over all rows and \*\*([\d.]+)%\*\*", _to_log).groups()]))
TO_ID = md_table(B_TO, L_TO, "group-rounds the identity holds on")
TO_N, TO_AR, TO_GR = (int(re.search(r"all (\d+) timed-out agent-rounds record", _to_log).group(1)),
                      *[int(v.replace(",", "")) for v in re.search(
                          r"\(([\d,]+) agent-rounds, ([\d,]+) group-rounds\)", _to_log).groups()])
TO_T = F(re.search(r"a paired \*\*t = ([-\d.]+)\*\* over five folds", _to_log).group(1))
TO_PRED = F(re.search(r"confirmed to within ([\d.]+) on a number nobody tuned", _to_log).group(1))
TO_CV = {k: [F(x) for x in re.search(p, _to_log).groups()] for k, p in (
    ("lin", r"cross-validated log loss \*\*([\d.]+) -> ([\d.]+)\*\*"),
    ("lin_test", r"locked test fold, untouched during selection: \*\*([\d.]+) -> ([\d.]+)\*\*"),
    ("gnn", r"The graph family: CV log loss \*\*([\d.]+) -> ([\d.]+)\*\*"))}

# PR #197: what a timed-out player's contribution looks like to each model
st_ba = dkey(B_ST, P_ST + "before_after.csv", "metric")
st_dec = dkey(B_ST, P_ST + "copula_closed_loop/cg_decomposition.csv", "arm")
st_blk = {(r["arm"], r["rounds"]): r for r in drows(B_ST, P_ST + "copula_closed_loop/round_blocks.csv")}
st_sl = md_tables(B_ST, P_ST + "before_after.md", "| RCE slopes |")
st_pc = pchecks(B_ST, P_ST + "before_after.md")
st_vd = verdicts(B_ST, P_ST + "before_after.md")[0]
st_pb, st_pa = djson(B_ST, P_ST + "probe_before.json"), djson(B_ST, P_ST + "probe_after.json")
ST_MODEL = {m["label"]: m for m in st_pb["models"]}
ST_MODEL_A = {m["label"]: m for m in st_pa["models"]}
PREV_B = ST_MODEL["contribution"]["served_at_timeout_cells"]["prev_contribution"]
PREV_A = ST_MODEL_A["contribution"]["served_at_timeout_cells"]["prev_contribution"]
# arms of the state-spread diagnostic: A parent+copula, B candidate+copula, C candidate off, D parent off
ST_ARM = {"parent_on": "A", "cand_on": "B", "cand_off": "C", "parent_off": "D"}

# PR #198: was the shared-noise strength calibrated against the defect?
cr_ba = dkey(B_CR, P_CR + "before_after.csv", "metric")
cr_dec = dkey(B_CR, P_CR + "copula_closed_loop/cg_decomposition.csv", "arm")
cr_par = djson(B_CR, "artifacts/artificial_humans/group_switching_contribution_50ep_vnode_stimulus_skip"
                     "_herding_copula_recal/calibration/copula_params.json")
cr_pc = pchecks(B_CR, P_CR + "before_after.md")
cr_vd = verdicts(B_CR, P_CR + "before_after.md")[0]
_cr_log = gitfile(B_CR, L_CR).read_text()
CR_SHA = re.search(r"output sha256 \*\*`([0-9a-f]{64})`\*\*", _cr_log).group(1)
CR_FIELDS = int(re.search(r"finds \*\*all (\d+) estimate and provenance fields identical\*\*", _cr_log).group(1))
CR_MIN = int(re.search(r"the refit \(measured -- Raven job \d+, (\d+) min", _cr_log).group(1))
CR_FIGS = int(re.search(r"`metrics.csv` and all (\d+) figures", _cr_log).group(1))
# the open escalation, quoted from the two logs that raise it
ESCALATION = re.search(r"(`convert.load_human` NaNs the human's timed-out contributions[^*]*?frozen surface \(§8\))",
                       gitfile(B_ST, L_ST).read_text()).group(1)
ESC_HTML = re.sub(r"`([^`]+)`", r"<code>\1</code>", ESCALATION)
ESC_PLAIN = ESCALATION.replace("`", "")

# --------------------------------------------------- the runs that followed the re-baseline
# key, parent, spine, PR, kind, short label, long label, verdict, one-line note
FOLLOW = [
    ("f_ceil", "b_skip", "gnn", 192, "step", "ceil", "PR #192 ceiling flag on the frontier stack (#181 skip)",
     "FAIL -- gate 1", "declared row RCC moved further than it ever has and stayed in its band"),
    ("i_kexo", "b_skip", "gnn", 190, "step", "kexo", "PR #190 k-one-hot switch head ported onto the frontier trunk",
     "FAIL -- gate 1 and the protected row", "every pure switching row improved, every response row got worse"),
    ("g_clin", "e_lin", "gnn", 192, "step", "c&middot;lin", "PR #192 ceiling flag, main-sweep linear punisher",
     "reference rerun, reported not gated", "the protected-row clause misfired here on the thinnest band"),
    ("h_cgnn", "e_gnn", "gnn", 192, "step", "c&middot;gnn", "PR #192 ceiling flag, main-sweep GNN punisher",
     "reference rerun, reported not gated", "the broadest improvement of the four runs; 8 -> 11 rows at or under the ceiling"),
    ("m_tout", "f_ceil", "gnn", 196, "step", "timeout",
     "PR #196 the punisher is told whether the player gave any input at all",
     "FAIL -- gate 1", "the declared row cleared the noise floor in the right direction and still held its band"),
    ("n_simto", "m_tout", "gnn", 197, "step", "sim 0",
     "PR #197 the simulation serves the recorded 0, not the imputed 9, to the contribution model",
     "FAIL -- gate 1 and the protected row", "a confirmed serving defect, fixed; its whole cost falls on the group-spread rows"),
    ("o_recal", "n_simto", "gnn", 198, "step", "rho",
     "PR #198 the contribution copula refitted against the corrected serving path",
     "FAIL -- the hypothesis is refuted", "the refit returns the shipped strength to the last digit; every row moves by exactly zero"),
    ("j_off", "b_skip", "gnn", 186, "abl", "no copula", "PR #186 arm B: the shared-noise machinery switched off",
     "ablation, not an experiment", "what the frontier scores with nothing coupling the group's draws"),
    ("k_redraw", "b_skip", "gnn", 186, "abl", "redrawn", "PR #186 arm C: the shared draw redrawn every round",
     "ablation, not an experiment", "the same strength with the episode-long persistence removed"),
    ("l_seeds", "b_skip", "gnn", 188, "abl", "5 seeds", "PR #188: five trained copies in place of the shared draw",
     "ablation, not an experiment", "one copy drawn per game, the Bayesian alternative made concrete"),
]
FKEYS = [f[0] for f in FOLLOW]
FOL = {f[0]: f for f in FOLLOW}
STEPKEYS = [f[0] for f in FOLLOW if f[4] == "step"]
ABLKEYS = [f[0] for f in FOLLOW if f[4] == "abl"]
FCOL = {"step": "#b4462c", "abl": "#7a7a95"}   # the two new edge colours: follow-up run / ablation

# every vector below is 22 rows read off a committed file; the parents' "after"
# columns are the baselines those files were written against (asserted).
VEC = {c: {st: {r: float(S[c][st][r]) for r in ROWS} for st in ("before", "after")} for c in ORDER}
_after = lambda k: VEC[k]["after"]
VEC["f_ceil"] = {"before": {r: F(ce_ba[r]["frontier_before"]) for r in ROWS},
                 "after": {r: F(ce_ba[r]["frontier_after"]) for r in ROWS}}
VEC["g_clin"] = {"before": {r: F(ce_ba[r]["ref_lin_before"]) for r in ROWS},
                 "after": {r: F(ce_ba[r]["ref_lin_after"]) for r in ROWS}}
VEC["h_cgnn"] = {"before": {r: F(ce_ba[r]["ref_gnn_before"]) for r in ROWS},
                 "after": {r: F(ce_ba[r]["ref_gnn_after"]) for r in ROWS}}
VEC["i_kexo"] = {"before": {r: F(sk_cmp[r]["before"]) for r in ROWS},
                 "after": {r: F(sk_cmp[r]["after"]) for r in ROWS}}
VEC["j_off"] = {"before": {r: F(cl_sc[r]["A"]) for r in ROWS}, "after": {r: F(cl_sc[r]["B"]) for r in ROWS}}
VEC["k_redraw"] = {"before": {r: F(cl_sc[r]["A"]) for r in ROWS}, "after": {r: F(cl_sc[r]["C"]) for r in ROWS}}
VEC["l_seeds"] = {"before": {r: F(cl_sc[r]["A"]) for r in ROWS}, "after": {r: se_sim[r] for r in ROWS}}
VEC["m_tout"] = {"before": {r: F(to_ba[r]["frontier_before"]) for r in ROWS},
                 "after": {r: F(to_ba[r]["frontier_after"]) for r in ROWS}}
VEC["n_simto"] = {"before": {r: F(st_ba[r]["frontier_before"]) for r in ROWS},
                  "after": {r: F(st_ba[r]["frontier_after"]) for r in ROWS}}
VEC["o_recal"] = {"before": {r: F(cr_ba[r]["before"]) for r in ROWS},
                  "after": {r: F(cr_ba[r]["after"]) for r in ROWS}}
for k in FKEYS:  # each follow-up's own baseline must be its parent's post-fix state
    p = FOL[k][1]
    assert all(abs(VEC[k]["before"][r] - _after(p)[r]) < 1e-9 for r in ROWS), k

# PR #195's six arms, as profiles against their own six-arm mean (no before/after)
SEED_SD = {r: F(ss_row[r]["sd"]) for r in ROWS}
SSMEAN = {r: F(ss_row[r]["mean"]) for r in ROWS}
SD_MEAN, SD_LE1 = F(ss_agg["mean_22"]["sd"]), F(ss_agg["rows_le1"]["sd"])
VMEAN_SS = F(ss_agg["mean_22"]["mean"])
SD_ROW = statistics.median(SEED_SD.values())            # the typical row's seed sd
SPAN_ROW = statistics.median(F(ss_row[r]["range"]) for r in ROWS)
UNGATE = [r for r in ROWS if ss_row[r]["gateable_on_one_run"] == "False"]
AKEYS = ["arm_" + a for a in ARMS]
for a in ARMS:
    VEC["arm_" + a] = {"before": dict(SSMEAN), "after": {r: F(ss_row[r][a]) for r in ROWS}}
# the control PR #195 claims: its shipped arm reproduces PR #192's after column digit for digit
assert all(abs(VEC["arm_shipped"]["after"][r] - VEC["f_ceil"]["after"][r]) < 1e-12 for r in ROWS)

VMEAN = {k: {st: sum(VEC[k][st][r] for r in ROWS) / len(ROWS) for st in ("before", "after")} for k in VEC}
VLE1 = {k: {st: sum(1 for r in ROWS if VEC[k][st][r] <= 1) for st in ("before", "after")} for k in VEC}
VGT2 = {k: {st: sum(1 for r in ROWS if VEC[k][st][r] > 2) for st in ("before", "after")} for k in VEC}
for c in ORDER:  # the derived means must still match the re-baseline CSV's own summary lines
    for st in ("before", "after"):
        assert abs(VMEAN[c][st] - MEAN[c][st]) < 1e-9 and VLE1[c][st] == LE1[c][st]


def band(v):
    return 0 if v <= 1 else 1 if v <= 2 else 2 if v <= 5 else 3


BANDLBL = ["&lt;= 1", "1-2", "2-5", "&gt; 5"]
BANDTXT = ["<= 1", "1-2", "2-5", "> 5"]
f3 = lambda v: f"{v:.3f}"
f2 = lambda v: f"{v:.2f}"
sgn = lambda v, d=3: f"{v:+.{d}f}"
arrow = lambda c, r: f"{f3(VEC[c]['before'][r])} -> {f3(VEC[c]['after'][r])}"
band_arrow = lambda c, r: f"{BANDTXT[band(VEC[c]['before'][r])]} -> {BANDTXT[band(VEC[c]['after'][r])]}"
pr_link = lambda n: f'<a href="{PRURL}{n}" target="_blank">#{n}</a>'


def stats(c):
    b, a = VEC[c]["before"], VEC[c]["after"]
    up = [r for r in ROWS if band(a[r]) < band(b[r])]
    down = [r for r in ROWS if band(a[r]) > band(b[r])]
    return dict(base_mean=round(VMEAN[c]["before"], 4), cand_mean=round(VMEAN[c]["after"], 4),
                d_mean=round(VMEAN[c]["after"] - VMEAN[c]["before"], 4),
                d_le1=VLE1[c]["after"] - VLE1[c]["before"],
                d_gt2=VGT2[c]["after"] - VGT2[c]["before"],
                upgrades=len(up), up_rows=", ".join(up), down_rows=", ".join(down))


ST = {k: stats(k) for k in list(ORDER) + FKEYS}
# labels: the six reruns keep their own, the follow-ups take theirs from FOLLOW
LABEL = {c: CASE[c][3] for c in ORDER}
SHORT = {c: CASE[c][5] for c in ORDER}
SPINEOF = {c: CASE[c][1] for c in ORDER}
for k, par, sp, pr, kind, short, label, verdict, note in FOLLOW:
    LABEL[k], SHORT[k], SPINEOF[k] = label, short, sp
for a in ARMS:
    SHORT["arm_" + a] = a.replace("seed_", "seed ")
TIPNAME = {c: LABEL[c] for c in list(ORDER) + FKEYS}

# ---------------------------------------------------- the measurement floor (PR #195)
# every movement below is quoted in units of its own row's seed standard deviation:
# how far that row travels when nothing changes but the contributor's training seed.
row_sd = lambda k, r: abs(VEC[k]["after"][r] - VEC[k]["before"][r]) / SEED_SD[r]
mean_sd = lambda k: abs(VMEAN[k]["after"] - VMEAN[k]["before"]) / SD_MEAN
le1_sd = lambda k: abs(VLE1[k]["after"] - VLE1[k]["before"]) / SD_LE1
n_legible = lambda k: sum(1 for r in ROWS if row_sd(k, r) >= 1)
# a node is "inside the floor" when its whole-stack movement is smaller than the floor
INSIDE = {k for k in list(ORDER) + FKEYS if mean_sd(k) < 1}
FLOOR_NOTE = ("the floor is the contributor-retrain spread measured on the frontier stack "
              "(PR #195); on the other stacks it is a lower bound, not a full error bar")
# PR #195's own re-reading of the verdicts recorded before it, by PR and movement
sv = lambda pr, key: next(v for v in ss_verd if v["pr"] == pr and v["movement"].startswith(key))
sv_sd = lambda pr, key: F(sv(pr, key)["in_seed_sd"])
sv_in = lambda pr, key: sv(pr, key)["inside_floor"] == "True"
# summaries of a before/after table that carries one column per arm
colmean = lambda tbl, col: sum(F(tbl[r][col]) for r in ROWS) / len(ROWS)
colle1 = lambda tbl, col: sum(1 for r in ROWS if F(tbl[r][col]) <= 1)
colsd = lambda tbl, a, b: abs(colmean(tbl, b) - colmean(tbl, a)) / SD_MEAN


def floor_tip(k):
    m = (f" | mean move {mean_sd(k):.2f} seed sd"
         f"{' -- not distinguishable from a retrain' if k in INSIDE else ''}"
         f" | rows moving more than their own seed sd: {n_legible(k)}/22")
    return m


def node_tip(k, stage_word=None):
    t = (f"{TIPNAME[k]} ({SPINEOF[k]} spine) | mean {f3(VMEAN[k]['before'])} -> {f3(VMEAN[k]['after'])}"
         f" | rows <= 1: {VLE1[k]['before']} -> {VLE1[k]['after']}"
         f" | RCE {arrow(k, 'RCE')} ({band_arrow(k, 'RCE')})" + floor_tip(k))
    return t if stage_word is None else f"{stage_word}: {t}"


NODE_TIP = {c: node_tip(c) for c in list(ORDER) + FKEYS}

# ---------------------------------------------------------------- 1. stack cards
def stack_cards():
    reruns = lambda sp: ", ".join(CASE[c][5] for c in ORDER if CASE[c][1] == sp)
    def card(sp, name, contr, switch, pun):
        return (f'<div class="scard">\n<h3 style="color:{SPINE[sp]}">{name} <small>(reruns {esc(reruns(sp))})</small></h3>\n'
                f'<div class="srow"><span class="schip" style="background:{SLOT["contribution"]}">contribution</span><span>{contr}</span></div>\n'
                f'<div class="srow"><span class="schip" style="background:{SLOT["switch"]}">switch</span><span>{switch}</span></div>\n'
                f'<div class="srow"><span class="schip" style="background:{SLOT["punisher"]}">punisher</span><span>{pun}</span></div>\n</div>')
    pun = (f"a multinomial logistic regression over 31 punishment levels, now retrained on the current round's contribution "
           f"(it used to read last round's), its group draws coupled by the severity copula, recalibrated from rho {RHO['before']:.3f} to {RHO['after']:.3f}; "
           f"PR #192 added a gave-the-maximum flag on top, which made the behaviour at the ceiling essentially exact and still missed its band, "
           f"and PR #196 a gave-no-input flag, which earns its place in this linear family "
           f"(cross-validated log loss {TO_CV['lin'][0]:.4f} &rarr; {TO_CV['lin'][1]:.4f}) and nothing in the graph one &mdash; "
           f"neither flag is merged")
    return ('<div class="stacks">\n' +
            card("gnn", "the gnn stack",
                 "a graph neural network: members exchange messages each round, each keeps a small recurrent memory; "
                 "the #179 stack adds a per-group virtual node, the #181 stack a direct punishment-to-output skip; draws coupled by a herding copula, "
                 "whose strength and persistence PR #189 froze after PRs #186 and #188 measured what each of them does; "
                 "PR #191 tested the gaussian-MLP's centre-and-width output against these 21 free scores and the 21 scores won",
                 "a graph-network switch predictor; on decision rounds a joint head draws how many leave each group, then a conditional-Bernoulli step picks who; "
                 "PR #190 swapped in the gaussian-MLP line's k-one-hot version of that head, which improved every pure switching row and damaged every response row",
                 pun + " -- the main-sweep reference stack runs the plain multinomial (no copula) and, as a second run, the GNN punisher, both retrained the same way") +
            card("gmlp", "the gaussian-MLP stack",
                 "a tiny 2-layer neural network predicting a bell curve (centre and spread) per member with a group copula on the draws; "
                 "the #177 stack adds probability spikes at repeat / 0 / 20",
                 "the same graph-network switch predictor and joint exodus head, with the group sizes one-hot encoded (#174)",
                 pun + " -- unchanged between the two stacks, as before") +
            '</div>')


# ---------------------------------------------------------------- 2. progress tree
FAILED = {"f_ceil", "i_kexo", "m_tout", "n_simto", "o_recal"}   # missed their declared gate
# x of each rerun's "before" node: the frontier's gap is wide because five runs hang in it
XB = {"e_lin": 110, "e_gnn": 310, "a_vnode": 510, "b_skip": 710, "d_kexo": 1140, "c_infl": 1340}
STEP_OFF = {"f_ceil": 92, "i_kexo": 140, "g_clin": 50, "h_cgnn": 50,
            "m_tout": 104, "n_simto": 54, "o_recal": 54}
SEED_RAIL = 56   # offset of PR #195's six-arm rail from the frontier's post-fix node
DIAGS = [  # diagnostics with no 22-row score: parked on a strip under the runs they inform
    ("b_skip", "#187", "sharederr",
     "PR #187, diagnostic: what the shared error is made of, measured on the 50 human games. No simulation "
     "and so no 22-row score. Observable group state explains about a seventh of it; the lasting part is zero."),
    ("d_kexo", "#191", "head",
     "PR #191, diagnostic: the emission-head and state-spread probe. Three copula-off simulations (inflated, v2 "
     "and a matched skip+k-one-hot arm) measured for the variety of states they reach and for off-manifold gain; "
     "no 22-row scores were produced, so there is no height to plot."),
    ("c_infl", "#191", "head",
     "PR #191, diagnostic: the inflated Gaussian-MLP arm of the emission-head probe, run with the shared-noise "
     "machinery off (variety 16.51 against the categorical 18.88 and the human 27.93). No 22-row score."),
]


def tree_svg():
    W, Hh = 1540, 610
    x0, x1, ytop, ybot, ydiag = 56, 1516, 60, 500, 540
    m_lo, m_hi = 0.95, 1.95
    y = lambda m: ybot - (m - m_lo) / (m_hi - m_lo) * (ybot - ytop)
    o = [f'<svg viewBox="0 0 {W} {Hh}" font-family="system-ui, sans-serif">']
    for g in (1.0, 1.2, 1.4, 1.6, 1.8):
        o.append(f'<line x1="{x0}" y1="{y(g):.1f}" x2="{x1}" y2="{y(g):.1f}" stroke="{GRID}"/>'
                 f'<text x="{x0-8}" y="{y(g)+3:.1f}" text-anchor="end" font-size="10" fill="{MUTED}">{g:.1f}</text>')
    o.append(f'<text x="14" y="280" font-size="11" fill="{MUTED}" transform="rotate(-90 14 280)" text-anchor="middle">stack mean score over 22 rows (lower is better)</text>')
    o.append(f'<text x="{(x0+x1)/2:.1f}" y="592" font-size="11" fill="{MUTED}" text-anchor="middle">'
             'hollow = before the punisher fix, filled = after it; square = a follow-up run scored against the node '
             'it hangs off, diamond = an ablation of it, triangle = a diagnostic with no 22-row score, '
             'six-tick rail = one model on six training seeds</text>')
    xb = {c: XB[c] for c in ORDER}
    xa = {c: xb[c] + 70 for c in ORDER}
    pb = {c: (xb[c], y(MEAN[c]["before"])) for c in ORDER}
    pa = {c: (xa[c], y(MEAN[c]["after"])) for c in ORDER}
    xf = {}
    for k in STEPKEYS:      # FOLLOW order, so a chained run's parent is already placed
        p = FOL[k][1]
        xf[k] = (xa[p] if p in ORDER else xf[p]) + STEP_OFF[k]
    xf.update({k: xa[FOL[k][1]] + 26 for k in ABLKEYS})
    pf = {k: (xf[k], y(VMEAN[k]["after"])) for k in FKEYS}
    pos = lambda k: pa[k] if k in ORDER else pf[k]
    # lineage spines between the before states: main -> #179 -> #181 (gnn), main -> #174 -> #177 (gmlp)
    def spine(a, b, sp, dashed=False):
        (ax, ay), (bx, by) = pb[a], pb[b]
        dash = ' stroke-dasharray="6 4" stroke-opacity="0.55"' if dashed else ""
        o.append(f'<line class="t-{sp} v-success" x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{SPINE[sp]}" stroke-width="{1.1 if dashed else 2.4}"{dash}/>')
    spine("e_lin", "a_vnode", "gnn"); spine("a_vnode", "b_skip", "gnn")
    spine("e_lin", "d_kexo", "gmlp"); spine("d_kexo", "c_infl", "gmlp")
    spine("e_lin", "e_gnn", "gnn", dashed=True)
    # dotted step: best mean on record so far, in x order, over every scored result node
    res = sorted([(xa[c], MEAN[c]["after"]) for c in ORDER] + [(xf[k], VMEAN[k]["after"]) for k in STEPKEYS])
    best, pts = math.inf, []
    for xr, m in res:
        pts.append((xr, y(best if best < math.inf else m)))
        best = min(best, m)
        pts.append((xr, y(best)))
    pts.append((x1, y(best)))
    d = " ".join(f"{'M' if i == 0 else 'L'} {xx:.1f} {yy:.1f}" for i, (xx, yy) in enumerate(pts))
    o.append(f'<path d="{d}" fill="none" stroke="#8f8e89" stroke-dasharray="2 4"/>')
    # punisher-fix edges and nodes
    for c in ORDER:
        sp = CASE[c][1]; cls = "" if c == "e_lin" else f" t-{sp}"
        (bx, by), (ax, ay) = pb[c], pa[c]
        o.append(f'<line class="v-success{cls}" x1="{bx:.1f}" y1="{by:.1f}" x2="{ax:.1f}" y2="{ay:.1f}" stroke="{FIX}" stroke-width="2.4"/>')
        tip = esc(NODE_TIP[c])
        halo = (f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="12.5" fill="none" stroke="{NOISE}" stroke-width="1.2" '
                f'stroke-dasharray="2 3"/>' if c in INSIDE else "")
        o.append(f'<a href="#story-rebaseline" class="node v-success{cls}" data-tip="before: {tip}">'
                 f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="7.5" fill="{PAPER}" stroke="{SPINE[sp]}" stroke-width="2"/>'
                 f'<text x="{bx:.1f}" y="{by-13:.1f}" text-anchor="middle" font-size="9" fill="{MUTED}">{esc(CASE[c][5])}</text></a>')
        o.append(f'<a href="#story-punisher" class="node v-success{cls}" data-tip="after the punisher fix: {tip}">{halo}'
                 f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="7.5" fill="{SPINE[sp]}" stroke="{PAPER}" stroke-width="1.5"/>'
                 f'<text x="{ax:.1f}" y="{ay+18:.1f}" text-anchor="middle" font-size="9" fill="{MUTED}">{f3(MEAN[c]["after"])}</text></a>')
    # the k-one-hot switch head is borrowed from the gaussian-MLP line: mark where it came from
    (kx, ky), (dx, dy) = pf["i_kexo"], pa["d_kexo"]
    o.append(f'<line class="t-gmlp" x1="{dx:.1f}" y1="{dy:.1f}" x2="{kx:.1f}" y2="{ky:.1f}" stroke="{SPINE["gmlp"]}" '
             f'stroke-width="1.4" stroke-dasharray="3 4" stroke-opacity="0.8"/>')
    # PR #195's six arms: one model, six training seeds, on the frontier's own stack.
    # A rail, not a node -- it is a measurement of the scoreboard, not a step along it.
    sx = xa["b_skip"] + SEED_RAIL
    arm_y = sorted((VMEAN["arm_" + a]["after"], a) for a in ARMS)
    rtip = esc("PR #195, a measurement rather than an experiment: the frontier's contributor architecture "
               "trained six ways (five fresh seeds plus the shipped artifact) and run through one stack with "
               "everything else held identical. 22-row means "
               + " / ".join(f3(m) for m, _ in arm_y) + f"; six-arm mean {f3(VMEAN_SS)} against the shipped "
               f"{f3(VMEAN['arm_shipped']['after'])}, which is the lowest of the six. A typical row moves by "
               f"sd {SD_ROW:.3f} on the training draw alone and ten of the 22 cannot be gated on one run.")
    o.append(f'<g class="node t-gnn" data-tip="{rtip}">'
             f'<line x1="{sx:.1f}" y1="{y(arm_y[0][0]):.1f}" x2="{sx:.1f}" y2="{y(arm_y[-1][0]):.1f}" '
             f'stroke="{NOISE}" stroke-width="1.6"/>'
             + "".join(f'<line x1="{sx-5:.1f}" y1="{y(m):.1f}" x2="{sx+5:.1f}" y2="{y(m):.1f}" '
                       f'stroke="{NOISE}" stroke-width="{2.2 if a == "shipped" else 1.3}"/>'
                       for m, a in arm_y)
             + f'<circle cx="{sx:.1f}" cy="{y(VMEAN_SS):.1f}" r="3.6" fill="{NOISE}" stroke="{PAPER}" stroke-width="1.2"/>'
             f'<text x="{sx:.1f}" y="{y(arm_y[-1][0])-9:.1f}" text-anchor="middle" font-size="8.5" fill="{NOISE}">6 retrains</text>'
             f'<text x="{sx-7:.1f}" y="{y(VMEAN_SS)+3:.1f}" text-anchor="end" font-size="8" fill="{NOISE}">{f3(VMEAN_SS)}</text>'
             '</g>')
    o.append(f'<line class="t-gnn" x1="{pa["b_skip"][0]:.1f}" y1="{pa["b_skip"][1]:.1f}" x2="{sx:.1f}" '
             f'y2="{y(VMEAN_SS):.1f}" stroke="{NOISE}" stroke-width="1" stroke-dasharray="2 3" stroke-opacity="0.7"/>')
    # the follow-up runs: squares, hung off the node they were scored against
    for k in STEPKEYS:
        par = FOL[k][1]; (px, py) = pos(par); (fx, fy) = pf[k]
        story = {192: "ceiling", 190: "switchport", 196: "timeout",
                 197: "simtimeout", 198: "recal"}[FOL[k][3]]
        o.append(f'<line class="t-gnn" x1="{px:.1f}" y1="{py:.1f}" x2="{fx:.1f}" y2="{fy:.1f}" stroke="{FCOL["step"]}" stroke-width="2"/>')
        tip = esc(node_tip(k) + f" | verdict: {re.sub('<[^>]+>', '', FOL[k][7])}")
        ring = (f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="11.5" fill="none" stroke="{ALERT}" stroke-width="1.3" stroke-dasharray="3 3"/>'
                if k in FAILED else "")
        halo = (f'<rect x="{fx-11:.1f}" y="{fy-11:.1f}" width="22" height="22" rx="4" fill="none" stroke="{NOISE}" '
                f'stroke-width="1.2" stroke-dasharray="2 3"/>' if k in INSIDE else "")
        o.append(f'<a href="#story-{story}" class="node t-gnn" data-tip="{tip}">{ring}{halo}'
                 f'<rect x="{fx-6.5:.1f}" y="{fy-6.5:.1f}" width="13" height="13" rx="2" fill="{FCOL["step"]}" stroke="{PAPER}" stroke-width="1.5"/>'
                 f'<text x="{fx:.1f}" y="{fy-14:.1f}" text-anchor="middle" font-size="9" fill="{MUTED}">{FOL[k][5]}</text>'
                 f'<text x="{fx:.1f}" y="{fy+19:.1f}" text-anchor="middle" font-size="9" fill="{MUTED}">{f3(VMEAN[k]["after"])}</text></a>')
    # the ablations of the frontier's shared-noise machinery: diamonds on a short rail
    ax0 = xf[ABLKEYS[0]]
    (px, py) = pa[FOL[ABLKEYS[0]][1]]
    ylo = max(pf[k][1] for k in ABLKEYS); yhi = min(pf[k][1] for k in ABLKEYS)
    o.append(f'<path class="t-gnn" d="M {px:.1f} {py:.1f} L {ax0:.1f} {ylo+10:.1f} L {ax0:.1f} {yhi:.1f}" fill="none" '
             f'stroke="{FCOL["abl"]}" stroke-width="1.2" stroke-dasharray="2 3"/>')
    for k in ABLKEYS:
        fx, fy = pf[k]
        story = "sharederr"
        tip = esc(node_tip(k) + " | ablation of the frontier stack, not a candidate for the ledger")
        o.append(f'<a href="#story-{story}" class="node t-gnn" data-tip="{tip}">'
                 f'<polygon points="{fx:.1f},{fy-6:.1f} {fx+6:.1f},{fy:.1f} {fx:.1f},{fy+6:.1f} {fx-6:.1f},{fy:.1f}" '
                 f'fill="{PAPER}" stroke="{FCOL["abl"]}" stroke-width="1.8"/>'
                 f'<text x="{fx-10:.1f}" y="{fy+3:.1f}" text-anchor="end" font-size="8" fill="{MUTED}">{esc(FOL[k][5])} {f3(VMEAN[k]["after"])}</text></a>')
    # the diagnostics strip: no score, so no height -- parked under the run each one informs
    for parent, lbl, story, tip in DIAGS:
        dxx, dyy = pa[parent]
        sp = CASE[parent][1]
        o.append(f'<line class="t-{sp}" x1="{dxx:.1f}" y1="{dyy+26:.1f}" x2="{dxx:.1f}" y2="{ydiag-9:.1f}" '
                 f'stroke="{MUTED}" stroke-width="1" stroke-dasharray="2 3" stroke-opacity="0.6"/>')
        o.append(f'<a href="#story-{story}" class="node t-{sp}" data-tip="{esc(tip)}">'
                 f'<polygon points="{dxx:.1f},{ydiag-6:.1f} {dxx+6:.1f},{ydiag+5:.1f} {dxx-6:.1f},{ydiag+5:.1f}" '
                 f'fill="{PAPER}" stroke="#e87ba4" stroke-width="1.8"/>'
                 f'<text x="{dxx:.1f}" y="{ydiag+17:.1f}" text-anchor="middle" font-size="8.5" fill="{MUTED}">{lbl}</text></a>')
    o.append(f'<text x="{x0}" y="{ydiag+5:.1f}" font-size="9" fill="{MUTED}">diagnostics (no 22-row score)</text>')
    o.append("</svg>")
    return "".join(o)


# ---------------------------------------------------------------- 3. all 22 scores
SCOLS = list(ORDER) + STEPKEYS
SCLBL = {**{c: CASE[c][5].replace("main · ", "") for c in ORDER},
         **{k: FOL[k][5] for k in STEPKEYS},
         # these columns are 17.6 units apart, so the three longest labels are abbreviated
         "m_tout": "tout", "n_simto": "sim0", "o_recal": "rho"}
SSTORY = {**{c: "rebaseline" for c in ORDER},
          **{k: {192: "ceiling", 190: "switchport", 196: "timeout", 197: "simtimeout",
                 198: "recal"}[FOL[k][3]] for k in STEPKEYS}}


def small_chart(r):
    ylog = lambda v: 98.6 - 29.89 * math.log(v)
    col = SLOT[ROWSLOT[r]]
    sd, mu = SEED_SD[r], SSMEAN[r]
    o = [f'<svg viewBox="0 0 256 150" font-family="system-ui, sans-serif">']
    # the measurement floor, drawn first so every marker sits on top of it: +-1 seed sd
    # around the six-arm mean of PR #195, i.e. how far this row moves on the training draw
    ntip = esc(f"{r} measurement floor (PR #195): six-arm mean {f3(mu)}, seed sd {sd:.3f}, "
               f"range {f3(F(ss_row[r]['min']))}-{f3(F(ss_row[r]['max']))} over six retrains of one "
               f"architecture. {ss_stab[r]['status']}; "
               f"{'not gateable on a single run' if r in UNGATE else 'gateable on a single run'}. "
               f"The band is +-1 seed sd around the six-arm mean; a segment inside it is dashed.")
    o.append(f'<g class="node" data-tip="{ntip}">'
             f'<rect x="30" y="{ylog(mu + sd):.1f}" width="220" height="{ylog(mu - sd) - ylog(mu + sd):.1f}" '
             f'fill="{NOISE}" fill-opacity="0.13"/>'
             f'<line x1="30" y1="{ylog(mu):.1f}" x2="250" y2="{ylog(mu):.1f}" stroke="{NOISE}" '
             f'stroke-opacity="0.5" stroke-dasharray="3 3"/></g>')
    o.append(f'<text x="140.0" y="13" text-anchor="middle" font-size="11" font-weight="600" fill="{col}">{r}</text>')
    o.append(f'<text x="250.0" y="13" text-anchor="end" font-size="7" fill="{NOISE}">'
             f'seed sd {sd:.3f}{" &middot; ungateable" if r in UNGATE else ""}</text>')
    if r == "RCE":
        o.append(f'<text x="140.0" y="22" text-anchor="middle" font-size="7" font-weight="600" fill="#4a3aa7">protected row</text>')
    for g in (1, 2, 5):
        o.append(f'<line x1="30" y1="{ylog(g):.1f}" x2="250" y2="{ylog(g):.1f}" stroke="{GRID}"/>'
                 f'<text x="26" y="{ylog(g)+3:.1f}" text-anchor="end" font-size="8" fill="{MUTED}">{g}</text>')
    o.append(f'<line x1="128.8" y1="26" x2="128.8" y2="134" stroke="{GRID}" stroke-dasharray="2 3"/>')
    for i, c in enumerate(SCOLS):
        x = 32 + i * 17.6
        step = c in STEPKEYS
        sp = SPINEOF[c]; mc = FCOL["step"] if step else SPINE[sp]
        b, a = VEC[c]["before"][r], VEC[c]["after"][r]
        base = "its own baseline" if step else "before the punisher fix"
        n_sd = row_sd(c, r)
        legible = n_sd >= 1
        tip = esc(f"{LABEL[c]} ({sp} spine) - {r} {f3(b)} -> {f3(a)} ({band_arrow(c, r)}), against {base}"
                  f" | {sgn(a - b)} = {n_sd:.2f} seed sd (floor {sd:.3f})"
                  f" | {'legible' if legible else 'NOT distinguishable from a retrain'}")
        if step:
            m1 = (f'<rect x="{x-5.4:.1f}" y="{ylog(b)-2.7:.1f}" width="5.4" height="5.4" rx="1" fill="{PAPER}" stroke="{mc}" stroke-width="1.4"/>'
                  f'<rect x="{x:.1f}" y="{ylog(a)-2.7:.1f}" width="5.4" height="5.4" rx="1" fill="{mc}" stroke="{PAPER}" stroke-width="1"/>')
        else:
            m1 = (f'<circle cx="{x-2.7:.1f}" cy="{ylog(b):.1f}" r="3.2" fill="{PAPER}" stroke="{mc}" stroke-width="1.4"/>'
                  f'<circle cx="{x+2.7:.1f}" cy="{ylog(a):.1f}" r="3.2" fill="{mc}" stroke="{PAPER}" stroke-width="1"/>')
        seg = "" if legible else ' stroke-dasharray="1.6 1.6"'
        o.append(f'<a href="#story-{SSTORY[c]}" class="node" data-tip="{tip}">'
                 f'<line x1="{x-2.7:.1f}" y1="{ylog(b):.1f}" x2="{x+2.7:.1f}" y2="{ylog(a):.1f}" stroke="{mc}" stroke-width="1.8"{seg}/>'
                 f'{m1}</a>')
        o.append(f'<text x="{x:.1f}" y="142" text-anchor="middle" font-size="6" fill="{MUTED}">{SCLBL[c]}</text>')
    o.append("</svg>")
    return "".join(o)


# ---------------------------------------------------------------- 4. breakdown
def breakdown_svg(cases, title, words=("before", "after")):
    ylog = lambda v: 275.3 - 94.07 * math.log(v)
    xs = [40 + i * (460 / (len(cases) - 1)) for i in range(len(cases))]
    o = [f'<svg viewBox="0 0 560 400" font-family="system-ui, sans-serif">'
         f'<text x="270.0" y="16" text-anchor="middle" font-size="12.5" font-weight="600" fill="{INK}">{title}</text>']
    for g in (1, 2, 5):
        o.append(f'<line x1="40" y1="{ylog(g):.1f}" x2="500" y2="{ylog(g):.1f}" stroke="{GRID}"/>'
                 f'<text x="35" y="{ylog(g)+3:.1f}" text-anchor="end" font-size="9" fill="{MUTED}">{g}</text>')
    for x, c in zip(xs, cases):
        o.append(f'<text x="{x:.1f}" y="392" text-anchor="middle" font-size="9" fill="{MUTED}">{SHORT[c]}</text>')
    labels = []
    def line(vals, col, cls, tip, dashed, width=1.6, op=0.7):
        pts = " ".join(f"{x:.1f},{ylog(v):.1f}" for x, v in zip(xs, vals))
        dash = ' stroke-dasharray="4 3"' if dashed else ""
        o.append(f'<polyline points="{pts}" fill="none" class="bline{cls}" stroke="{col}" stroke-width="{width}" stroke-opacity="{op}"{dash} data-tip="{esc(tip)}"/>')
    for r in ROWS:
        col = SLOT[ROWSLOT[r]]; cls = f" f-{ROWSLOT[r]}"
        bv = [VEC[c]["before"][r] for c in cases]; av = [VEC[c]["after"][r] for c in cases]
        line(bv, col, cls, f"{r} ({ROWSLOT[r]}) {words[0]}: " + " -> ".join(f2(v) for v in bv), True, 1.2, 0.5)
        line(av, col, cls, f"{r} ({ROWSLOT[r]}) {words[1]}: " + " -> ".join(f2(v) for v in av), False)
        labels.append((ylog(av[-1]), r, col, cls))
    mb = [VMEAN[c]["before"] for c in cases]; ma = [VMEAN[c]["after"] for c in cases]
    line(mb, INK, "", f"mean of all 22 rows, {words[0]}: " + " -> ".join(f3(v) for v in mb), True, 2, 0.6)
    line(ma, INK, "", f"mean of all 22 rows, {words[1]}: " + " -> ".join(f3(v) for v in ma), False, 3, 1)
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


BA_STAGES = ["before", "after", "ceiling", "timeout", "simtimeout"]
CEIL_RUN = {"b_skip": "f_ceil", "e_gnn": "h_cgnn"}   # which #192 run each column's ceiling figure is
# the two later runs that committed the suite's own figures, per column
LATE_SIM = {
    "timeout": (B_TO, {"b_skip": SIM["b_skip"] + "_timeout", "e_gnn": SIM["e_gnn"] + "_timeout"}, "m_tout"),
    "simtimeout": (B_ST, {"b_skip": SIM["b_skip"] + "_simtimeout", "e_gnn": SIM["e_gnn"] + "_simtimeout"}, "n_simto"),
}


def fig_files(r):
    vis = D / "plots/simulation" / (SIM["b_skip"] + "_curpun") / "evaluation/visuals"
    return sorted(f.name for f in vis.glob(f"{r}_*.jpg"))


def fig_path(c, st, fname):
    if st in LATE_SIM:
        branch, sims, _ = LATE_SIM[st]
        if c not in sims:
            return None
        try:
            return gitfile(branch, f"plots/simulation/{sims[c]}/evaluation/visuals/{fname}")
        except subprocess.CalledProcessError:
            return None
    if st == "ceiling":
        if c not in CE_SIM:
            return None
        try:
            return gitfile(B_CE, f"plots/simulation/{CE_SIM[c]}/evaluation/visuals/{fname}")
        except subprocess.CalledProcessError:
            return None
    d = D / "plots/simulation" / (SIM[c] + ("_curpun" if st == "after" else "")) / "evaluation/visuals" / fname
    return d if d.exists() else None


def ba_cards():
    EMBEDDED.clear(); MISSING.clear()
    rows = [r for r in ROWS if fig_files(r)]
    def stack_name(c): return CASE[c][3].split(",")[0]
    def figure(r, fname, c, st):
        d = fig_path(c, st, fname)
        if st in LATE_SIM:   # the later runs scored the frontier and the GNN reference
            tbl = {"timeout": to_ba, "simtimeout": st_ba}[st]
            col = {"b_skip": "frontier_after", "e_gnn": "ref_gnn_after"}.get(c)
            v = F(tbl[r][col]) if col else None
            m = sum(F(tbl[q][col]) for q in ROWS) / len(ROWS) if col else None
        else:
            k = CEIL_RUN.get(c) if st == "ceiling" else c
            stg = "before" if st == "before" else "after"
            v, m = (VEC[k][stg][r], VMEAN[k][stg]) if k else (None, None)
        who = {"before": "lagged punisher", "after": "current-contribution punisher",
               "ceiling": "+ the gave-the-maximum flag (PR #192)",
               "timeout": "+ the timed-out-player flag (PR #196)",
               "simtimeout": "+ the recorded 0 served to the contributor (PR #197)"}[st]
        num = f" ({r} {f3(v)}, mean {f3(m)})" if v is not None else ""
        cap = f"{st}: {esc(stack_name(c))}, {who}{num}"
        if st == "simtimeout" and c == "b_skip":
            cap += "; PR #198's recalibrated run reproduces this one byte for byte, so it has no column"
        if c.startswith("e_"):
            cap += "; the sim's figure shows every manager of that run, the GNN punisher is gnn_self"
        if d is not None and d.read_bytes()[:3] == b"\xff\xd8\xff":
            EMBEDDED.append((r, fname, c, st))
            return f'<figure><img src="{b64(d)}" loading="lazy" alt="{esc(fname)} ({c} {st})"><figcaption>{cap}</figcaption></figure>'
        MISSING.append((r, fname, c, st))
        why = ("the ceiling flag was only run on the frontier stack and the main-sweep reference" if st == "ceiling" else
               "the later runs were only made on the frontier stack and the main-sweep GNN reference" if st in LATE_SIM else
               "the source sim dir carries no visuals" if c == "d_kexo" else
               "no before figure exists: the source sims were scored with the 21-row suite and only rescored, not re-plotted")
        return (f'<figure><div style="border:1px dashed #e1e0d9;border-radius:6px;aspect-ratio:14/9;display:grid;place-items:center;'
                f'color:#898781;font-size:12px;text-align:center;padding:8px">no figure &mdash; {why}</div><figcaption>{cap}</figcaption></figure>')
    btns, cards = [], []
    for i, r in enumerate(rows):
        col = SLOT[ROWSLOT[r]]; on = i == 0
        style = f"border-color:{col};color:{PAPER if on else col};background:{col if on else 'none'}"
        btns.append(f'<button data-m="{r}" data-color="{col}" class="{"on" if on else ""}" style="{style}">{r}</button>')
        body = "".join('<div class="barow">'
                       + "".join(figure(r, f, c, st) for st in BA_STAGES for c in BA_STACKS)
                       + "</div>" for f in fig_files(r))
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
    def box(x, y, t1, t2, tint=None, title=None, pills=(), changed=False, alert=False):
        cx = x + 140
        fill = f'fill="{tint}" fill-opacity="0.14" stroke="{tint}"' if tint else 'fill="none" stroke="currentColor" stroke-opacity="0.6"'
        o.append("<g>" + (f"<title>{esc(title)}</title>" if title else "") +
                 (f'<rect x="{x-8}" y="{y-8}" width="296" height="78" rx="11" fill="none" stroke="{ALERT}" '
                  f'stroke-width="1.8" stroke-dasharray="7 3"/>'
                  f'<text x="{x-4}" y="{y-14}" font-size="9.5" font-weight="600" fill="{ALERT}">'
                  f'open escalation &#8212; maintainer</text>' if alert else "") +
                 (f'<rect x="{x-4}" y="{y-4}" width="288" height="70" rx="9" fill="none" stroke="{FCOL["step"]}" '
                  f'stroke-width="1.2" stroke-dasharray="4 4"/>' if changed else "") +
                 f'<rect x="{x}" y="{y}" width="280" height="62" rx="6" {fill}/>'
                 f'<text x="{cx}" y="{y+25}" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">{t1}</text>'
                 f'<text x="{cx}" y="{y+42}" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.75">{t2}</text>')
        right = x + 280 - 6
        for text, story in reversed(list(pills)):
            w = 38 if len(text) <= 4 else 46
            right -= w
            o.append(f'<a href="#story-{story}"><title>the story behind {esc(text)}</title>'
                     f'<rect x="{right}" y="{y-8}" width="{w}" height="17" rx="8" fill="#1a1a19"/>'
                     f'<text x="{right+w/2}" y="{y+4.5}" fill="{PAPER}" text-anchor="middle" font-size="10">{esc(text)}</text></a>')
            right -= 4
        o.append("</g>")
    def arrow_(cx, y1, y2, label=None, both=False):
        ms = ' marker-start="url(#arr)"' if both else ""
        o.append(f'<line x1="{cx}" y1="{y1}" x2="{cx}" y2="{y2}" stroke="currentColor" marker-end="url(#arr)"{ms}/>')
        if label: o.append(f'<text x="{cx+8}" y="{(y1+y2)/2+4}" font-size="10" fill="currentColor" opacity="0.75">{label}</text>')
    # bay 1: players
    bay(30, "PLAYERS BAY", "five tests since the re-baseline, none of them merged")
    box(70, 108, "contribution trunk", "gnn skip trunk kept; its lagged input was faked",
        None, f"PR #191 tested the gaussian-MLP's location-scale head against these 21 free logits and it lost: "
        f"copula-off variety {F(hs_head['e_skip_kexo_rho0']['var_cond_mean']):.2f} for the categorical trunk against "
        f"{F(hs_head['c_infl_rho0']['var_cond_mean']):.2f} (inflated) and {F(hs_head['d_kexo_rho0']['var_cond_mean']):.2f} (v2), human "
        f"{F(cl_dec['human']['var_cond_mean']):.2f}. The hypothesis was its own author's and it was falsified. "
        f"PR #197 then found the trunk's own lagged input was wrong in simulation: a timed-out player's contribution "
        f"reached it as the imputed 9 on {st_pb['timeouts']}/{st_pb['agent_rounds']} = {st_pb['timeout_rate']:.2%} of agent-rounds, "
        f"now the recorded 0. With the shared-noise machinery off that lifts the variety of states reached from "
        f"{F(st_dec[ST_ARM['parent_off']]['var_cond_mean']):.2f} to {F(st_dec[ST_ARM['cand_off']]['var_cond_mean']):.2f} "
        f"against the human {F(st_dec['human']['var_cond_mean']):.2f} with no randomness added -- direct progress on the "
        f"one defect the campaign has left. The run still failed its gate: the cost lands on the group-spread rows.",
        [("#191", "head"), ("#197", "simtimeout")], changed=True)
    arrow_(210, 170, 208, "per-agent marginals")
    box(70, 212, "group copula unit", "herding latent -- frozen; its strength now closed",
        SLOT["contribution"], f"PRs #186/#187/#188: the strength is right -- redrawn every round it reproduces the human within-group co-movement "
        f"({F(cl_dec['C']['resid_corr_all_rounds']):.3f} against {F(cl_dec['human']['resid_corr_all_rounds']):.3f}; as shipped, "
        f"{F(cl_dec['A']['resid_corr_all_rounds']):.3f}) -- but the episode-long persistence, never fitted, supplies most of the group-spread row by "
        f"compounding ({F(cl_lat['compounding_factor']):.1f} times). PR #189 froze both settings. "
        f"PR #198 unfroze the strength once, to test whether it had been calibrated in the presence of PR #197's defect: "
        f"it had not and could not have been, because it is fitted against human data where the recorded 0 was always "
        f"correct. The refit returns rho = {cr_par['rho']:.10f}, the shipped value to the last digit, all {CR_FIELDS} "
        f"params fields identical and the stamped artifact the same file by sha256. Every row moves by exactly zero. "
        f"The strength is closed as a route to the group-spread row; the shape is not.",
        [("#186", "sharederr"), ("#189", "protocol"), ("#198", "recal")], changed=True)
    arrow_(210, 274, 312)
    box(70, 316, "switch model", "joint exodus head; the k-one-hot port was rejected",
        None, f"PR #190 swapped in the gaussian-MLP line's k-one-hot group-size head: SB {F(sk_cmp['SB']['before']):.3f} -> {F(sk_cmp['SB']['after']):.3f}, "
        f"CG {F(sk_cmp['CG']['before']):.3f} -> {F(sk_cmp['CG']['after']):.3f}, but RCD {F(sk_cmp['RCD']['before']):.3f} -> {F(sk_cmp['RCD']['after']):.3f} "
        f"and RSA {F(sk_cmp['RSA']['before']):.3f} -> {F(sk_cmp['RSA']['after']):.3f}. Failed gate 1 and the protected row -- "
        f"and its declared target SC moved {sv_sd('#190', 'SC'):.2f} seed sd, inside the floor PR #195 later measured. "
        f"PR #197 expected the timeout defect to reach this slot too and probed it with every predict call wrapped: on this "
        f"stack the switch encoder reads {', '.join(ST_MODEL['switch']['reads'])} and never a contribution key at all, "
        f"so the defect does not reach it. Measured, not assumed.",
        [("#190", "switchport"), ("#197", "simtimeout")], changed=True)
    arrow_(210, 378, 470)
    o.append(f'<text x="218" y="430" font-size="10.5" fill="currentColor">contributions c_t (0..20)</text><circle cx="210" cy="475" r="3.5" fill="currentColor"/>')
    # bay 2: punisher
    bay(420, "PUNISHER BAY", "the manager: one joint decision per group-round")
    box(460, 108, "feature intake -- fixed, then extended twice", "c_t, then flags for c = 20 and for no input at all",
        FIX, f"PR #184: the punisher reads the contribution it punishes; before that, prev contribution only, one round late "
        f"(CV log loss lin {CV['lin'][0]:.4f} -> {CV['lin'][1]:.4f}, GNN {CV['gnn'][0]:.4f} -> {CV['gnn'][1]:.4f}). PR #192 added an "
        f"indicator for c_t = 20 on both families ({F(ce_cv0['log_loss']):.4f} -> {F(ce_cv['log_loss']):.4f}); the experiment failed its gate, so whether it merges is a maintainer's call. "
        f"PR #196 added the other end: a flag for whether the player gave any input at all, so a timeout is no longer "
        f"read as a chosen zero. Real managers punished a timed-out player {TO_HUM['timeout']:.2f}% of the time and a genuine zero "
        f"{TO_HUM['zero_all']:.1f}%. It earns its place on the linear family (CV {TO_CV['lin'][0]:.4f} -> {TO_CV['lin'][1]:.4f}, locked test "
        f"{TO_CV['lin_test'][0]:.4f} -> {TO_CV['lin_test'][1]:.4f}) and nothing on the graph one ({TO_CV['gnn'][0]:.4f} -> {TO_CV['gnn'][1]:.4f}, "
        f"t = -0.46). The declared row RCC moved {arrow('m_tout', 'RCC')} "
        f"({row_sd('m_tout', 'RCC'):.2f} seed sd, legible) and still held its band.",
        [("#184", "punisher"), ("#192", "ceiling"), ("#196", "timeout")], changed=True)
    o.append(f'<text x="600" y="184" text-anchor="middle" font-size="9.5" fill="currentColor" opacity="0.6" text-decoration="line-through">was: prev contribution (round t-1) only</text>')
    arrow_(600, 192, 208)
    box(460, 212, "multinomial logistic trunk / GNN punisher", f"31 levels - test log loss {CV['lin_test'][0]:.3f} -> {CV['lin_test'][1]:.3f} (floor {CV['floor']:.3f})",
        None, f"Still linear in the contribution: its OLS response in self-play is {F(ce_mech['frontier after']['OLS c_t']):.3f} per point against the human "
        f"{F(ce_mech['human']['OLS c_t']):.3f}, a little over half, and the ceiling flag moves that by at most {CE_SLOPE_MAX:.3f} in any condition. Live, separately declarable defect.",
        [("#192", "ceiling")], changed=True)
    arrow_(600, 274, 312, "per-agent multinomial CDFs")
    box(460, 316, "severity copula unit", f"shared group-round latent - rho {RHO['before']:.3f} -> {RHO['after']:.3f}",
        SLOT["contribution"], f"PR #184 re-stamps the #160 copula on the new bundle: rho {RHO['before']:.4f} -> {RHO['after']:.4f} (SE {RHO['se']:.4f}); it rose instead of dropping. "
        f"PR #192 carried it over unchanged, as its plan required.",
        [("#184", "copula")])
    arrow_(600, 378, 470)
    o.append(f'<text x="608" y="430" font-size="10.5" fill="currentColor">punishments p_t (0..30)</text><circle cx="600" cy="475" r="3.5" fill="currentColor"/>')
    # bay 3: scoring
    bay(810, "SCORING BAY", "the 22-row evaluation suite, 500 repeats, seed 42")
    box(850, 108, "RCE: punishment response slope", "protected row -- repaired, then made noise-aware",
        "#4a3aa7", f"RCE: OLS slope of next-round change on punishment received, per contribution band; human {' / '.join(sgn(v) for v in HUMAN_SLOPES)}; ceiling {CEIL['RCE']:.4f}. "
        f"PR #189 added two qualifications after the halving clause fired twice on its first outing, once on an improvement. "
        f"PR #195 then measured the row itself: RCE's own seed sd is {SEED_SD['RCE']:.3f} and it is one of the ten rows that "
        f"cannot be judged on a single run -- it lands in band <= 1 in exactly one arm of six. Its band-drop clause now needs a "
        f"drop larger than that, and its sign clause is retired on the 10-14 and 15-19 bands, where retraining an unchanged "
        f"model flips the sign on its own ({sgn(F(ss_agg['rce_slope_10-14']['min']))} to {sgn(F(ss_agg['rce_slope_10-14']['max']))} "
        f"across six retrains).",
        [("RCE", "rce"), ("#189", "protocol"), ("#195", "noise")], changed=True)
    arrow_(990, 170, 208, "beside RCB, not instead")
    box(850, 212, "held-out teacher-forced test", f"5 folds: held-out {HO['pooled_held_out']:.3f} vs in-sample {HO['pooled_in_sample']:.3f}",
        "#e87ba4", f"PR #183: the contributor's reaction to punishment is learned, not memorised (pooled held-out {HO['pooled_held_out']:.4f}, in-sample {HO['pooled_in_sample']:.4f}, self-play {PR179_SELFPLAY_RCB:.3f}).",
        [("#183", "holdout")])
    arrow_(990, 274, 312, "so the closed loop is the culprit")
    box(850, 316, "ledger + frozen surface", "six reruns, seven follow-ups: five missed their gate",
        "#6b6a66", f"PR #184 stage D reset every baseline; PR #189 added the two shared-noise settings to the surface no experiment may move, "
        f"and made contributor changes judged with the machinery switched off, on the variety measure against the human {F(cl_dec['human']['var_cond_mean']):.1f}. "
        f"PR #195 reset the scoreboard again: the frontier's baseline is now the six-arm mean of its own reseed ensemble "
        f"(22-row mean {VMEAN_SS:.4f} not {VMEAN['arm_shipped']['after']:.4f}, rows <= 1 {int(F(ss_agg['rows_le1']['mean']))} not "
        f"{VLE1['f_ceil']['after']}, RCC {SSMEAN['RCC']:.4f} not {VEC['arm_shipped']['after']['RCC']:.4f}), because the shipped "
        f"draw ranks first of six on every aggregate while its training fit is ordinary -- selection, not merit. "
        f"OPEN ESCALATION, and this box is where it lives: the suite scores a simulated timeout and drops a human one. "
        f"{ESC_PLAIN}. A validity column in per_round.parquet plus one where() in load_sim fixes it; "
        f"no agent may touch that directory, so it needs the maintainer.",
        [("#184", "rebaseline"), ("#189", "protocol"), ("#195", "noise")], changed=True, alert=True)
    arrow_(990, 378, 470)
    o.append(f'<text x="998" y="430" font-size="10.5" fill="currentColor">scores.csv (22 rows)</text><circle cx="990" cy="475" r="3.5" fill="currentColor"/>')
    # round loop
    o.append('<line x1="30" y1="475" x2="1170" y2="475" stroke="currentColor" stroke-width="1.6" marker-end="url(#arr)"/>'
             '<text x="600" y="497" text-anchor="middle" font-size="11" fill="currentColor">round t: contributions c_t -&gt; the manager sees c_t and punishes -&gt; common good = 1.6 &#215; sum c_t &#8722; sum p_t -&gt; payoffs</text>'
             '<path d="M 1170 522 L 30 522" stroke="currentColor" stroke-dasharray="6 5" stroke-opacity="0.6" fill="none" marker-end="url(#arr)"/>'
             '<text x="600" y="542" text-anchor="middle" font-size="10.5" fill="currentColor" opacity="0.75">feeds round t+1 as prev contribution / prev punishment (24 rounds x 100 episodes, seed 42); every 4th round the switch bay regroups</text>'
             '<text x="600" y="574" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.6">the players\' models are the ones their PRs shipped; the re-baseline changed only the manager\'s input</text>'
             '<text x="600" y="590" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.6">the dashed outlines are the parts the eleven later PRs touched: of those, only the protocol change, the freeze and the noise-aware gates landed</text>'
             f'<text x="600" y="608" text-anchor="middle" font-size="10" fill="{ALERT}">the red outline is an open escalation for the maintainer: '
             'the scoring bay drops a human timeout from every metric and scores a simulated one, so the two sides are not measured alike</text></svg>')
    return "".join(o)


# ---------------------------------------------------------------- 7. leaderboard
FSLOT = {"f_ceil": "punisher", "g_clin": "punisher", "h_cgnn": "punisher", "i_kexo": "switch",
         "j_off": "contribution", "k_redraw": "contribution", "l_seeds": "contribution",
         "m_tout": "punisher", "n_simto": "contribution", "o_recal": "contribution"}
NPILL = (f'<span class="pill" style="background:{NOISE}">%s</span>')


def noise_cell(k):
    """What survives the measurement floor, in the note column of every record."""
    bits = [f"&Delta; mean {mean_sd(k):.2f} seed sd",
            f"&Delta; rows &le; 1 {le1_sd(k):.2f} sd",
            f"{n_legible(k)}/22 rows move more than their own seed sd"]
    exact = all(VEC[k]["after"][r] == VEC[k]["before"][r] for r in ROWS)
    pill = NPILL % ("every delta exactly zero &mdash; identical artifact" if exact else
                    "&Delta; mean inside the floor" if k in INSIDE else
                    f"&Delta; mean {mean_sd(k):.1f}&times; the floor")
    return (f'<div style="margin:3px 0 1px">{pill}</div>'
            f'<span class="means">{" &middot; ".join(bits)}</span>')


def up_detail(k):
    """Band upgrades and downgrades, each with how far it moved in its own seed sd."""
    def fmt(rows):
        return ", ".join(f"{r} {row_sd(k, r):.2f} sd" + ("" if row_sd(k, r) >= 1 else " [inside the floor]")
                         for r in rows.split(", ") if r)
    return fmt(ST[k]["up_rows"]), fmt(ST[k]["down_rows"])


def lb_rows():
    out = []
    for c in ORDER:
        u, d = up_detail(c)
        out.append(dict(pr=CASE[c][2], label=CASE[c][3], slot="punisher", stack=CASE[c][1],
                        note=f"{CASE[c][4]}{noise_cell(c)}", **{**ST[c], "up_rows": u, "down_rows": d}))
    for k, par, sp, pr, kind, short, label, verdict, note in FOLLOW:
        u, d = up_detail(k)
        out.append(dict(pr=pr, label=label, slot=FSLOT[k], stack=sp,
                        note=f"{verdict} &middot; {note} &middot; baseline: {SHORT[par]}{noise_cell(k)}",
                        **{**ST[k], "up_rows": u, "down_rows": d}))
    return out


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
    # helpers for the seven PRs that followed the re-baseline
    cg = {a: F(cl_sc["CG"][a]) for a in "ABC"}
    persist_share = (cg["C"] - cg["A"]) / (cg["B"] - cg["A"])
    variety_share = F(cl_dec["B"]["var_cond_mean"]) / F(cl_dec["human"]["var_cond_mean"])
    gain = lambda m, d: F(hs_gain[(m, d)]["gain_e"])
    grng = lambda m: f"{min(gain(m, d) for d in GAIN_D):.2f}-{max(gain(m, d) for d in GAIN_D):.2f}"
    blk = lambda a: " / ".join(f"{F(hs_blk[(a, b)]['sd_group_mean']):.2f}" for b in ("1-8", "9-16", "17-24"))
    hh = lambda a, k: F(hs_head[a][k])
    cm = lambda a, k: F(ce_mech[a][k])
    cb = lambda r, c: F(ce_ba[r][c])
    sk = lambda r, st: F(sk_cmp[r][st])
    under = abs(F(ce_rcc["human"][2])) / abs(F(ce_rcc["frontier after"][2]))
    # helpers for the four PRs that followed the four-step programme
    arm = lambda a: VMEAN["arm_" + a]["after"]
    tf = lambda a, k: F(to_tf[a][k])
    sd_of = lambda k, r: f"{arrow(k, r)} ({row_sd(k, r):.2f} seed sd)"
    stab = lambda status: [r for r in ROWS if ss_stab[r]["status"] == status]
    se_ratio = [F(ss_agg[f"rce_slope_{b}"]["sd"]) / F(ss_agg[f"rce_slope_{b}"]["within_run_se"]) for b in BANDS]
    sd_var = lambda a: F(st_dec[ST_ARM[a]]["var_cond_mean"])
    sd_cg = lambda a: F(st_dec[ST_ARM[a]]["cg_ratio"])
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
        f"RPA, the manager's policy row, {rng(rpa_b)} &rarr; {rng(rpa_a)}: from the 1-2 band to at or under the noise ceiling in all six runs, a row the lag had fabricated outright. RCB falls by {abs(max(rcb_d)):.2f} to {abs(min(rcb_d)):.2f} everywhere (#179 {arrow('a_vnode', 'RCB')}, #181 {arrow('b_skip', 'RCB')}: the band PR #181 missed by 4.34% is cleared by the punisher fix alone). RCC does not move ({', '.join(f'{sgn(rcc_d[c], 2)} in {CASE[c][5]}' for c in ORDER)}): the full contributors still being punished are exactly the population that row is made of. That residual is what PR {pr_link(192)} went after: it removed the invented population outright and moved RCC by {cb('RCC', 'frontier_after') - cb('RCC', 'frontier_before'):+.3f}, the largest move that row has ever had, and still missed its band &mdash; see the ceiling card."))
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
        f"A calibrated punisher for the PR stacks, and a question for a successor: whether the mood is a manager-level latent (one per episode) or a round-level one is testable on the human data with the same script and matters for PD (in the main-sweep stacks without the copula it still reads {arrow('e_lin', 'PD')} and {arrow('e_gnn', 'PD')}). "
        f"PR {pr_link(187)} has since run exactly that test on the <em>contributors'</em> copula and found a one-round shock with a two-thirds echo and no lasting part at all; the manager's severity copula was not measured, so the question is still open where it was asked. PR {pr_link(189)} then froze both settings of the contributors' unit, and PR {pr_link(192)} carried the severity rho over unchanged."))
    rce_after = "; ".join(f"{CASE[c][5]} {f3(float(S[c]['after']['RCE']))}, signs {signs(c, 'after')}" for c in ORDER)
    ex = dict(gave=3, pun=5, nxt=8)
    rate = ex["pun"] / (20 - ex["gave"])
    out.append(story("rce", "The RCE row: the punishment response slope, now protected",
        "evaluation", "#4a3aa7", f"branch rcb-alternative-response-slope, merged by PR {pr_link(184)} &middot; row 22 of the evaluation suite &middot; <code>reports/rcb_alternative_comparison.md</code>",
        f"The learning manager's only lever is punishment, so what the simulated players must get right is how they respond to it. RCB was the row scoring that: it sorts punished players by punishment <em>rate</em> (punishment divided by the shortfall from 20) and compares the average next-round change in each rate bin with the human one. The rate mixes how much a player gave with how hard they were hit: a rate above 1 is reached by a zero contributor punished 20 and by a 17 contributor punished 4, who react in opposite directions. So a stack can match the bin averages with the right mix of players and no within-level response at all. Across the 40 stacks of the sweep and the PRs, RCB and RCE rank the stacks almost independently (Spearman {SPEARMAN:.2f}); RCE tracks how many of the four human response signs a stack reproduces ({SPEAR_SIGNS['RCE']:.2f}), RCB does not ({SPEAR_SIGNS['RCB']:.2f}).",
        f"RCE takes the same punished non-full contributors, sorts them by what they gave (0-4, 5-9, 10-14, 15-19) and within each band fits a straight line of next-round change on punishment received, keeping the slope: how many more points a player gives per extra point of punishment. Humans comply at low levels and withdraw at high ones: {' / '.join(sgn(v) for v in HUMAN_SLOPES)}. The score is the human-frequency-weighted mean absolute slope gap over the four bands, divided by the human-vs-human ceiling of {CEIL['RCE']:.4f}. It sits beside RCB, not instead of it, and is the first protected row of the protocol.",
        [f"A worked example. A player gave {ex['gave']}, was punished {ex['pun']}, and gave {ex['nxt']} next round: a change of +{ex['nxt'] - ex['gave']}. RCB computes the rate {ex['pun']} / {20 - ex['gave']} = {rate:.2f}, files the +{ex['nxt'] - ex['gave']} into the (0.25, 0.5] bin, and compares that bin's average with the human one. RCE files the same player into the 0-4 band, where the +{ex['nxt'] - ex['gave']} is one point on the regression of change against punishment; only the slope of that line is scored.",
         f"Power. Two halves of the human data differ by {CEIL['RCE']:.3f} in slope, about three quarters of the human slopes themselves, so a simulation whose players ignore punishment scores {SYNTH['none']:.2f} ('minor deviation') and one with half the human response scores {SYNTH['half']:.2f}, at the ceiling. RCB's ceiling is {CEIL['RCB']:.3f}, but its sensitivity is to the level of the change among the punished, not to the dose. That is why RCE is protected on the statistic as well as on the band.",
         f"The protection rule (notes/autoresearch.md &sect;2): an experiment may not band-downgrade RCE against its baseline, may not flip any of the four band slopes away from the human sign, and may not halve any band's slope magnitude. Any of the three is a FAIL whatever the gates say; punisher-slot and punishment-response experiments are judged on RCE for their band upgrade. The third clause misfired twice on its first outing and PR {pr_link(189)} has since qualified it twice over &mdash; see the protocol card; the rule as written still cost PR {pr_link(190)} its result.",
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
         "Caveats: the 32-stack sweep matrix was not re-run, so the ranking rule of &sect;3 stays defined on the pre-fix matrix until the maintainer refreshes it; the copula rho rose (the previous card); the sign flips in #179 and the GNN-punisher run are bands with |slope| under 0.04 before and after.",
         f"Since this card was written the frontier's mean has been beaten by four runs, every one of which failed its gate: "
         f"the ported switch head reads {VMEAN['i_kexo']['after']:.4f}, the ceiling flag {VMEAN['f_ceil']['after']:.4f}, the "
         f"timeout flag {VMEAN['m_tout']['after']:.4f} (the lowest on record, {VLE1['m_tout']['after']} rows at or under the "
         f"ceiling) and the serving fix {VMEAN['n_simto']['after']:.4f}, against the frontier's {VMEAN['b_skip']['after']:.4f} "
         f"and {VLE1['b_skip']['after']}. None is a lineage step: a gate is a band change on a row declared in advance, and "
         f"none got one. The ledger's frontier is still the #181 skip stack as re-baselined here.",
         f"<b>Two of those comparisons should never have been read as results at all.</b> {pr_link(195)} has since measured "
         f"how far this evaluation moves when nothing changes but the contributor's training seed: the 22-row mean by "
         f"sd {SD_MEAN:.4f} and the rows-at-the-ceiling count by sd {SD_LE1:.2f}, swinging from "
         f"{int(F(ss_agg['rows_le1']['min']))} to {int(F(ss_agg['rows_le1']['max']))}. The ceiling flag's mean move is "
         f"{mean_sd('f_ceil'):.2f} of that floor and the switch port's {mean_sd('i_kexo'):.2f} &mdash; <b>neither is "
         f"distinguishable from a retrain</b>, and the &ldquo;lowest mean on record&rdquo; line above ranks draws as much as "
         f"models. The same measurement resets this card's own baselines: the frontier stack's honest scores are the six-arm "
         f"mean of its own ensemble, 22-row mean {VMEAN_SS:.4f} rather than {VMEAN['arm_shipped']['after']:.4f} and "
         f"{int(F(ss_agg['rows_le1']['mean']))} rows at the ceiling rather than {VLE1['f_ceil']['after']}. The six reruns "
         f"above changed a punisher artifact rather than a contributor seed, so the floor applies to them only as a lower "
         f"bound &mdash; but {len([c for c in ORDER if c in INSIDE])} of the six moved their mean by less than it "
         f"({', '.join(SHORT[c].replace('main &middot; ', '') for c in ORDER if c in INSIDE)}), which is as much as to say "
         f"the punisher fix is visible in <em>which rows</em> moved, not in the stack average."],
        ["<code>scripts/data_analysis/curpun_rebaseline.py</code> &mdash; writes rebaseline_table.{csv,md} and rce_bands.csv from the before and after scores",
         "<code>scripts/simulation/run_curpun_reruns.sh</code> &mdash; submit / fetch / evaluate for the five _curpun configs, with the gmlp-lineage checkout for cases c and d",
         "<code>configs/simulation/manager_testing/*_curpun.yml</code> &mdash; the five rerun configs, byte-identical to their sources except for the punisher path and output dir",
         "<code>plots/data_analysis/evaluation/punisher_current_contr/</code> &mdash; the tables this page is built from, plus before/&lt;case&gt;/scores.csv and mechanism_selfplay.csv",
         "<code>plots/simulation/&lt;source&gt;_curpun/</code> &mdash; the five sim dirs with per_round.parquet and the 22-row evaluations"],
        "A ledger whose baselines are the closed-loop states of the accepted contributor and switch models under a manager that punishes what it sees; a frontier ranking under those baselines; and a first protected row from which every successor is judged."))
    # ---------------------------------------------- the seven PRs after the re-baseline
    out.append(story("sharederr", "Three answers on the shared error: a variety source wearing a correlation's clothes",
        "shared error", SLOT["contribution"],
        f"PRs {pr_link(186)}, {pr_link(187)} and {pr_link(188)} &middot; three experiments run side by side on the frontier stack "
        f"&middot; the copula unit of the players bay",
        f"Two members of one group face the same situation and then decide for themselves, so once the situation is known their two choices "
        f"should be independent. The contributor is fitted one player at a time and nothing in it produces togetherness, so a shared random "
        f"number per group is mixed into every member's draw: strength rho = {se_sum['rho_copula_json']:.4f}, fitted; persistence one draw held "
        f"for all 24 rounds, never fitted. On the human games the raw within-group correlation of contributions is "
        f"{F(ms_base['raw contribution, plain Pearson']['value']):.2f} and what is left once the network's own expectation is taken out is "
        f"{F(ms_base['level residual, plain Pearson']['value']):.3f} &mdash; small, and yet the machinery is worth several score bands on the rows "
        f"that measure how far groups drift apart. Three explanations were on the table: the players' own randomness is too small, the model "
        f"cannot see part of the situation, or the model is uncertain about itself.",
        f"All three were tested at once. {pr_link(186)} reran the frontier stack three ways: machinery on as it ships (arm A), off (arm B), and "
        f"redrawn every round with the same strength, which keeps the correlation and removes the persistence (arm C). {pr_link(187)} measured the "
        f"leftover co-movement on the 50 human games and asked what observable group state explains it and whether any of it lasts. "
        f"{pr_link(188)} trained five copies of the contributor with different random seeds, measured their disagreement, and ran a simulation "
        f"that draws one copy per game in place of the shared number.",
        [f"The randomness inside a round is already right; the variety of situations reached is not. With the machinery off, the players' "
         f"leftover variance is {F(cl_dec['B']['var_resid']):.1f} against the human {F(cl_dec['human']['var_resid']):.1f}, but the variance of the "
         f"model's own expectation over the states it visits is {F(cl_dec['B']['var_cond_mean']):.1f} against {F(cl_dec['human']['var_cond_mean']):.1f} "
         f"&mdash; {variety_share:.0%}, about two thirds. That is the defect, and it is one defect, not three.",
         f"Strength and persistence do different jobs. Redrawn every round (arm C) the shared number reproduces the human within-group co-movement "
         f"almost exactly ({F(cl_dec['C']['resid_corr_all_rounds']):.3f} against the human {F(cl_dec['human']['resid_corr_all_rounds']):.3f}; with the "
         f"machinery off it is {F(cl_dec['B']['resid_corr_all_rounds']):.4f}). On the group-spread row CG that buys "
         f"{cg['B']:.2f} &rarr; {cg['C']:.2f} of the {cg['B']:.2f} &rarr; {cg['A']:.2f} the shipped setting buys; the episode-long persistence carries "
         f"the remaining {persist_share:.0%}, and it does it by compounding &mdash; a push of {F(cl_lat['resid_on_z_slope']):.2f} contribution points in "
         f"the round the number is drawn becomes a shift of {F(cl_lat['cell_slope']):.2f} in the group's level, {F(cl_lat['compounding_factor']):.1f} "
         f"times over. Over all 22 rows: mean {VMEAN['j_off']['after']:.3f} with the machinery off, {VMEAN['k_redraw']['after']:.3f} redrawn each round, "
         f"{VMEAN['b_skip']['after']:.3f} as it ships.",
         f"Can the missing part of the situation just be handed to the model? Only about a seventh of it. The best three observable group facts &mdash; "
         f"how many of your group were punished last round, which way the group is drifting, how far apart its members are &mdash; explain "
         f"{F(ms_joint['mle']['share']):.0%} of the leftover co-movement (interval {F(ms_joint['mle']['share_lo']):.0%} to "
         f"{F(ms_joint['mle']['share_hi']):.0%}); every legal candidate together explains {F(ms_ref['all legal candidates']['share_mle']):.0%}.",
         f"And it is not a lasting group trait. Between two different members of a group the shared deviation is "
         f"{F(ms_pers['before partialling']['lag0_moment']):.3f} within a round, {F(ms_pers['before partialling']['lag1_moment']):.3f} one round later "
         f"and {F(ms_pers['before partialling']['lag>=2_moment']):+.4f} pooled over every pair two or more rounds apart "
         f"(interval {F(ms_pers['before partialling']['lag>=2_lo']):+.4f} to {F(ms_pers['before partialling']['lag>=2_hi']):+.4f}); the static share is "
         f"{F(ms_pers['before partialling']['static_share']):+.3f} (interval {F(ms_pers['before partialling']['static_share_lo']):+.3f} to "
         f"{F(ms_pers['before partialling']['static_share_hi']):+.3f}). A one-round shock with a two-thirds echo into the next round, and nothing "
         f"beyond. The shipped shape has no counterpart in the data and is kept only because nothing replaces the variety it supplies.",
         f"Nor is it the model's own uncertainty, which closes the Bayesian route. Five copies trained with different seeds on the same "
         f"{se_sum['n_episodes']} games disagree by {F(se_dis['sd_E_between_seeds']['mean']):.2f} contribution points per player-round, "
         f"{F(se_sum['ratio_sd_E_to_sd_pred']):.0%} of one model's own spread. That is worth a shared-draw strength of {se_sum['implied_rho']:.4f} "
         f"(interval {se_sum['implied_rho_ci'][0]:.4f} to {se_sum['implied_rho_ci'][1]:.4f}) against the fitted {se_sum['rho_copula_json']:.4f}, and it "
         f"decays fast: the between-seed disagreement correlates {se_sum['persistence_lag_corr']['1']:.2f} one round on, about half, where the shipped "
         f"latent is held constant. Run as a simulation, one copy per game scores like having nothing at all: CG {se_sim['CG']:.2f} against "
         f"{cg['A']:.2f} with the machinery and {cg['B']:.2f} without, 22-row mean {VMEAN['l_seeds']['after']:.3f}.",
         f"Caveat, and it matters: five copies trained on the same {se_sum['n_episodes']} games is the narrowest kind of ensemble, so "
         f"{se_sum['implied_rho']:.4f} is a <b>lower bound</b> on the model's uncertainty, not a measurement of it. Resampling the games themselves "
         f"would be wider and has not been tried. The conclusion that the ensemble cannot substitute for the machinery is safe by a factor of five; "
         f"the exact number is not.",
         f"One later check this card passes: {pr_link(195)}'s noise floor is small next to everything measured here. The three "
         f"arms move the 22-row mean by {mean_sd('j_off'):.1f}, {mean_sd('k_redraw'):.1f} and {mean_sd('l_seeds'):.1f} times "
         f"the floor and the group-spread row by more again, so the decomposition above is not a story about training draws. "
         f"The one number that should be read with the floor beside it is the ceiling count, which swings by "
         f"{SD_LE1:.1f} rows on the seed alone."],
        ["<code>scripts/data_analysis/copula_closed_loop_variance.py</code> &mdash; the three-arm ablation, the variance decomposition and the latent regression",
         "<code>scripts/data_analysis/copula_missing_state.py</code> and its analysis companion &mdash; the candidate screen, the joint fit and the persistence bootstrap",
         "<code>src/aimanager/simulation/ensemble_ah.py</code> &mdash; SeedEnsembleAH, one trained copy drawn per game",
         "<code>plots/data_analysis/evaluation/copula_closed_loop/</code>, <code>.../copula_missing_state/</code>, <code>plots/data_analysis/copula_seed_ensemble/</code> &mdash; every number above"],
        f"Three routes closed and one defect named, for a day of cluster time and no new model worth keeping. The strength and the persistence are "
        f"now frozen ({pr_link(189)}), and a change to the players is judged with the machinery switched off, on the variety measure, because with it "
        f"on the row that was being used cannot tell you whether the change helped."))
    out.append(story("head", "Step 1: the output design that was supposed to hold the line, and did the opposite",
        "step 1, falsified", "#e87ba4",
        f"PR {pr_link(191)} &middot; three short simulations, nothing trained &middot; the hypothesis was its own author's",
        f"A simulated player picks a number from 0 to 20. The graph-network line scores all 21 possibilities separately; the Gaussian-MLP line "
        f"predicts a centre and a width and draws from a bell curve. The argument for building a combined model around the bell curve was that when "
        f"a simulated game wanders somewhere no real game went, 21 unconnected scores have nothing tying them together and should sag back toward the "
        f"training average, while a single centre keeps tracking. If true, that would explain the contraction the shared-error work had just isolated, "
        f"and it would decide which line to build on.",
        f"No training and no new models: the two Gaussian stacks' existing post-fix simulations were measured with {pr_link(186)}'s decomposition, "
        f"plus copula-off counterparts so the comparison is like for like, plus an off-manifold probe that pushes a group's recent contribution level "
        f"2, 4 and 6 points away from anything real and records how much each model's predicted expectation still moves.",
        [f"It is wrong in its mechanism. On the common evaluation set the 21-score design's gain is {grng('skip_categorical')} across the six shifts "
         f"and it is the only one that rises at the extremes; the inflated bell curve reads {grng('infl')} and the plain one {grng('v2')}. No head's "
         f"gain decays with distance, which is the specific failure the hypothesis predicted for the 21 scores.",
         f"It is wrong in its consequence. With the shared-noise machinery off, the 21-score design holds the most variety in the states it reaches, "
         f"{hh('e_skip_kexo_rho0', 'var_cond_mean'):.2f} against {hh('c_infl_rho0', 'var_cond_mean'):.2f} (inflated) and "
         f"{hh('d_kexo_rho0', 'var_cond_mean'):.2f} (v2), human {F(cl_dec['human']['var_cond_mean']):.2f}. Normalised by each model's own fit to human "
         f"histories, which is the reading most favourable to the bell curve, the inflated head ties at {hh('c_infl_rho0', 'retention'):.3f} "
         f"(interval {F(hs_boot['c_infl_rho0']['ret_lo']):.3f} to {F(hs_boot['c_infl_rho0']['ret_hi']):.3f}) against "
         f"{hh('e_skip_kexo_rho0', 'retention'):.3f} and v2 is clearly lower at {hh('d_kexo_rho0', 'retention'):.3f}.",
         f"The sharper finding is not the one it was aimed at. Real groups keep drifting further apart as a game runs: the spread of group averages "
         f"goes {blk('human (infl)')} across the three thirds of an episode. With the machinery off, both bell-curve arms stall or reverse in the last "
         f"third ({blk('c_infl_rho0')}; {blk('d_kexo_rho0')}) where the 21-score arm keeps climbing ({blk('e_skip_kexo_rho0')}). All three fall well "
         f"short. <b>The defect is a failure of late divergence, not a level offset</b>, which is what the successor card is now pointed at.",
         f"Within every model the shared-noise machinery is worth about twice what the choice of output design is worth: the variety measure goes "
         f"{hh('c_infl_rho0', 'retention'):.3f} &rarr; {hh('c_infl', 'retention'):.3f} and {hh('d_kexo_rho0', 'retention'):.3f} &rarr; "
         f"{hh('d_kexo', 'retention'):.3f} when it is switched on, a bigger move than any head difference.",
         f"The one confound was closed rather than argued about. The two lines also used different group-switching components, so a matched pair was "
         f"run: it moves the 21-score design's variety measure by "
         f"{hh('e_skip_kexo_rho0', 'retention') - hh('skip B (no copula)', 'retention'):.3f}, two orders of magnitude less than the gap it was supposed "
         f"to explain away, and the ordering does not change.",
         f"One component of the bell-curve family does earn its place, inside that family. Without the extra weight it puts on the corners and on "
         f"repeating last round's number, the plain bell curve emits "
         f"{(F(hs_recon['human (v2)'][1]) / F(hs_recon['human (v2)'][2]) - 1) * 100:.0f}% less randomness on real games than its own errors turn out to "
         f"need ({F(hs_recon['human (v2)'][2]):.2f} predicted against {F(hs_recon['human (v2)'][1]):.2f} realised). A design that gets the corners for "
         f"free has nothing to take from it."],
        ["<code>scripts/data_analysis/head_state_spread.py</code> &mdash; five stages, everything regenerable from committed inputs",
         "<code>plots/data_analysis/evaluation/head_state_spread/</code> &mdash; headline.csv, gain_curves.csv, retention_bootstrap.csv, round_blocks.csv",
         "<code>notes/autoresearch_log/head-state-spread-diagnostic.md</code> &mdash; the log, including the teacher-forcing validation table quoted above"],
        f"The combined head is dead as argued for, at the cost of three two-minute simulations and no training at all &mdash; and the author's own "
        f"hypothesis is the thing it killed. There is no 22-row score here and no gate: this is a diagnostic, which is why it sits on the strip under "
        f"the tree rather than on a spine."))
    out.append(story("ceiling", "Step 2: fixing the manager at the ceiling found the fault was in the players",
        "step 2, failed its row", FIX,
        f"PR {pr_link(192)} &middot; one flag added to both punisher families &middot; two short retrainings, two simulations "
        f"&middot; run on the frontier stack and on the main-sweep reference",
        f"Real managers almost never punish someone who gave the full 20 &mdash; {cm('human', 'P(p>0|c_t=20)'):.1%} of the time &mdash; and when they "
        f"do they punish hard, {cm('human', 'E[p|p>0] 20'):.1f} points. Both simulated punishers read the amount given as a single number on a scale, "
        f"so neither can make a sharp break at exactly 20; they read the ceiling off the 15-19 band below it. The frontier stack therefore punished "
        f"full contributors {cm('frontier before', 'P(p>0|c_t=20)'):.1%} of the time and too lightly "
        f"({cm('frontier before', 'E[p|p>0] 20'):.2f} points). RCC, the reaction at the ceiling, is built entirely out of punished full contributors, "
        f"so the simulation was inventing the population that row measures &mdash; and RCC was the one row the punisher timing fix had not moved.",
        f"Confirm the diagnosis on the human games first: an explicit gave-the-maximum indicator is worth "
        f"{F(ce_logit['+max']['contribution_max_coef']):+.2f} on the log-odds scale and takes the fitted rate at the ceiling from "
        f"{F(ce_logit['linear']['fit_P(p>0|c=20)']):.3f} onto the observed {F(ce_logit['+max']['fit_P(p>0|c=20)']):.3f}, with the fit over all 31 "
        f"punishment levels improving from {F(ce_cv0['log_loss']):.4f} to {F(ce_cv['log_loss']):.4f}. Then add the indicator to both punisher families, "
        f"retrain, carry the severity copula over unchanged, and rerun the frontier stack and the main-sweep reference. A companion "
        f"gave-nothing flag was tested and dropped.",
        [f"The mechanism is now essentially exact and <b>the experiment still fails</b>. The punish rate at the ceiling goes to "
         f"{cm('frontier after', 'P(p>0|c_t=20)'):.3f} against the human {cm('human', 'P(p>0|c_t=20)'):.3f} &mdash; right to a thousandth &mdash; and "
         f"the severity there from {cm('frontier before', 'E[p|p>0] 20'):.2f} to {cm('frontier after', 'E[p|p>0] 20'):.2f} against the human "
         f"{cm('human', 'E[p|p>0] 20'):.2f}. The declared row moved {cb('RCC', 'frontier_after') - cb('RCC', 'frontier_before'):+.3f}, "
         f"{cb('RCC', 'frontier_before'):.4f} &rarr; {cb('RCC', 'frontier_after'):.4f}, the largest move it has ever had, and did not cross a band. "
         f"Gate 1 was declared on the band, so the verdict is FAIL. Gate 2 passes ({VMEAN['f_ceil']['after']:.4f} against a ceiling of "
         f"{F(ce_verdict['gate2_ceiling']):.4f}) and the protected row holds on the gated stack, every band inside one standard error "
         f"(largest {max(ce_se[0]['change_in_se'].values()):.2f}).",
         f"Splitting the row says where the rest of the distance is, and it is not the manager's. The invented population is gone: "
         f"{ce_rcc['frontier before'][6]} of full contributors punished before, {ce_rcc['frontier after'][6]} after, {ce_rcc['human'][6]} in the real "
         f"games. What is left is that a punished full contributor in the simulation gives up {abs(F(ce_rcc['frontier after'][2])):.2f} points the next "
         f"round where a real person gives up {abs(F(ce_rcc['human'][2])):.2f}: <b>the contributors under-react to a heavy ceiling punishment by about "
         f"{under:.1f} times</b>, and no change to the manager can touch it. RCC is the only row that measures it, because RCE's population is the "
         f"punished <em>non-full</em> contributors by construction.",
         f"A second defect is untouched and now separately live: the simulated manager's punishment falls with contribution at "
         f"{cm('frontier after', 'OLS c_t'):.3f} per point against the human {cm('human', 'OLS c_t'):.3f}, a little over half the human strength. The "
         f"flag changes that by at most {CE_SLOPE_MAX:.3f} in any condition and was never meant to. Nobody should read &ldquo;the manager was fixed at "
         f"the ceiling&rdquo; as &ldquo;the manager's response to contribution was fixed&rdquo;.",
         f"What went the wrong way: the older binned reaction row got worse, {cb('RCB', 'frontier_before'):.3f} &rarr; "
         f"{cb('RCB', 'frontier_after'):.3f}, precisely because removing the punishments at the ceiling removed the rows where its denominator was "
         f"smallest, leaving it shaped by the slope defect above. Group spread rose on the frontier ({cb('CG', 'frontier_before'):.3f} &rarr; "
         f"{cb('CG', 'frontier_after'):.3f}) while falling on the linear reference ({cb('CG', 'ref_lin_before'):.2f} &rarr; "
         f"{cb('CG', 'ref_lin_after'):.2f}) and rising slightly on the graph one ({cb('CG', 'ref_gnn_before'):.2f} &rarr; "
         f"{cb('CG', 'ref_gnn_after'):.2f}).",
         f"The reference reruns are reported, not gated. The graph-punisher one improved broadly on the way past: mean "
         f"{VMEAN['h_cgnn']['before']:.4f} &rarr; {VMEAN['h_cgnn']['after']:.4f}, rows at or under the ceiling {VLE1['h_cgnn']['before']} &rarr; "
         f"{VLE1['h_cgnn']['after']}; the linear one {VMEAN['g_clin']['before']:.4f} &rarr; {VMEAN['g_clin']['after']:.4f}. The protected-row clause "
         f"fired on both of them, and both firings were wrong &mdash; that is the next card.",
         f"<b>Read under the floor {pr_link(195)} later measured, this card keeps its verdict and loses two of its numbers.</b> "
         f"The declared row is the part that survives: RCC moved {sv_sd('#192', 'RCC'):.2f} seed sd, legibly and in the right "
         f"direction, and genuinely did not cross its band &mdash; and RCC is one of the twelve rows that can be gated on a "
         f"single run at all. What does not survive is the context around it. The 22-row mean moved "
         f"{sv_sd('#192', '22-row mean'):.2f} of its floor and the rows-at-the-ceiling count {sv_sd('#192', 'rows'):.2f} of "
         f"its own, so &ldquo;{VLE1['f_ceil']['before']} &rarr; {VLE1['f_ceil']['after']} rows at the ceiling&rdquo; is a "
         f"property of the draw, not of the flag; and the two rows the branch reported as collateral, RCB at "
         f"{sv_sd('#192', 'RCB'):.2f} sd and CG at {sv_sd('#192', 'CG'):.2f} sd, are inside the floor in the other direction "
         f"&mdash; they were not damage either. The baseline it beat was itself the six-arm minimum on RCC "
         f"({VEC['arm_shipped']['after']['RCC']:.4f} against the ensemble's {SSMEAN['RCC']:.4f}), which is the hardest "
         f"version of the comparison and the one recorded."],
        ["<code>scripts/data_analysis/punisher_ceiling_check.py</code> &mdash; the diagnosis on the real games, before anything was built",
         "<code>scripts/baselines/handcrafted_grid.py</code> &mdash; the derived contribution_max indicator and its legality; "
         "<code>src/aimanager/generic/data.py</code> and <code>src/aimanager/manager/api_manager.py</code> &mdash; the same on the graph path",
         "<code>plots/data_analysis/evaluation/punisher_ceiling_fix/</code> &mdash; before_after.{csv,md}, the two mechanism tables, the human logit fit",
         "<code>notes/autoresearch_log/punisher-ceiling-fix.md</code> &mdash; the log and the RCC decomposition table quoted above"],
        f"The manager's half of RCC is finished and the other half is named, with a decomposition table ready to serve as the baseline for whoever "
        f"declares against the contributors. Whether artifacts from a failed experiment are worth merging anyway is a maintainer's call: the mechanism "
        f"evidence says yes and the gate says no."))
    out.append(story("switchport", "Step 3: the right number of switches, the wrong people leaving",
        "step 3, failed", SLOT["switch"],
        f"PR {pr_link(190)} &middot; one component swapped, one two-minute simulation, no retraining &middot; the k-one-hot head of the "
        f"gaussian-MLP line on the frontier trunk",
        f"Every fourth round players may move to the other group. Real people never empty a group of five or more and often abandon a singleton, so "
        f"group size matters in a lumpy way that one smooth curve cannot follow. The gaussian-MLP line has a component that gives each possible group "
        f"size its own free setting, and the stack built around it posts the best switching numbers in the set. Do those numbers belong to the "
        f"component or to the players it was paired with? Swapping only the component across is the cheapest way to find out.",
        f"The trained artifact from {pr_link(174)} was reused rather than retrained &mdash; the switch model is fitted on the human games alone and "
        f"its inputs are game observables, so nothing in its training depends on which contributor it is later paired with. Only "
        f"<code>switch_model</code>, the output dir and the figure name differ from the frontier config; no copula was recalibrated, because the "
        f"noise settings are frozen for this programme.",
        [f"Every measure that is purely about who ends up in which group improved, and every measure of how players respond got worse. Switch timing "
         f"gained a band, {sk('SB', 'before'):.3f} &rarr; {sk('SB', 'after'):.3f}; group spread fell sharply, {sk('CG', 'before'):.3f} &rarr; "
         f"{sk('CG', 'after'):.3f}; segregation improved without crossing a band, {sk('SC', 'before'):.3f} &rarr; {sk('SC', 'after'):.3f}. The "
         f"switching-pull row went the wrong way, {sk('RCD', 'before'):.3f} &rarr; {sk('RCD', 'after'):.3f}. The 22-row mean is the lowest on record at "
         f"{VMEAN['i_kexo']['after']:.4f} with {VLE1['i_kexo']['after']} rows at or under the ceiling, gate 2 passes, and the experiment still fails, "
         f"because the mean is not what it declared.",
         f"The protected row was violated. All four human signs survive ({sk_rce['after']['signs_vs_human']}) and the score stays in its band "
         f"({sk('RCE', 'before'):.3f} &rarr; {sk('RCE', 'after'):.3f}), but the 10-14 band slope falls from "
         f"{F(sk_rce['before']['slope_10-14']):+.3f} to {F(sk_rce['after']['slope_10-14']):+.3f}, "
         f"{abs(F(sk_rce['after']['slope_10-14'])) / abs(F(sk_rce['before']['slope_10-14'])):.0%} of its baseline, below the half threshold. Under the "
         f"protected-row rule that is a failure whatever the gates say.",
         f"The row split is the useful result. Only segregation, switch timing and group spread are decided by the switching component alone. The "
         f"switching-pull row is the slope of a <em>switcher's contribution change</em> on the gap to the group they join: the component picks who "
         f"goes, the contributors decide what they then give. Declaring that row as a switch-slot target was a mis-specification in the plan itself, "
         f"independent of how the run came out.",
         f"Undeclared, and the largest single regression: switching after being punished, {sk('RSA', 'before'):.3f} &rarr; {sk('RSA', 'after'):.3f}. "
         f"Right number of switches, right group sizes, wrong people leaving after a punishment. That is a concrete mismatch and the most informative "
         f"follow-up in the set.",
         f"Caveat, and it is load-bearing: one seed, one run, no repeats, so the small movements carry no measured spread. The gate-1 miss is safe "
         f"&mdash; the switching-pull row moves by {abs(F(sk_cmp['RCD']['delta'])):.2f} and neither target comes near a band edge &mdash; but "
         f"<b>the protected-row violation rests on a single run's band slope</b> over {int(F(sk_rce['after']['n_10-14'])):,} observations, with no "
         f"estimate of how much that slope moves between seeds. It is recorded as the rule reads; anyone wanting to overturn it should refit that band "
         f"across seeds rather than argue about it.",
         f"What it cannot settle: whether the component needs the gaussian-MLP contributor or clashes with this one specifically. Both readings fit "
         f"this run and they imply different successors. Pairing it with a third set of players separates them, at one simulation each.",
         f"<b>The caveat above has since been measured, and it cuts both ways.</b> {pr_link(195)} retrained the frontier's "
         f"contributor six times and scored all six: the declared target SC moved {sv_sd('#190', 'SC'):.2f} seed sd here, "
         f"which is <b>inside the floor</b> &mdash; the row this branch was judged on never really moved, so the gate-1 miss "
         f"was never close and the &ldquo;improvement without crossing a band&rdquo; was not an improvement either. The other "
         f"declared target RCD moved {sv_sd('#190', 'RCD'):.2f} sd, outside it and genuinely the wrong way. The largest "
         f"regression, RSA at {sv_sd('#190', 'RSA'):.2f} sd, is real and reaches a value no arm of the reseeded ensemble "
         f"reaches &mdash; that one stands exactly as written. And the protected-row violation this branch was failed on "
         f"survives the arithmetic without being settled by it: the 10-14 band slope moved {sv_sd('#190', 'RCE 10-14'):.2f} "
         f"seed sd, past the floor, but that band runs {sgn(F(ss_agg['rce_slope_10-14']['min']))} to "
         f"{sgn(F(ss_agg['rce_slope_10-14']['max']))} across six retrains of an unchanged model, which is why the sign clause "
         f"is now retired there. The branch asked for exactly this measurement; it got it, and its verdict is unchanged."],
        ["<code>src/aimanager/generic/joint_exodus.py</code> &mdash; the size encoding ported across with a default so older components keep loading",
         "<code>configs/simulation/manager_testing/23_2g8a_switch_kexo_port_...yml</code> &mdash; the frontier config with one line changed",
         "<code>plots/data_analysis/evaluation/switch_kexo_port/</code> &mdash; the 22-row comparison and the band slopes",
         "<code>notes/autoresearch_log/switch-kexo-port.md</code> &mdash; the log, the gate outcome and the no-error-bars note"],
        f"A rule for every future switching experiment: declare only the rows the component actually decides. And one clear counter-example to the "
        f"premise the whole lineage merge rested on, that the two lines' strengths are separable and additive."))
    out.append(story("protocol", "Step 4: a freeze, and a safety rule that failed two experiments it should not have",
        "step 4, repaired", "#4a3aa7",
        f"PR {pr_link(189)} &middot; not an experiment &middot; the four-step plan and the rules the other three are judged by "
        f"&middot; <code>doc/plans/post-rebaseline-program.md</code>",
        f"Two protocol problems had to be fixed before the other three steps could be read. First, a copula recalibration used to ride along with "
        f"every change to the players, so a player experiment and a noise experiment moved at once and could not be told apart. Second, RCE is "
        f"protected: no experiment may drop its band, flip any of the four human signs, or halve a band's slope magnitude. That last clause is a "
        f"magnitude test on a signed quantity, and the bands it guards are thin &mdash; the middle one has had the wrong sign in every condition since "
        f"the held-out test, learned or not.",
        f"The freeze landed as planned: the correlation strength and the persistence are now part of the surface no experiment may modify, altering "
        f"either is its own declared experiment, and a change to the contributor is judged with the machinery switched off, on the variety measure "
        f"against the human {F(cl_dec['human']['var_cond_mean']):.1f}. The magnitude clause then misfired twice on its first outing and was amended "
        f"afterwards.",
        [f"Firing one, on the linear reference stack of {pr_link(192)}: the 15-19 band, the thinnest in the suite, on a change of "
         f"{ce_se[1]['change_in_se']['15-19']:.2f} standard errors. On a single run that is not distinguishable from noise.",
         f"Firing two, on the graph reference stack: a slope moving from {lead(ce_sl[2]['before'][3]):+.3f}, which is the <em>wrong</em> sign, to "
         f"{lead(ce_sl[2]['after'][3]):+.3f}, toward the human {HUMAN_SLOPES[2]:+.3f} &mdash; a change of {ce_se[2]['change_in_se']['10-14']:.2f} "
         f"standard errors, and an improvement read as an erosion, because a magnitude test cannot tell a slope passing through zero in the right "
         f"direction from one wasting away.",
         f"Two qualifications were added. The clause does not fire when the new slope is closer to the human value than the old one was, and it fires "
         f"only when the change exceeds one pooled standard error of the two slopes. <b>Neither changes a verdict already recorded</b>: both failing "
         f"experiments failed their declared row independently, so the repair cannot be read as rescuing anything. The violation recorded on "
         f"{pr_link(190)} is untouched either way: that band's slope moved <em>away</em> from the human value rather than toward it, so the first "
         f"qualification does not apply, and that branch reports no standard errors at all, which is the gap its own log flags.",
         f"Every experiment that touches the row now reports each band's slope with its standard error, its row count and the change in pooled "
         f"standard errors, so a reader can tell erosion from noise without re-running anything. The frontier stack of {pr_link(192)} is the first to "
         f"be read that way: largest band change {max(ce_se[0]['change_in_se'].values()):.2f} standard errors, so the row is intact.",
         f"Deliberately not done: {pr_link(187)} showed the human dependence has the shape of a one-round echo rather than the episode-long "
         f"persistence that ships. Changing the shape now would cost the group-spread row and the high-contribution withdrawal slope with nothing in "
         f"place to replace the variance they borrow from it. Keep the shipped shape, stop letting it move, and measure past it.",
         f"<b>The rule has since been amended a second time, and this one is bigger than the first.</b> {pr_link(195)} "
         f"measured what the whole scoreboard does when only the training seed changes, and &sect;2 was rewritten around it. "
         f"A band upgrade now counts only if the target row also moves by more than that row's seed sd, and ten rows cannot "
         f"serve as a gate-1 target on a single run at all. The <b>symmetry rule</b> applies the same threshold to losses: a "
         f"row worsening by less than its floor is not a cost, it is a retrain &mdash; because a threshold applied to gains "
         f"alone would make improvement nearly impossible, one row having to beat the noise to help while twenty-one could "
         f"hurt by luck. Gate 2 is deliberately left where it is ({SD_MEAN:.4f} against a margin of about "
         f"{0.10 * VMEAN['arm_shipped']['after']:.3f}, roughly two floors) as the backstop against many sub-floor losses "
         f"accumulating. And the protected row takes the same treatment: RCE's band-drop clause fires only on a drop larger "
         f"than {SEED_SD['RCE']:.3f}, and its sign clause is retired on the 10-14 and 15-19 bands, where an unchanged model "
         f"flips sign on its own. None of this reverses a recorded verdict either &mdash; every failure failed on a band its "
         f"target did not cross."],
        ["<code>notes/autoresearch.md</code> &sect;2 and &sect;8 &mdash; the protected row, its two qualifications, the freeze and the frozen surface",
         "<code>doc/plans/post-rebaseline-program.md</code> &mdash; the four steps as declared and then as they came out",
         "<code>src/aimanager/evaluation_suite/metrics.py</code> &mdash; the per-band standard errors and row counts the reports now carry"],
        f"A rule that fails experiments over differences too small to be real, and once over an improvement, is worse than no rule: it teaches the "
        f"people it governs to argue with it instead of respecting it. It is now stated with the arithmetic that makes a firing readable."))
    out.append(story("noise", "The noise floor: how far the whole scoreboard moves when nothing changes",
        "measurement", NOISE,
        f"PR {pr_link(195)} &middot; six arms, no training, no model proposed &middot; the frontier stack, contributor "
        f"artifact swapped and everything else held identical",
        f"Every verdict on this page rests on one training run and one simulation, and the protocol had no notion of "
        f"run-to-run variability. The three experiments before this one were decided on margins that might sit inside it. "
        f"So: take the contributor architecture the frontier already accepts, train it six ways &mdash; {pr_link(188)}'s five "
        f"seeds plus the shipped artifact &mdash; and run all six through the same stack under the same simulation seed, "
        f"the same punisher, the same switch model and the same copula parameters, carried bit for bit rather than refitted. "
        f"Nothing else differs. Whatever the 22 rows then do is the measurement error of the scoreboard.",
        f"Six arms, {int(F(ss_lev['seed_1']['n'])):,} agent-rounds each, 22 rows each with identical noise-ceiling denominators. "
        f"The six contributors are genuinely different models (pairwise, only "
        f"{SS_COINCIDE[0]:.1f}-{SS_COINCIDE[1]:.1f}% of their contribution draws coincide) "
        f"and genuinely exchangeable: on the training objective the five members score "
        f"{min(SS_CV[0]):.4f}-{max(SS_CV[0]):.4f} and the shipped artifact {SS_CV[1]:.4f}, an ordinary draw.",
        [f"<b>A typical row moves by sd {SD_ROW:.3f} and spans {SPAN_ROW:.2f} end to end</b> (per-row sds "
         f"{min(SEED_SD.values()):.3f} to {max(SEED_SD.values()):.3f}). The 22-row mean moves by sd {SD_MEAN:.4f} "
         f"(range {F(ss_agg['mean_22']['range']):.4f}) against a gate-2 margin of about "
         f"{0.10 * VMEAN['arm_shipped']['after']:.3f} &mdash; that gate sits at roughly two floors, which is why it stands. "
         f"And the count of rows at or under the human-vs-human ceiling swings from {int(F(ss_agg['rows_le1']['min']))} to "
         f"{int(F(ss_agg['rows_le1']['max']))}, sd {SD_LE1:.2f}: <b>the least stable number in the suite, and one several "
         f"verdicts had quoted as though it were a property of the model</b>.",
         f"<b>Ten of the 22 rows cannot be judged from a single training run.</b> A band boundary lies inside one seed sd of "
         f"their six-arm mean and the six arms genuinely land in two bands: {', '.join(UNGATE)}. "
         f"{len(stab('always <= 1'))} rows are always at the ceiling ({', '.join(stab('always <= 1'))}), "
         f"{len(stab('never <= 1'))} never are ({', '.join(stab('never <= 1'))}), and the rest flip on the draw alone. "
         f"<b>RCE, the protected row, is one of the ten</b>: it sits in band &lt;= 1 in exactly one arm of six "
         f"(its sd is {SEED_SD['RCE']:.3f}, its six-arm mean {SSMEAN['RCE']:.4f} against the shipped "
         f"{VEC['arm_shipped']['after']['RCE']:.4f}).",
         f"Every clause of the protected-row rule turns out to be decided by the training draw. Two of RCE's four band "
         f"slopes change sign across six retrains of the same model: the 10-14 band runs "
         f"{sgn(F(ss_agg['rce_slope_10-14']['min']))} to {sgn(F(ss_agg['rce_slope_10-14']['max']))} and the 15-19 band "
         f"{sgn(F(ss_agg['rce_slope_15-19']['min']))} to {sgn(F(ss_agg['rce_slope_15-19']['max']))}. On all four bands the "
         f"seed spread is larger than the <em>within-run</em> sampling error ({min(se_ratio):.2f}-{max(se_ratio):.2f} times "
         f"it): retraining moves a slope further than resampling the same run does, and the standard errors every report "
         f"carries had been understating the real uncertainty.",
         f"<b>The frontier is the bottom of its own spread, and that is not a coincidence anyone can rule out.</b> The shipped "
         f"contributor is first of six on the 22-row mean ({VMEAN['arm_shipped']['after']:.4f} against the six-arm "
         f"{VMEAN_SS:.4f}), first on rows &lt;= 1 ({VLE1['f_ceil']['after']} against {int(F(ss_agg['rows_le1']['mean']))}), "
         f"holds the six-arm minimum on {ss_rank['shipped']['rows_best']} of 22 rows including RCC "
         f"({VEC['arm_shipped']['after']['RCC']:.4f} against the members' {min(F(ss_row['RCC'][a]) for a in ARMS[:5]):.4f}"
         f"-{max(F(ss_row['RCC'][a]) for a in ARMS[:5]):.4f}) and lands "
         f"{F(ss_rank['seed_5']['mean_c_err']) / F(ss_rank['shipped']['mean_c_err']):.0f} times closer to the human "
         f"contribution level than the nearest member &mdash; while its training fit is unremarkable. The obvious mechanism "
         f"is selection: it became the frontier by scoring well on this evaluation, and every candidate since has been "
         f"measured against a favourable tail. The protocol's answer is to make the frontier's baseline the six-arm mean "
         f"vector, which is a change to the scoreboard and not to any model.",
         f"<b>What this does to the verdicts already on this page.</b> {pr_link(190)}'s declared target SC moved "
         f"{sv_sd('#190', 'SC'):.2f} seed sd &mdash; inside the floor, so its gate-1 miss was never in doubt but its "
         f"target never really moved either; its largest regression RSA at {sv_sd('#190', 'RSA'):.2f} sd is real. "
         f"{pr_link(192)}'s 22-row mean moved {sv_sd('#192', '22-row mean'):.2f} sd and its rows &lt;= 1 count "
         f"{sv_sd('#192', 'rows'):.2f} sd &mdash; neither is distinguishable from a retrain &mdash; while its declared "
         f"RCC move at {sv_sd('#192', 'RCC'):.2f} sd is, and the two rows it was faulted for worsening "
         f"(RCB {sv_sd('#192', 'RCB'):.2f} sd, CG {sv_sd('#192', 'CG'):.2f} sd) are not. Two further experiments "
         f"({pr_link(193)} and {pr_link(194)}) sit between {pr_link(192)} and the timeout run and are not plotted here; "
         f"the floor re-reads them too, and it explains most of what {pr_link(194)} was faulted for.",
         f"<b>What it does not cover.</b> The simulation draw, the punisher and switch slots' own retrain spread, and any "
         f"interaction between them. Six draws give an sd with about 30% relative uncertainty, and every omission points the "
         f"same way: the true run-to-run variability is larger than this, not smaller. A second seed per candidate was "
         f"considered and rejected &mdash; it doubles the cost of exactly the marginal experiments and leaves the "
         f"<em>baseline</em> single-seeded, which makes the comparison worse, not better."],
        ["<code>scripts/data_analysis/seed_spread_noise_floor.py</code> &mdash; the six-arm comparison, the per-row spread "
         "and the re-reading of the recorded verdicts",
         "<code>plots/data_analysis/evaluation/seed_spread_noise_floor/per_row.csv</code> &mdash; the per-row seed sd this "
         "page now prints beside every score; <code>aggregates.csv</code>, <code>arm_ranks.csv</code>, "
         "<code>ceiling_stability.csv</code>, <code>verdicts.csv</code>",
         "<code>scripts/artificial_humans/carry_contribution_copula_params.py</code> &mdash; the copula stamped, not "
         "refitted, onto each member, so the arms differ only in the training draw",
         "<code>notes/autoresearch.md</code> &sect;2 &mdash; the noise-aware gate, the symmetry rule, the amended "
         "protected-row clauses and the six-arm baseline"],
        f"An error bar for a scoreboard that had none, at the cost of six two-minute simulations and no training at all. "
        f"Gate 1 now requires a band upgrade that also clears the target row's seed sd; a movement smaller than its row's "
        f"floor counts neither for nor against an experiment, which is a rule about losses as much as gains; and the "
        f"frontier is scored against the mean of its own ensemble rather than the draw that happened to win. "
        f"<b>No verdict on this page is reversed and none should be</b> &mdash; every failure failed on a band its target "
        f"did not cross &mdash; but several of the numbers around those verdicts should never have been read as results."))
    out.append(story("timeout", "The manager and the player who said nothing at all",
        "correctness, failed its row", FIX,
        f"PR {pr_link(196)} &middot; one flag added to both punisher families &middot; two retrainings, two simulations "
        f"&middot; the first run judged under the noise floor",
        f"Sometimes a player simply did not answer. The game charged them 0, paid out on 0 and showed everyone 0, and the "
        f"human manager could see that no input had arrived: over the 50 games they punished a timed-out player "
        f"<b>{TO_HUM['timeout']:.2f}% of the time</b> and a player who <em>chose</em> to give nothing "
        f"<b>{TO_HUM['zero_all']:.1f}%</b>. The artificial punisher could not tell the two apart, so every timeout was served "
        f"to it as a defiant zero. The declared target was RCC, the reaction at the ceiling, through the punisher's response "
        f"to low contributions.",
        f"Step 0 checked the premise with an accounting identity rather than an argument: the stored common good of a group "
        f"in a round must equal 1.6 &times; its contributions minus its punishments. Under the recorded 0 that holds on "
        f"<b>{TO_ID['the recorded 0'][1]}</b> group-rounds; under the imputed 9 it fails on every one of the "
        f"{TO_GR - int(TO_ID['the imputed 9'][1].split('/')[0].replace(',', '').strip())} that contain a timeout. It also "
        f"<em>relocated</em> the fix: the imputed 9 never reaches the training data at all (all {TO_N} timed-out "
        f"agent-rounds record 0), it is injected by the environment at simulation time. So one half of this branch is a fix "
        f"to what the manager is <em>served</em>, and the other is a new input: a flag saying the input was missing.",
        [f"<b>The gate.</b> RCC {sd_of('m_tout', 'RCC')} &mdash; it moved further than its own noise floor, in the predicted "
         f"direction, and still did not cross the band, missing by {F(to_vd['rcc_after']) - 1:.4f}, which is itself "
         f"{(F(to_vd['rcc_after']) - 1) / SEED_SD['RCC']:.2f} seed sd. Gate 2 passes (mean "
         f"{VMEAN['m_tout']['before']:.4f} &rarr; {VMEAN['m_tout']['after']:.4f} against a ceiling of "
         f"{F(to_vd['gate2_ceiling']):.4f}, a move of {mean_sd('m_tout'):.2f} seed sd) and the protected row holds, every "
         f"band inside {max(to_pc[0]['change_in_se'].values()):.2f} pooled standard errors. The baseline it had to beat was "
         f"the six-arm <em>minimum</em> on RCC, {(SSMEAN['RCC'] - VEC['arm_shipped']['after']['RCC']) / SEED_SD['RCC']:.1f} "
         f"seed sd below the ensemble mean: a favourable draw to start from.",
         f"<b>The correctness result is separate from the gate, and it is real &mdash; on one family only.</b> For the linear "
         f"punisher the flag earns its place: cross-validated log loss {TO_CV['lin'][0]:.4f} &rarr; {TO_CV['lin'][1]:.4f}, "
         f"locked test {TO_CV['lin_test'][0]:.4f} &rarr; {TO_CV['lin_test'][1]:.4f}, and teacher-forced its weight on the "
         f"current contribution goes {tf('lin_ceiling_parent', 'OLS c_t'):.3f} &rarr; {tf('lin_timeout_new', 'OLS c_t'):.3f} "
         f"against the human {tf('human', 'OLS c_t'):.3f} &mdash; closing "
         f"{(tf('lin_timeout_new', 'OLS c_t') - tf('lin_ceiling_parent', 'OLS c_t')) / (tf('human', 'OLS c_t') - tf('lin_ceiling_parent', 'OLS c_t')):.0%} "
         f"of the gap, at a magnitude {pr_link(193)} had predicted in advance to within {TO_PRED}. For the graph punisher it "
         f"earns nothing and costs something: CV {TO_CV['gnn'][0]:.4f} &rarr; {TO_CV['gnn'][1]:.4f} at a paired t of {TO_T:.2f}, "
         f"a slope moving <em>away</em> from the human ({tf('gnn_ceiling_parent', 'OLS c_t'):.3f} &rarr; "
         f"{tf('gnn_timeout_new', 'OLS c_t'):.3f}) and a teacher-forced likelihood that worsens "
         f"({tf('gnn_ceiling_parent', 'nll'):.4f} &rarr; {tf('gnn_timeout_new', 'nll'):.4f}). The graph architecture already "
         f"reconstructs &ldquo;this player gave no input&rdquo; from its recurrent state; the explicit channel displaces "
         f"capacity rather than adding information.",
         f"<b>Other legible movements on the frontier</b>, each past its own floor: RCB {sd_of('m_tout', 'RCB')}, the largest "
         f"gain; CG {sd_of('m_tout', 'CG')}; PB {sd_of('m_tout', 'PB')}; RPB {sd_of('m_tout', 'RPB')}; and the one real cost, "
         f"<b>RSA {sd_of('m_tout', 'RSA')}, a band downgrade</b> that no protection clause covers and that carries forward as "
         f"a watch row. The watch row RCD moved {sd_of('m_tout', 'RCD')} &mdash; not distinguishable from a retrain, and "
         f"reported as such rather than claimed.",
         f"<b>The loudest number in the run is on a stack that was not gated.</b> On the main-sweep GNN-punisher reference the "
         f"punishment family blows out far beyond anything the floor explains: mean "
         f"{colmean(to_ba, 'ref_gnn_before'):.4f} &rarr; {colmean(to_ba, 'ref_gnn_after'):.4f} "
         f"({colsd(to_ba, 'ref_gnn_before', 'ref_gnn_after'):.2f} seed sd), rows &lt;= 1 "
         f"{colle1(to_ba, 'ref_gnn_before')} &rarr; {colle1(to_ba, 'ref_gnn_after')}, with RPB, PA, PB and RPA each moving "
         f"between 8 and 28 seed sd. The linear reference barely moves ({colmean(to_ba, 'ref_lin_before'):.4f} &rarr; "
         f"{colmean(to_ba, 'ref_lin_after'):.4f}, {colsd(to_ba, 'ref_lin_before', 'ref_lin_after'):.2f} sd &mdash; not "
         f"legible). Same feature, opposite outcomes, and the recommendation follows the measurement: keep it in the linear "
         f"punisher, do not carry it into the graph one."],
        ["<code>scripts/baselines/handcrafted_grid.py</code> &mdash; <code>contribution_valid</code> in the feature pool and "
         "the punishment legal set; <code>src/aimanager/generic/data.py</code> and "
         "<code>src/aimanager/manager/api_manager.py</code> &mdash; the same on the graph path",
         "<code>src/aimanager/simulation/linear_ah.py</code>, <code>src/aimanager/simulation/simulate.py</code> &mdash; the "
         "manager is served the recorded 0 on both simulation paths",
         "<code>configs/training/baselines/punishment/multinomial_timeout.yml</code>, "
         "<code>configs/training/artificial_humans/punishment/rnn_edge_50ep_doubled_timeout.yml</code> &mdash; the two retrains",
         "<code>plots/data_analysis/evaluation/punisher_timeout_feature/</code> &mdash; before_after.{csv,md} with every seed "
         "sd beside its delta, and the teacher-forced mechanism tables",
         "<code>notes/autoresearch_log/punisher-timeout-feature.md</code> &mdash; the log, the accounting identity and the "
         "artifact-provenance resolution"],
        f"A data-handling defect settled by arithmetic rather than by intent, a feature that belongs in one punisher family "
        f"and not the other, and RCC left within one noise floor of its band &mdash; so the next change to that row decides it. "
        f"It also left two things open on purpose: the same substitution was still reaching the contributor and the switch "
        f"model, which is the next card, and RSA is now a watch row nobody has claimed."))
    out.append(story("simtimeout", "The nine that was never there: what the players were being shown",
        "correctness, failed its row", SLOT["contribution"],
        f"PR {pr_link(197)} &middot; nothing retrained &mdash; every slot loads the same file on disk as its parent, "
        f"under the same seed &middot; four simulations and a probe",
        f"The previous card fixed what the <em>manager</em> was served and deliberately left the other two slots alone, so "
        f"that its own rows stayed attributable. The defect was still there: during simulation the environment overwrote a "
        f"timed-out player's contribution with the default of 9 before the state was passed on, although the game charged 0, "
        f"paid out on 0, showed everyone 0 and stores 0 in the training tensors. The parent's note said this reached the "
        f"switch model as a 9 that round and the contribution model as a 9 the round before. This branch measured which half "
        f"of that is actually true, fixed it, and re-ran.",
        f"A probe wrapped every model's <code>predict</code> call and recorded the state keys each encoder actually consumes "
        f"and the value served at each timed-out cell, on the real machinery. On {st_pb['timeouts']} timed-out cells in "
        f"{st_pb['agent_rounds']:,} agent-rounds ({st_pb['timeout_rate']:.2%}, against {TO_N / TO_AR:.1%} in the human data), "
        f"the contribution model's <code>prev_contribution</code> read 9.0 on all "
        f"{PREV_B['9.0'] - PREV_A['9.0']} of the cells that are not round-0 lag defaults (those "
        f"{PREV_A['9.0']} keep the 9, exactly as the training tensors' own shift does), and now reads 0.0 on all "
        f"{PREV_A['0.0']}. The switch model in this stack "
        f"reads {', '.join(ST_MODEL['switch']['reads'])} &mdash; <b>never a contribution key at all</b>, so the defect never "
        f"reached it. Half the parent's prediction was right and half was wrong, and the probe is what settled which.",
        [f"<b>This is direct progress on the one defect the campaign has left.</b> With the shared-noise machinery switched "
         f"off &mdash; which is how &sect;2 says a contributor change must be judged &mdash; the variety of situations the "
         f"simulation reaches rises from {sd_var('parent_off'):.2f} to {sd_var('cand_off'):.2f} against the human "
         f"{F(st_dec['human']['var_cond_mean']):.2f}, while the randomness inside each round was already about right "
         f"({F(st_dec[ST_ARM['cand_off']]['var_resid']):.2f} against the human "
         f"{F(st_dec['human']['var_resid']):.2f}) and stays so. <b>No randomness was added to get it</b>: the model is "
         f"simply conditioning on a value the game actually used, on the rounds where the game used it.",
         f"<b>The gates fail and the cost is legible.</b> No declared target upgraded a band: CG {sd_of('n_simto', 'CG')}, "
         f"SC {sd_of('n_simto', 'SC')}, <b>SB {sd_of('n_simto', 'SB')}, a band downgrade</b>, and SA "
         f"{sd_of('n_simto', 'SA')} &mdash; not distinguishable from a retrain. Gate 2 passes "
         f"({VMEAN['n_simto']['before']:.4f} &rarr; {VMEAN['n_simto']['after']:.4f} against a ceiling of "
         f"{F(st_vd['gate2_ceiling']):.4f}). RCC goes {sd_of('n_simto', 'RCC')} &mdash; which means the previous card's "
         f"{VEC['m_tout']['after']['RCC']:.4f} was measured with a wrong lag, and anyone continuing that line must "
         f"re-baseline first. PD improves legibly, {sd_of('n_simto', 'PD')}. In all, {22 - n_legible('n_simto')} of the 22 "
         f"rows move by less than their own seed sd.",
         f"<b>The protected row fires, on the band where an unchanged model flips sign anyway.</b> RCE's 10-14 slope goes "
         f"{lead(st_sl[0]['before'][3]):+.3f} to {lead(st_sl[0]['after'][3]):+.3f} (human {HUMAN_SLOPES[2]:+.3f}), a change of "
         f"{st_pc[0]['change_in_se']['10-14']:.2f} pooled standard errors and "
         f"{st_pc[0]['change_in_seed_sd']['10-14']:.2f} seed sd, so the amended magnitude clause fires. Context rather than "
         f"excuse: {pr_link(195)} measured that band running {sgn(F(ss_agg['rce_slope_10-14']['min']))} to "
         f"{sgn(F(ss_agg['rce_slope_10-14']['max']))} across six retrains of an unchanged model, which is why the sign clause "
         f"is retired there; the slope crossed zero rather than reversing. The row's score itself, "
         f"{sd_of('n_simto', 'RCE')}, is not distinguishable from a retrain and its band holds.",
         f"<b>Why the cost is all in the group-spread rows, and why that is informative.</b> With the shared-noise machinery "
         f"off the fix leaves the group-spread ratio flat ({sd_cg('parent_off'):.4f} &rarr; {sd_cg('cand_off'):.4f} against "
         f"the human {F(st_dec['human']['cg_ratio']):.4f}); with it on, the ratio falls further below the human "
         f"({sd_cg('parent_on'):.4f} &rarr; {sd_cg('cand_on'):.4f}). The mis-served nines had been a spurious source of "
         f"between-group dispersion inside the closed loop, sitting on top of a copula calibrated to supply exactly that "
         f"dispersion. The stack whose contributor carries no such machinery improves instead: the linear-punisher reference "
         f"goes {colmean(st_ba, 'ref_lin_before'):.4f} &rarr; {colmean(st_ba, 'ref_lin_after'):.4f} "
         f"({colsd(st_ba, 'ref_lin_before', 'ref_lin_after'):.2f} seed sd, legible) with rows &lt;= 1 "
         f"{colle1(st_ba, 'ref_lin_before')} &rarr; {colle1(st_ba, 'ref_lin_after')}. That asymmetry is the whole result, and "
         f"the next card tests the reading of it.",
         f"<b>This comparison has no noise floor at all, and that is unusual enough to say plainly.</b> Nothing was retrained: "
         f"both runs load the same artifacts under seed 42, so the difference contains none of the "
         f"{SD_ROW:.3f}-per-row training spread the floor describes. There is no better draw to be had by re-running, which "
         f"means the cost above is as real as the gain.",
         f"<b>An open escalation, and it is the reason the environment's substitution was kept rather than removed.</b> "
         f"{ESC_HTML}. Recording the 0 in the simulation's own output would push "
         f"about {st_pb['timeout_rate']:.1%} of <em>scored</em> rows to a hard zero against human rows that are not there at "
         f"all, so the substitution was intercepted at serving and left in the recording. The suite is frozen surface and no "
         f"agent may change it: <b>a maintainer has to decide</b>, and the fix is a validity column in "
         f"<code>per_round.parquet</code> plus one <code>where()</code> in <code>load_sim</code>."],
        ["<code>src/aimanager/manager/environment.py</code> &mdash; the new <code>served_state()</code>: contribution and "
         "prev_contribution read the recorded 0 wherever the validity flag is false, round 0 excluded; the env's own "
         "dynamics, accounting and recorded output are untouched",
         "<code>src/aimanager/simulation/linear_ah.py</code> &mdash; the validity mask threaded through the env-driven path "
         "so a linear bundle cannot hit the same defect",
         "<code>scripts/data_analysis/sim_timeout_serving_probe.py</code> &mdash; the probe that reports what each encoder "
         "reads and what it is served; <code>probe_before.json</code> / <code>probe_after.json</code>",
         "<code>plots/data_analysis/evaluation/sim_timeout_imputation/</code> &mdash; before_after.{csv,md} and the "
         "state-spread decomposition with the copula on and off",
         "<code>notes/autoresearch_log/sim-timeout-imputation.md</code> &mdash; the log, including why the recorded output "
         "keeps the 9 and the escalation that follows from it"],
        f"The last place in the pipeline where a model was shown a value the game never used, found and closed &mdash; and "
        f"{sd_var('cand_off') - sd_var('parent_off'):+.2f} of state variety bought without adding a single unit of noise, "
        f"which is the first movement on the late-divergence defect that did not come out of the copula. The verdict is still "
        f"a failure: no declared target upgraded, SB downgraded, and the protected row fired. Both facts are true and neither "
        f"cancels the other."))
    out.append(story("recal", "A refutation, measured four times over: the copula never saw the defect",
        "refuted", SLOT["contribution"],
        f"PR {pr_link(198)} &middot; nothing retrained; one scalar restamped &middot; the one parameter &sect;2 allows an "
        f"experiment to unfreeze, unfrozen once and on the record",
        f"The previous card's reading of its own cost was that the herding copula had been <em>calibrated in the presence of "
        f"the defect</em> &mdash; that its strength had quietly absorbed the dispersion the mis-served nines were supplying, "
        f"so removing them left a debt. That is a testable claim and it points at a specific number: the shared-noise "
        f"strength rho, shipped at {cr_par['rho']:.4f}. If the reading is right, refitting rho against a model that no longer "
        f"sees those nines should give a larger value, by more than the estimator's own spread, and stamping it should recover "
        f"the group-spread row.",
        f"The parameter was unfrozen exactly once, with the existing estimator, the same base trunk by sha256, the same flags "
        f"and the same seed as the job that produced the shipped value, then stamped onto a copy of the contributor and run "
        f"through the frontier stack and the noise-off arm. Nothing else moved: the persistence stayed at its frozen "
        f"{cr_par['phi_final']:.1f} and the switch-every at {cr_par['copula_switch_every']}.",
        [f"<b>The refit returns the shipped value to the last digit</b>: rho = {cr_par['rho']:.17f}, a difference of exactly "
         f"zero, with bootstrap SE {cr_par['rho_se']:.4f} and 95% interval "
         f"[{cr_par['rho_ci'][0]:.4f}, {cr_par['rho_ci'][1]:.4f}]. Four independent equalities follow and each was measured "
         f"rather than argued: all {CR_FIELDS} estimate and provenance fields of the parameter file identical (only the date "
         f"and the git head differ); the stamped artifact identical to the shipped one in all 64 hex digits of its sha256 "
         f"(<code>{CR_SHA[:12]}&hellip;</code>); the simulation's per-round output byte-identical; and all 22 rows, the mean "
         f"and the rows &lt;= 1 count exactly unchanged &mdash; every delta 0.0000, i.e. 0.00 of its seed sd.",
         f"<b>Why it could not have been otherwise.</b> rho is fitted by pairwise-likelihood maximum likelihood against "
         f"<em>human</em> histories, teacher-forced, from a training tensor in which a timed-out player's contribution and the "
         f"lag that follows it are the recorded 0. The imputed 9 lived only on the simulation serving path, which the "
         f"estimator neither imports nor executes. The refit was a null by construction and it measured as one &mdash; but the "
         f"chain was run end to end anyway, so that someone who does not trust the argument can check the equalities.",
         f"<b>The consequence is a correction to the previous card, not to this one.</b> The mis-served nines were not a bias "
         f"the calibration had internalised; they were a spurious source of between-group dispersion inside the closed loop, "
         f"sitting on top of a correctly fitted copula. Removing them did not create calibration debt &mdash; <b>it exposed a "
         f"group-spread deficit the defect had been covering</b>. That also explains the asymmetry the parent found most "
         f"informative: the reference stack has no copula compounding the spurious dispersion, so the fix simply made it more "
         f"correct.",
         f"<b>And the route is closed, not merely unhelpful.</b> The only rho the estimator supports is the one it measures; "
         f"choosing a larger one because the group-spread row would score better is tuning at a metric's definition rather "
         f"than at behaviour, which &sect;5 forbids, and the calibration script's own pre-flight says in as many words that "
         f"rho is never tuned to that ratio. <b>The copula's strength is closed as a route to CG. Its shape is not</b>: with "
         f"the machinery off the stack reaches {sd_var('cand_off'):.2f} of the human "
         f"{F(st_dec['human']['var_cond_mean']):.2f} worth of state variety while its residual variance is already right, so "
         f"the gap is in what the model conditions on. {pr_link(187)} separately measured the human dependence to be a "
         f"round-local shock with a one-round echo rather than the episode-long latent that ships.",
         f"<b>Two controls fell out of the run and both are worth more than the experiment.</b> The simulation is "
         f"bit-reproducible across isolated remote directories, GPU nodes and sessions &mdash; both arms reproduced the "
         f"parent's recorded output byte for byte &mdash; and so is the evaluation: re-running the scorer over the parent's "
         f"own simulation rewrote every table and all {CR_FIGS} figures and left the working tree clean. <b>A before/after on "
         f"unchanged artifacts therefore has a noise floor of exactly zero</b>, which is the opposite end of the same "
         f"question {pr_link(195)} answered for retraining.",
         f"<b>One trap, flagged because it nearly became a second undeclared change.</b> The estimator writes its lag-1 ratio "
         f"as <code>phi</code> and never writes <code>phi_final</code>; the stamping script falls back to the bare "
         f"<code>phi</code> when <code>phi_final</code> is absent. A refit of rho alone would silently have stamped "
         f"<code>copula_phi = {cr_par['phi']:.4f}</code> over the frozen {cr_par['phi_final']:.1f} &mdash; a second change to "
         f"a separately frozen parameter. A small script now carries the ruling across and refuses to guess."],
        ["<code>scripts/artificial_humans/contribution_copula_rho.py</code> &mdash; the estimator, unchanged, re-run with the "
         "same flags and seed; <code>freeze_phi_in_params.py</code> &mdash; the guard against the fallback above",
         "<code>artifacts/artificial_humans/..._herding_copula_recal/calibration/copula_params.json</code> &mdash; the "
         "refitted parameter file quoted here, field for field the shipped one",
         "<code>plots/data_analysis/evaluation/contribution_copula_recalibrated/</code> &mdash; the 22 rows with every delta "
         "at 0.0000 beside its seed sd, and the state-spread arms",
         "<code>notes/autoresearch_log/contribution-copula-recalibrated.md</code> &mdash; the log, the four equalities and "
         "the two reproducibility controls"],
        f"A hypothesis refuted at the only point where it was testable, for {CR_MIN} minutes of estimator time; a correction to "
        f"how the previous experiment's cost should be read; one route to the group-spread row closed for good and the "
        f"remaining one named; and two bit-reproducibility controls the campaign did not have. A refutation is worth what a "
        f"confirmation would have been, and the serving fix stands on its own measurement either way."))
    out.append(story("successor", "What this leaves for a successor",
        "successor", "#898781",
        f"after the four-step programme and the four runs that followed it &middot; one large defect, three isolated faults "
        f"in the player models and one escalation a maintainer has to take "
        f"&middot; <code>doc/plans/post-rebaseline-program.md</code> and one log per experiment",
        f"Eleven pull requests since the re-baseline, and the picture is narrower, harder, and now measured. "
        f"<b>The target is still the late-divergence failure</b>: real groups keep pulling apart as a game runs and the "
        f"models stop. With the shared-noise machinery switched off &mdash; which is how a contributor change must be judged "
        f"&mdash; the variety of situations the simulation reaches is {sd_var('cand_off'):.2f} against the human "
        f"{F(st_dec['human']['var_cond_mean']):.2f}, while the randomness inside each round is already correct "
        f"({F(st_dec[ST_ARM['cand_off']]['var_resid']):.2f} against {F(st_dec['human']['var_resid']):.2f}). "
        f"Two routes to it are now closed by measurement rather than by argument: the output design is not the lever "
        f"({pr_link(191)}) and neither is the shared-noise strength ({pr_link(198)}, which refits to the shipped value "
        f"exactly and therefore cannot be tuned). One route has just opened: {pr_link(197)} moved that number for the first "
        f"time without adding noise, from {sd_var('parent_off'):.2f} to {sd_var('cand_off'):.2f}, by fixing what the model "
        f"was being shown. What remains is the copula's <em>shape</em> &mdash; {pr_link(187)} measured the human dependence "
        f"to be a round-local shock with a one-round echo, not the episode-long latent that ships &mdash; and anything that "
        f"carries a group's state across rounds and survives the models playing against each other.",
        f"Nothing is declared yet. Below: the three faults that are cleanly isolated in the player models with a baseline "
        f"ready, the one open escalation, the standing manager-side defect, and what is deliberately out of scope. "
        f"Every declaration from here has to name its target row's seed sd and say whether the movement it expects can be "
        f"seen at all &mdash; ten of the 22 rows cannot serve as a gate-1 target on a single run, and the frontier's "
        f"baseline is now the mean of its own six-arm ensemble rather than the draw that happened to win ({pr_link(195)}).",
        [f"<b>Fault 1, the players under-react to a heavy punishment at the ceiling.</b> A punished full contributor gives up "
         f"{abs(F(ce_rcc['frontier after'][2])):.2f} points the next round where a real person gives up "
         f"{abs(F(ce_rcc['human'][2])):.2f}, about {under:.1f} times too little. The manager's side of RCC is finished "
         f"({pr_link(192)}), the population is the right size ({ce_rcc['frontier after'][6]} of full contributors punished "
         f"against the human {ce_rcc['human'][6]}), and the decomposition table is the baseline. RCC is the only row that "
         f"measures it and it must be declared against the contributor. Two cautions: RCC is <em>gateable</em> "
         f"(seed sd {SEED_SD['RCC']:.3f}) but its shipped baseline was the six-arm <em>minimum</em>, so the honest baseline "
         f"is {SSMEAN['RCC']:.4f} rather than {VEC['arm_shipped']['after']['RCC']:.4f}; and the row's current value on the "
         f"corrected serving path is {VEC['n_simto']['after']['RCC']:.4f}, not the {VEC['m_tout']['after']['RCC']:.4f} the "
         f"timeout branch recorded with a wrong lag. Re-baseline before declaring.",
         f"<b>Fault 2, the wrong people leave after being punished.</b> Two separate experiments have now damaged this row "
         f"and neither meant to: {pr_link(190)} got the number of switches and the group sizes right and regressed RSA "
         f"hardest ({sk('RSA', 'before'):.3f} &rarr; {sk('RSA', 'after'):.3f}, {sv_sd('#190', 'RSA'):.2f} seed sd &mdash; the "
         f"largest legible movement in that run), and {pr_link(196)} downgraded its band as a side effect "
         f"({sd_of('m_tout', 'RSA')}). No protection clause covers it. It is a concrete mismatch, diagnosable on one "
         f"simulation, and it is the cheapest open item on this page.",
         f"<b>Fault 3, the middle contribution band has the wrong sign and never learned it.</b> RCE's 10-14 slope is wrong "
         f"teacher-forced and held out alike ({pr_link(183)}: pooled held-out {sgn(HO_SLOPES['pooled_held_out'][2])}, "
         f"in-sample {sgn(HO_SLOPES['pooled_in_sample'][2])}, human {sgn(HUMAN_SLOPES[2])}), so no closed-loop fix will "
         f"supply it. What {pr_link(195)} adds is that <b>this band cannot be used as evidence either way on a single run</b>: "
         f"across six retrains of an unchanged model it runs {sgn(F(ss_agg['rce_slope_10-14']['min']))} to "
         f"{sgn(F(ss_agg['rce_slope_10-14']['max']))} and its seed spread exceeds the within-run standard error. The sign "
         f"clause is retired there for exactly that reason. Fixing it is a training-side question, and it needs more than "
         f"one seed on both sides to be answered.",
         f"<b>The open escalation, and it is not an agent's to close.</b> The evaluation suite does not measure the two sides "
         f"alike: {ESC_HTML}. About {st_pb['timeout_rate']:.1%} of simulated agent-rounds therefore enter every contribution "
         f"and response row at a value the game never used, with no human counterpart to compare against &mdash; and it is "
         f"the reason {pr_link(197)} intercepted the substitution at serving instead of removing it. This is now the last "
         f"place in the pipeline where the imputed 9 is doing work. The fix is small &mdash; a "
         f"<code>contribution_valid</code> column in <code>per_round.parquet</code> and one <code>where()</code> in "
         f"<code>load_sim</code> &mdash; and <code>evaluation_suite/</code> is frozen surface, so <b>a maintainer has to make "
         f"the call</b>. Until then every C and R row on this page carries that asymmetry.",
         f"<b>Still standing on the manager's side: its response to contribution is about half the human strength</b>, "
         f"{cm('frontier after', 'OLS c_t'):.3f} per point against {cm('human', 'OLS c_t'):.3f}. Nothing has moved it far: "
         f"the timing fix did not, the ceiling flag changes it by at most {CE_SLOPE_MAX:.3f} in any condition, and the "
         f"timeout flag closed "
         f"{(tf('lin_timeout_new', 'OLS c_t') - tf('lin_ceiling_parent', 'OLS c_t')) / (tf('human', 'OLS c_t') - tf('lin_ceiling_parent', 'OLS c_t')):.0%} "
         f"of it teacher-forced but only in the linear family. It wants a bent response rather than another flag.",
         f"<b>A second maintainer decision, cheap and already computed.</b> {pr_link(195)}'s six arms are committed, so the "
         f"frontier can be re-baselined on the six-arm mean at no cost: 22-row mean {VMEAN_SS:.4f} rather than "
         f"{VMEAN['arm_shipped']['after']:.4f}, rows &lt;= 1 {int(F(ss_agg['rows_le1']['mean']))} rather than "
         f"{VLE1['f_ceil']['after']}, RCE {SSMEAN['RCE']:.4f} rather than {VEC['arm_shipped']['after']['RCE']:.4f}. "
         f"It makes the frontier look worse and the comparisons honest, and it moves the baselines of the open branches. "
         f"The cheap check it suggests is whether the <em>next</em> accepted model also lands at the extreme of its own "
         f"reseed ensemble; if it does, the selection is structural rather than accidental.",
         f"<b>Deliberately out of scope.</b> The three group facts from {pr_link(187)} are worth folding into the next "
         f"contributor retrain but account for about a seventh of a small quantity. The rollout-training family is the wrong "
         f"tool for this defect, for a reason that is now understood. Whether the <em>manager's</em> shared mood is an "
         f"episode-level or a round-level latent was answered for the contributor's copula ({pr_link(187)}) and is still "
         f"open for the severity copula, which was never measured. A refit of the contribution copula's persistence is "
         f"unasked and its machinery now exists ({pr_link(198)}). The manager's room to act must be bounded or audited "
         f"before a learning manager explores it: real managers rarely punished above 10 points or punished high "
         f"contributors, about 300 rows of evidence in total. And the 32-stack sweep matrix still has not been rerun under "
         f"the fixed punisher, so the ledger's deficit profiles are all pre-fix."],
        ["<code>doc/plans/post-rebaseline-program.md</code> &mdash; the four steps as declared and as they came out",
         "<code>notes/autoresearch_log/</code> &mdash; one log per experiment, each with its own successor section and caveats",
         "<code>notes/autoresearch.md</code> &sect;2 &mdash; the noise-aware gate, the symmetry rule, the amended "
         "protected-row clauses and the six-arm frontier baseline; &sect;8 &mdash; the frozen surface the escalation sits on",
         "<code>plots/data_analysis/evaluation/seed_spread_noise_floor/per_row.csv</code> &mdash; the seed sd every future "
         "declaration has to quote beside its target row",
         "Raven clean-up as each pull request closes: one isolated folder per experiment under <code>~/repros/ai-runs/</code>"],
        f"A map with one large defect, three isolated faults in the player models, one manager-side defect and one "
        f"escalation &mdash; each with a baseline a successor can declare against, and each now quoted beside the noise it "
        f"has to beat. The protocol judges punishment-response work on the mechanism rather than on a composition row, "
        f"noise-model work as its own experiment, and no movement at all below its own row's measurement floor. "
        f"Five of the eleven later runs failed their gate and two falsified hypotheses their own authors had proposed; "
        f"none of that is blocked work, it is the map."))
    return "\n\n".join(out)


# ---------------------------------------------------------------- page
def page():
    mnav, cards = ba_cards()
    rows_json = json.dumps(lb_rows())
    lb_script = SCRIPTS[1]
    assert "const ROWS = [" in lb_script
    lb_script = re.sub(r"const ROWS = \[.*?\];", lambda m: f"const ROWS = {rows_json};", lb_script, count=1, flags=re.S)
    best = min(list(ORDER) + STEPKEYS, key=lambda c: VMEAN[c]["after"])
    tree_legend = ("<p class=\"legend\">\n"
                   f'<span style="color:{SPINE["gnn"]}">&#9473;</span> gnn spine &nbsp; <span style="color:{SPINE["gmlp"]}">&#9473;</span> gaussian-MLP spine &nbsp; '
                   f'<span style="color:{FIX}">&#9473;</span> punisher fix (before &rarr; after) &nbsp; &#9675; before (lagged punisher) &nbsp; &#9679; after (current-contribution punisher)\n'
                   f'&nbsp; <span style="color:{FCOL["step"]}">&#9473;&#9632;</span> a follow-up run, scored against the filled node it hangs off '
                   f'&nbsp; <span style="color:{FCOL["abl"]}">&#9670;</span> an ablation of the frontier, not a candidate '
                   f'&nbsp; <span style="color:#e87ba4">&#9650;</span> a diagnostic with no 22-row score, parked on the strip under the runs it informs '
                   f'&nbsp; <span style="color:{ALERT}">&#9711;</span> a red ring marks an experiment that missed its declared gate\n'
                   f'&nbsp; <span style="color:{NOISE}">&#9478;</span> the six-tick rail is PR #195: the same contributor architecture '
                   f'trained on six seeds and run through this stack with everything else identical, so its height is the spread of '
                   f'one stack\'s mean when nothing changes but the training draw. It is a measurement, not a step: nothing moved '
                   f'along it. The thick tick is the shipped artifact, which is the lowest of the six, and the dot is the six-arm mean '
                   f'that is now the frontier\'s baseline.\n'
                   f'&nbsp; <span style="color:{NOISE}">&#9711;</span> a gold dotted ring marks a node whose whole-stack score moved '
                   f'by less than that spread &mdash; <b>its mean is not distinguishable from a retrain</b> '
                   f'({", ".join(SHORT[k].replace("main &middot; ", "") for k in list(ORDER) + STEPKEYS if k in INSIDE)}). '
                   f'{FLOOR_NOTE}.\n'
                   "&nbsp; (solid = lineage spine from the main-sweep stack, dashed = the same stack under the GNN punisher, dotted gaussian-MLP line = "
                   "the k-one-hot switch component borrowed from that spine, dotted step = best mean on record so far) "
                   "&mdash; hover a node for its mean, rows &lt;= 1, RCE before &rarr; after and how far it moved in seed sd; click it for its story. "
                   "Five of the seven follow-up experiments failed their gate and are shown, not hidden.\n</p>")
    return f"""<meta charset="utf-8">
<title>Rebaseline Atlas</title>
{STYLES[0]}
{STYLES[1]}
<div class="wrap">
<h1>Rebaseline Atlas</h1>
<p class="sub">The companion to the Autoresearch Atlas: the frontier stacks rerun after the simulated
manager was fixed to punish the current round's contribution (PR #184), scored on 22 rows including
the new, protected RCE row -- six runs, before and after, on two spines -- and then the eleven pull
requests that followed it: three ablations of the shared-noise machinery, a falsified test of the
output design, a ceiling fix and a switch-component port that both missed their gate, the protocol
change that froze the noise settings and repaired its own safety rule, and then four runs that
changed what the page can claim. One of them measured how far the whole scoreboard moves when
nothing changes but a training seed -- a typical row by sd {SD_ROW:.3f}, the count of rows at the
human ceiling between {int(F(ss_agg['rows_le1']['min']))} and {int(F(ss_agg['rows_le1']['max']))} --
so every score here now carries its own measurement noise, and several
earlier results turn out to sit inside it. Hover any node for its numbers and how far it moved in
units of that noise; click it for the plain-language story.</p>
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
  <p class="legend">Focus a spine, or select both (nothing is hidden, including the five
  experiments that failed their gate):</p>
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
<p class="legend"><b>Every row now carries its own measurement noise.</b> The
<span style="background:{NOISE};opacity:0.35">&nbsp;&nbsp;&nbsp;</span> gold band on each chart is
&plusmn;1 seed standard deviation around the six-arm mean of PR #195 &mdash; how far that row travels
when six copies of one contributor architecture, differing only in their training seed, are run
through the same stack with everything else identical. The gold figure at the top right is that
standard deviation; {len(UNGATE)} of the 22 rows are marked <em>ungateable</em>, meaning a band
boundary falls inside one deviation and the six arms genuinely land in two bands, so a single run
cannot decide them ({", ".join(UNGATE)} &mdash; RCE, the protected row, among them).
<b>A segment drawn dashed moved by less than its row's deviation and is not distinguishable from a
retrain.</b> Columns: the six reruns grouped by spine
(<span style="color:{SPINE['gnn']}">&#9473; gnn</span>: main-lin, main-gnn, #179, #181;
<span style="color:{SPINE['gmlp']}">&#9473; gaussian-MLP</span>: #174, #177), then, past the dotted
divider, the seven <span style="color:{FCOL['step']}">&#9632; follow-up runs</span> (ceil = the
ceiling flag on the frontier, kexo = the ported switch head, c&middot;lin and c&middot;gnn = the
ceiling flag on the two main-sweep references, tout = the punisher's timed-out-player flag,
sim0 = the recorded 0 served to the contributor, rho = the copula refit, whose every delta is
exactly zero). In each column the hollow marker is the score before that run's own change and the
filled one after: for the first six that is before and after the punisher fix, for the last seven the
run's own baseline and the run (guides at the 1 / 2 / 5 band edges, log scale; titles colored by slot:
<span style="color:{SLOT['contribution']}">contribution</span> <span style="color:{SLOT['switch']}">switch</span> <span style="color:{SLOT['punisher']}">punisher</span>).
The ablation arms and the six seed arms have no column here; their profiles are the last two panels of
the Score breakdown. Hover a pair for the numbers, the band change and the movement in seed sd; hover
the band for the row's own spread.</p>
<div class="grid21">{"".join(small_chart(r) for r in ROWS)}</div>
</section>
<section class="layer" id="breakdown">
<div class="treehead">
<p class="legend">All 22 rows, colored by the slot each row measures. The first two panels are the
two spines' reruns: solid = after the punisher fix, dashed = before. The third is the four follow-up
runs of PRs #192 and #190, the fourth the timeout chain of PRs #196, #197 and #198, and the fifth the
three ablations of the frontier's shared-noise machinery; in all three, dashed = that run's own
baseline and solid = the run. <b>The sixth is not a lineage at all</b>: it is PR #195's six arms, one
contributor architecture trained on six seeds, each against the six-arm mean &mdash; the fan it draws
is the noise every other panel has to beat. The bold line is the 22-row mean (always shown) &mdash;
hover a line for its values, or focus one or more slots:</p>
  <div class="focus">
    <button data-key="contribution" data-color="{SLOT['contribution']}">contribution</button>    <button data-key="switch" data-color="{SLOT['switch']}">switch</button>    <button data-key="punisher" data-color="{SLOT['punisher']}">punisher</button>
  </div>
</div>
<div class="two" id="breakbox">{breakdown_svg([c for c in ORDER if CASE[c][1] == "gnn"], 'gnn spine: ' + " &#8594; ".join(esc(CASE[c][5]) for c in ORDER if CASE[c][1] == "gnn") + ', before (dashed) and after (solid)')}
{breakdown_svg([c for c in ORDER if CASE[c][1] == "gmlp"], 'gaussian-MLP spine: ' + " &#8594; ".join(esc(CASE[c][5]) for c in ORDER if CASE[c][1] == "gmlp") + ', before (dashed) and after (solid)')}
{breakdown_svg(["g_clin", "h_cgnn", "f_ceil", "i_kexo"], 'follow-up runs (#192, #190): each against its own post-fix baseline (dashed)', ("baseline", "after"))}
{breakdown_svg(["m_tout", "n_simto", "o_recal"], 'the timeout chain (#196, #197, #198): each against its own baseline (dashed)', ("baseline", "after"))}
{breakdown_svg(ABLKEYS, 'ablations of the frontier (#186, #188): the frontier itself is the dashed line', ("frontier", "arm"))}
{breakdown_svg(AKEYS, 'the measurement floor (#195): one model, six training seeds', ("six-arm mean", "arm"))}
</div>
</section>
<section class="layer" id="beforeafter">
<p class="legend">The evaluation suite's own figure for each score row, in five states, each one
change further along the frontier's lineage: before the punisher fix (the source sim, lagged
punisher), after it (the _curpun rerun), after the ceiling flag of PR #192, after the
timed-out-player flag of PR #196, and after PR #197 stopped serving the contributor an imputed 9.
Three stacks, one column each: #181 stimulus skip, #174 k-one-hot Gaussian-MLP, and the main-sweep
stack with the GNN punisher; rows with two figures show both. Pick a row. (SA has no figure; its
score is a single rate. The #174 source sim carries no visuals and none of the later runs was made on
that stack; no before figure exists for RCE, which was added to the suite after those sims were
plotted; the ceiling flag ran only on the frontier and the main-sweep reference. <b>PR #198 has no
column</b>: its simulation reproduced PR #197's recorded output byte for byte, so its figures are the
same files. Every figure here is downscaled to 800 px on the long edge to keep the page under 14 MB;
they are displayed at about a third of the page width, so nothing visible is lost.)</p>
{mnav}
{cards}
</section>
<section class="layer" id="machinery">

<p class="legend">
<span style="color:{FIX}">&#9632;</span> the fixed piece (punisher input) &nbsp;
<span style="color:{SLOT['contribution']}">&#9632;</span> correlated-sampling unit: recalibrated, then measured and frozen &nbsp;
<span style="color:#4a3aa7">&#9632;</span> protected row &nbsp;
<span style="color:#e87ba4">&#9632;</span> diagnostic &nbsp;
<span style="color:#6b6a66">&#9632;</span> ledger &nbsp;
&#9633; stock part &nbsp; &#11044;<small>#PR</small> installed by &nbsp;
<span style="color:{FCOL['step']}">&#11040;</span> dashed outline = a part one of the eleven later PRs
touched; the tooltip says whether the change landed, was frozen, or was measured and rejected &nbsp;
<span style="color:{ALERT}">&#11040;</span> red dashed outline = an open escalation a maintainer has
to decide, because the part sits in a directory no agent may modify
&mdash; hover a tinted or outlined part for its story, <b>click a pill</b> for
the plain-language page.
</p>
<h2>One round of the simulation loop, with the punisher's input fixed (lowest 22-row mean on record: {SHORT[best]}, {f3(VMEAN[best]['after'])}, rows &lt;= 1: {VLE1[best]['after']}/22 &mdash; from an experiment that failed its gate; the ledger's frontier is still {SHORT['b_skip']}, now baselined at the six-arm mean {f3(VMEAN_SS)} rather than the single run's {f3(MEAN['b_skip']['after'])})</h2>
<figure>
{machinery_svg()}
</figure>

</section>
<section class="layer" id="lb">
<p class="legend">Sixteen records, each scored against its own before state: the six reruns against
the same stack under the lagged punisher, the seven follow-up runs against their own baseline, and
the three ablation arms against the frontier they ablate. The note column carries the verdict as the
log recorded it &mdash; five of the seven declared experiments failed their gate, and the ablations
were never candidates. <b>It also carries the noise.</b> Every record says how far its mean moved in
units of PR #195's seed standard deviation, how far its rows-at-the-ceiling count moved in units of
its own, and how many of the 22 rows moved by more than their own; a
<span class="pill" style="background:{NOISE}">&Delta; mean inside the floor</span> tag means the
whole-stack movement is smaller than what a retrain produces on its own, so <b>that record's place in
the ranking is a draw and not a result</b>. The band-upgrade tooltip gives each upgraded and
downgraded row with its own movement in seed sd, marking those inside the floor. Pick the ranking
criterion:</p>
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
better). Hover a band-upgrade count for the rows and their movements in seed sd. The PR column is the
stack's PR; the two main-sweep reruns carry the re-baseline PR #184. Ties break on &Delta; mean.
A good &Delta; is not a pass: a gate is a band change on a row declared in advance, which is why
PR #192 sits high here and still failed, and why the lowest 22-row mean on record (PR #196) is a
failure too. <b>And a &Delta; is not automatically real.</b> The mean's own seed standard deviation is
{SD_MEAN:.4f} and the rows-at-the-ceiling count's is {SD_LE1:.2f}, so {len([k for k in list(ORDER) + FKEYS if k in INSIDE])}
of these sixteen records have a mean movement smaller than a retrain of an unchanged model produces &mdash;
they are tagged, and their rank should be read as a tie rather than an ordering. The floor was measured on
the frontier stack by retraining its contributor six ways; on the other stacks it is a lower bound rather
than a full error bar, because their punisher or switch slot differs too. Two records need a further
caveat of their own: PR #198's every delta is exactly 0.0000 because its artifact is byte-identical to its
baseline's, so its floor really is zero rather than merely small; and the six reruns' own baselines are the
pre-fix scores of a different punisher, which this floor does not cover at all.</p>
</section>
<section class="layer" id="stories">
<p class="legend">The plain-language story of the re-baseline and of the eleven pull requests that
followed it, in fifteen cards &mdash; also reachable by clicking tree nodes, score markers and
machinery pills. Cards six to ten are the post-re-baseline programme: two of its four steps failed
their gate, one falsified the hypothesis its own author had proposed, and the fourth had to repair
the safety rule it had just written. The four after them are what changed the page's terms: a
measurement of the scoreboard's own noise, which several earlier results turn out to sit inside; a
manager that can now tell a timeout from a chosen zero; a simulation that stopped showing the players
a number the game never used; and a recalibration that refuted the reading of the run before it, four
times over and to the last digit. The last card is what all of it leaves open.</p>

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
for c in list(ORDER) + FKEYS:
    print(f"  {c:8s} d_mean {ST[c]['d_mean']:+.4f} d_le1 {ST[c]['d_le1']:+d} d_gt2 {ST[c]['d_gt2']:+d} upgrades {ST[c]['upgrades']} up [{ST[c]['up_rows']}] down [{ST[c]['down_rows']}]")
