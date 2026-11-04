"""Reference sims, ah-managed side: (a) persistence of a punishment's effect on
contributions k rounds later; (b) switch sensitivity to punishment by position
in the 4-round block; (c) pool by block for ah vs zero sides."""
import pandas as pd
import statsmodels.formula.api as smf

d = pd.read_parquet("plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet")
d = d[d.run.str.contains("ah_vs_zero|zero_vs_ah|ah_vs_ah")].copy()
d["ep"] = d.run + d.episode.astype(str)
d = d.sort_values(["ep", "participant_code", "round_number"])
d["gmean"] = d.groupby(["ep", "round_number", "agent_group"]).contribution.transform("mean")
d["dev"] = d.contribution - d.gmean
g = d.groupby(["ep", "participant_code"])
for k in (1, 2, 3):
    d[f"c{k}"] = g.contribution.shift(-k)
    d[f"grp{k}"] = g.agent_group.shift(-k)
x = d[d.contribution_valid & (d.round_number % 4 == 0) & (d.grp3 == d.agent_group)]
for k in (1, 2, 3):
    r = smf.ols(f"c{k} ~ punishment + contribution + dev + gmean", x).fit()
    print("k", k, "punishment coef", r.params["punishment"].round(4), "se", r.bse["punishment"].round(4))
# low contributors only
xl = x[x.dev < -2]
for k in (1, 2, 3):
    r = smf.ols(f"c{k} ~ punishment + I(punishment**2) + contribution + dev + gmean", xl).fit()
    print("below-mean k", k, r.params[["punishment", "I(punishment ** 2)"]].round(4).tolist())

# switching vs punishment per position in block
d["pos"] = d.round_number % 4
d["block"] = d.round_number // 4
w = d.pivot_table(index=["ep", "block", "participant_code"], columns="pos", values="punishment").add_prefix("p")
s = d.groupby(["ep", "block", "participant_code"]).agg(share=("common_good", "mean"), grp=("agent_group", "last"))
s["nxt"] = d[d.pos == 3].set_index(["ep", "block", "participant_code"]).grp1
s = s.join(w).reset_index()
s = s[s.block < 5].dropna()
s["switch"] = (s.nxt != s.grp).astype(float)
print(smf.logit("switch ~ p0 + p1 + p2 + p3 + share + block", s).fit(disp=0).params.round(4))
