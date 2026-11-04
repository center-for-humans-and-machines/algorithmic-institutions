"""Reference sims: AH contribution response to ah's punishment, and AH
switching vs punishment received and share gap (the stack the rule faces)."""
import pandas as pd
import statsmodels.formula.api as smf

d = pd.read_parquet("plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet")
d = d[d.run.str.contains("ah_vs_zero|zero_vs_ah|ah_vs_ah")].copy()
d["ep"] = d.run + d.episode.astype(str)
d = d.sort_values(["ep", "participant_code", "round_number"])
d["gmean"] = d.groupby(["ep", "round_number", "agent_group"]).contribution.transform("mean")
d["dev"] = d.contribution - d.gmean
g = d.groupby(["ep", "participant_code"])
d["c_next"] = g.contribution.shift(-1)
d["v_next"] = g.contribution_valid.shift(-1)
d["grp_next"] = g.agent_group.shift(-1)
# rule-managed side: ah side only (p>0 possible)
ahside = ((d.run.str.contains("ah_vs_zero")) & (d.agent_group == 0)) | ((d.run.str.contains("zero_vs_ah")) & (d.agent_group == 1)) | d.run.str.contains("ah_vs_ah")
x = d[ahside & d.contribution_valid & (d.v_next == True) & (d.grp_next == d.agent_group) & (d.round_number % 4 != 3)].copy()
x["dc"] = x.c_next - x.contribution
x["pbin"] = pd.cut(x.punishment, [-.1, 0, 2, 5, 10, 30])
x["cbin"] = pd.cut(x.contribution, [-.1, 5, 10, 15, 20])
print(x.pivot_table(index="cbin", columns="pbin", values="dc", aggfunc="mean", observed=True).round(2))
print(x.pivot_table(index="cbin", columns="pbin", values="dc", aggfunc="size", observed=True))
print(smf.ols("dc ~ punishment + I(punishment**2) + contribution + dev", x).fit().summary2().tables[1].round(4))

# switching
d["share"] = d.common_good
d["block"] = d.round_number // 4
b = d.groupby(["ep", "block", "participant_code"]).agg(pun=("punishment", "mean"), share=("share", "mean"), grp=("agent_group", "last"), sw=("grp_next", "last")).reset_index()
gs = d.groupby(["ep", "block", "agent_group"]).share.mean().rename("gshare").reset_index()
gs["other"] = 1 - gs.agent_group
b = b.merge(gs[["ep", "block", "other", "gshare"]].rename(columns={"other": "grp", "gshare": "oshare"}), on=["ep", "block", "grp"], how="left")
b = b[b.block < 5].dropna(subset=["sw"])
b["switch"] = (b.sw != b.grp).astype(float)
b["oshare"] = b.oshare.fillna(0)
b["gap"] = b.share - b.oshare
print("switch rate", b.switch.mean())
print(b.groupby(pd.cut(b.pun, [-.1, 0, 1, 3, 6, 30]), observed=True).switch.agg(["mean", "count"]))
print(b.groupby(pd.cut(b.gap, [-30, -6, -3, -1, 1, 3, 6, 30]), observed=True).switch.agg(["mean", "count"]))
print(smf.logit("switch ~ pun + gap + block", b).fit(disp=0).summary2().tables[1].round(4))
