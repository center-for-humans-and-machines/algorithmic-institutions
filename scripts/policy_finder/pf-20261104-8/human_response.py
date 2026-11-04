"""Human data: next-round contribution change after punishment, by contribution
relative to the group's mean; and what human managers punished."""
import pandas as pd
import statsmodels.formula.api as smf

d = pd.read_csv("experiments/2group_8agent_50ep.csv")
d = d.drop_duplicates(["session", "episode", "round_number", "participant_code"])
d["game"] = d.session.astype(str) + "_" + d.episode.astype(str)
d = d.sort_values(["game", "player_id", "round_number"])
v = d[d.player_no_input == 0].copy()
d["gmean"] = d.groupby(["game", "round_number", "group_id"]).contribution.transform("mean")
d["dev"] = d.contribution - d.gmean
g = d.groupby(["game", "player_id"])
d["c_next"] = g.contribution.shift(-1)
d["same_next"] = g.group_id.shift(-1) == d.group_id
d["nv_next"] = g.player_no_input.shift(-1)
x = d[(d.player_no_input == 0) & (d.nv_next == 0) & d.same_next & (d.round_number % 4 != 3)].copy()
x["dc"] = x.c_next - x.contribution
x["pbin"] = pd.cut(x.punishment, [-.1, 0, 2, 5, 10, 30])
x["cbin"] = pd.cut(x.contribution, [-.1, 5, 10, 15, 20])
print(x.pivot_table(index="cbin", columns="pbin", values="dc", aggfunc="mean", observed=True).round(2))
print(x.pivot_table(index="cbin", columns="pbin", values="dc", aggfunc="size", observed=True))
print(smf.ols("dc ~ punishment + I(punishment**2) + contribution + dev", x).fit().summary2().tables[1].round(4))
# human managers' policy
y = d[(d.player_no_input == 0) & (d.manager_no_input == 0)]
print("mean punishment", y.punishment.mean(), "share punished", (y.punishment > 0).mean())
y = y.assign(cbin=pd.cut(y.contribution, [-.1, 0, 5, 10, 15, 19, 20]), devbin=pd.cut(y.dev, [-20, -8, -4, -1, 1, 4, 20]))
print(y.groupby("cbin", observed=True).punishment.agg(["mean", "count", lambda s: (s > 0).mean()]))
print(y.groupby("devbin", observed=True).punishment.agg(["mean", "count", lambda s: (s > 0).mean()]))
# per game: manager punishment level vs group pool and members (first block excluded)
gg = d.groupby(["game", "group_id"]).agg(p=("punishment", "mean"), c=("contribution", "mean"))
gg["members"] = d.groupby(["game", "group_id"]).size() / 24
gg["pool"] = d.assign(pool=1.6 * d.contribution - d.punishment).groupby(["game", "group_id"]).pool.sum() / 24
print(gg.corr().round(2))
print(gg.groupby(pd.qcut(gg.p, 4)).mean().round(2))
