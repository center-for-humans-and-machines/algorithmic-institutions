"""Human data: who switches, as a function of punishment received and the
per-member share gap between own and other group, at the 5 decisions."""
import numpy as np
import pandas as pd

d = pd.read_csv("experiments/2group_8agent_50ep.csv")
d = d.drop_duplicates(["session", "episode", "round_number", "participant_code"])
d["game"] = d.session.astype(str) + "_" + d.episode.astype(str)
print("games", d.game.nunique(), "rows", len(d))
d = d.sort_values(["game", "player_id", "round_number"])
# switch: group changes from round r to r+1, for r in 3,7,11,15,19
d["next_group"] = d.groupby(["game", "player_id"]).group_id.shift(-1)
dec = d[d.round_number.isin([3, 7, 11, 15, 19])].copy()
dec["switch"] = (dec.next_group != dec.group_id).astype(float)
print("switch rate", dec.switch.mean())
# block stats: last 4 rounds of own punishment, own group's share, other group's share
d["block"] = d.round_number // 4
d["nvalid"] = d.groupby(["game","round_number","group_id"]).player_no_input.transform(lambda x: (1-x).sum()).clip(lower=1)
d["common_good"] = d.common_good / d.nvalid  # per-member share, as the sim feeds the switch model
share = d.groupby(["game", "block", "group_id"]).common_good.mean().rename("share").reset_index()
own = d.groupby(["game", "block", "player_id"]).agg(pun=("punishment", "mean"), contr=("contribution", "mean"), gid=("group_id", "last")).reset_index()
own = own.merge(share.rename(columns={"group_id": "gid", "share": "own_share"}), on=["game", "block", "gid"], how="left")
share["og"] = 1 - share.group_id
own = own.merge(share[["game", "block", "og", "share"]].rename(columns={"og": "gid", "share": "oth_share"}), on=["game", "block", "gid"], how="left")
dec["block"] = dec.round_number // 4
m = dec[["game", "block", "player_id", "switch"]].merge(own, on=["game", "block", "player_id"])
m["gap"] = m.own_share - m.oth_share
print(m.groupby(pd.cut(m.pun, [-.1, 0, 1, 3, 6, 30])).switch.agg(["mean", "count"]))
print(m.groupby(pd.cut(m.gap, [-30, -6, -3, -1, 1, 3, 6, 30])).switch.agg(["mean", "count"]))
mm = m.dropna(subset=["oth_share"])
import statsmodels.formula.api as smf
print(smf.logit("switch ~ pun + gap + contr + block", mm).fit(disp=0).summary2().tables[1])
