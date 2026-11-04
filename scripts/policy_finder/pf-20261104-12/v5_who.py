"""v5: in a final per_round parquet (rule = group 0), who is punished at punished rounds: newcomers vs stayers,
by contribution band; share of the round's punishment bill each takes; then their next-block contribution and leaving."""
import sys, pandas as pd, numpy as np
f = sys.argv[1]; run = sys.argv[2] if len(sys.argv) > 2 else None
d = pd.read_parquet(f, columns=['run','episode','participant_code','round_number','contribution','punishment','agent_group','contribution_valid'])
if run: d = d[d.run.str.contains(run)]
else: d = d[d.run == d.run.iloc[0]]
d = d.sort_values(['episode','participant_code','round_number'])
g = d.groupby(['episode','participant_code'])
d['prev_group'] = g.agent_group.shift(1)
d['new'] = (d.prev_group != d.agent_group) & d.prev_group.notna()
blk = d.round_number // 4
d['block'] = blk
# newcomer status for the block = joined at block start
st = d[d.round_number % 4 == 0][['episode','participant_code','block','new']].rename(columns={'new':'newb'})
d = d.merge(st, on=['episode','participant_code','block'])
d['band'] = pd.cut(d.contribution, [-1, 2, 5, 8, 11, 14, 20])
x = d[(d.agent_group == 0) & (d.round_number.isin([0,1,4,5,8,9,12,13]))]
pd.set_option('display.width', 220)
print('mean punishment by round x newcomer x band'); print(x.pivot_table(index=['round_number','newb'], columns='band', values='punishment', aggfunc='mean', observed=True).round(1).to_string())
print('share of members (count) by band'); print(x.pivot_table(index=['round_number','newb'], columns='band', values='punishment', aggfunc='size', observed=True).div(d.episode.nunique()).round(2).to_string())
tot = x.groupby(['round_number','newb']).punishment.sum().unstack() / d.episode.nunique()
print('bill per episode, stayers vs newcomers'); print(tot.round(1))
