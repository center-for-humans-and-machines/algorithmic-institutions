"""Human managers: punishment level vs group's pool and size, per game-group."""
import sys; sys.path.insert(0, 'scripts/policy_finder/pf-20261104-2')
from human_load import load
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
d = load()
d['pool'] = 1.6 * d.contribution - d.punishment
gr = d.groupby(['game', 'round_number', 'group_id']).agg(pool=('pool', 'sum'), members=('participant_code', 'size'), p=('punishment', 'mean'), c=('contribution', 'mean')).reset_index()
full = pd.MultiIndex.from_product([d.game.unique(), range(24), [0, 1]], names=['game', 'round_number', 'group_id'])
gr = gr.set_index(['game', 'round_number', 'group_id']).reindex(full).reset_index()
gr[['pool', 'members']] = gr[['pool', 'members']].fillna(0)
gg = gr.groupby(['game', 'group_id']).agg(pool=('pool', 'mean'), members=('members', 'mean'), p=('p', 'mean'), c=('c', 'mean'))
gg['pbin'] = pd.qcut(gg.p, 4)
print(gg.groupby('pbin')[['pool', 'members', 'c', 'p']].mean().round(2), gg.groupby('pbin').size())
# within game: the group with the lower-punishing manager vs the other
w = gg.reset_index().pivot(index='game', columns='group_id', values=['pool', 'p', 'members'])
lowp_wins = ((w['p'][0] < w['p'][1]) == (w['pool'][0] > w['pool'][1]))
print('games where lower-punishing manager had larger pool:', lowp_wins.mean().round(2), 'of', len(w))
# by block, punishment mean and who it hits
d['block_pos'] = d.round_number % 4
print('human punishment by round position in block', d.groupby('block_pos').punishment.mean().round(2).to_dict())
d['deficit'] = d.others_mean_c - d.contribution
print('human mean punishment by deficit bin', d.groupby(pd.cut(d.deficit, [-21, -3, 0, 3, 6, 10, 21])).punishment.mean().round(2).to_dict())
