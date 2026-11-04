"""Reference sims: group pool, size, contribution per round in ah vs zero pairings."""
import pandas as pd
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
print(d.groupby('run').episode.nunique())
print(d[['agent_group','group_id']].drop_duplicates().head())
# group-level per round
g = d.groupby(['run','episode','round_number','group_id']).agg(
    members=('participant_code','size'), pool=('common_good','first'),
    contrib=('contribution','mean'), pun=('punishment','mean')).reset_index()
# fill empty groups with zeros
for run, gr in g.groupby('run'):
    print('\n', run)
    s = gr.groupby(['group_id']).agg(pool=('pool','mean'), members=('members','mean'),
                                     contrib=('contrib','mean'), pun=('pun','mean'))
    print(s)
    # by block
    gr = gr.assign(block=gr.round_number//4)
    print(gr.pivot_table(index='block', columns='group_id', values=['members','pool','contrib','pun'], aggfunc='mean').round(2))

# The real score: group sum of 1.6c - p (pool_scores.against_anchor)
import sys; sys.path.insert(0, 'src')
from aimanager.simulation.pool_scores import add_pool, against_anchor
add_pool(d)
for m, f in against_anchor(d, 'ah').items():
    print(m, 'pool %.2f se %.2f members %.2f' % (f.pool.mean(), f.pool.std()/len(f)**.5, f.members.mean()))
