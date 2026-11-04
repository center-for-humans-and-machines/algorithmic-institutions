"""Individual response in v1's sweep (my group, group 0): next-round contribution change by current
contribution and punishment, split by whether the player was already punished before (repeat
offender); and the switch-out rate by punishment."""
import numpy as np
import pandas as pd
pd.set_option('display.width', 250)
d = pd.read_parquet('plots/simulation/policy_finder/pf-20261104-7_sweep/per_round_p1of2.parquet',
                    columns=['run', 'episode', 'participant_code', 'round_number', 'punishment', 'contribution', 'agent_group', 'contribution_valid'])
d = d[d.episode < 150]
for col in ['run', 'participant_code']: d[col] = d[col].astype('category')
d = d.sort_values(['run', 'episode', 'participant_code', 'round_number']).reset_index(drop=True)
key = ['run', 'episode', 'participant_code']
g = d.groupby(key, sort=False)
d['c_next'] = g.contribution.shift(-1)
d['grp_next'] = g.agent_group.shift(-1)
d['c_next3'] = g.contribution.shift(-3)
d['grp_next3'] = g.agent_group.shift(-3)
d['p_before'] = g.punishment.cumsum() - d.punishment
d = d[d.agent_group == 0]
d['cb'] = pd.cut(d.contribution, [-1, 0, 4, 9, 12, 15, 20], labels=['0', '1-4', '5-9', '10-12', '13-15', '16-20'])
d['pb'] = pd.cut(d.punishment, [-1, 0, 3, 7, 12, 20, 30], labels=['0', '1-3', '4-7', '8-12', '13-20', '21-30'])
d['rep'] = np.where(d.p_before > 0, 'repeat', 'first')
s = d[(d.grp_next == 0) & (d.round_number < 23)]
s = s.assign(dc=s.c_next - s.contribution)
print('next-round change, rounds<23, staying'); print(s.pivot_table(index=['rep', 'cb'], columns='pb', values='dc', aggfunc='mean', observed=True).round(2))
print(s.pivot_table(index=['rep', 'cb'], columns='pb', values='dc', aggfunc='size', observed=True))
s3 = d[(d.grp_next3 == 0) & (d.round_number < 21)]
s3 = s3.assign(dc=s3.c_next3 - s3.contribution)
print('t+3 change'); print(s3.pivot_table(index='cb', columns='pb', values='dc', aggfunc='mean', observed=True).round(2))
# switching: at decision rounds 3,7,11,15,19
sw = d[d.round_number.isin([3, 7, 11, 15, 19])].copy()
sw['left'] = (sw.grp_next == 1).astype(float)
print('leave rate by this-round punishment and contribution'); print(sw.pivot_table(index='cb', columns='pb', values='left', aggfunc='mean', observed=True).round(2))
# zero-contributor persistence: of players contributing 0 at round t after already being punished >=20 before
z = d[(d.contribution == 0) & (d.round_number.between(1, 10))]
print('0-contributors: share contributing 0 next round, by prior punishment', z.assign(pp=pd.cut(z.p_before, [-1, 0, 20, 50, 1000]), z0=(z.c_next == 0)).groupby('pp', observed=True).z0.mean().round(2).to_dict())
