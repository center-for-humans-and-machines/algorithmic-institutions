"""What drives switching in the human data, and how persistent is the punishment effect?"""
import sys; sys.path.insert(0, 'scripts/policy_finder/pf-20261104-2')
from human_load import load
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
d = load()
dec = d[d.round_number.isin([3, 7, 11, 15, 19])].copy()
print('switch rate by decision round', dec.groupby('round_number').switched_next.mean().round(3).to_dict())
# own group vs other group stats at decision round
gs = d.groupby(['game', 'round_number', 'group_id']).agg(share=('common_good', 'first'), members=('participant_code', 'size'),
     mc=('contribution', 'mean'), mp=('punishment', 'mean')).reset_index()
# common_good in human data: check if per group total
print(gs.head())
other = gs.copy(); other['group_id'] = 1 - other.group_id
dec = dec.merge(other.rename(columns={'share': 'o_share', 'members': 'o_members', 'mc': 'o_mc', 'mp': 'o_mp'}), on=['game', 'round_number', 'group_id'], how='left')
dec['own_share'] = dec.common_good / dec.members
dec['oth_share'] = dec.o_share / dec.o_members
# punishment received over the last 4 rounds
d['p4'] = d.groupby(['game', 'participant_code']).punishment.transform(lambda x: x.rolling(4, 1).sum())
dec = dec.merge(d[['game', 'participant_code', 'round_number', 'p4']], on=['game', 'participant_code', 'round_number'])
dec['p4bin'] = pd.cut(dec.p4, [-1, 0, 3, 8, 20, 200])
print(dec.groupby('p4bin').switched_next.agg(['mean', 'size']).round(3))
dec['pbin'] = pd.cut(dec.punishment, [-1, 0, 3, 8, 30])
print(dec.groupby('pbin').switched_next.agg(['mean', 'size']).round(3))
dec['sharediff'] = dec.own_share - dec.oth_share
dec['sbin'] = pd.cut(dec.sharediff, [-100, -8, -3, 0, 3, 8, 100])
print(dec.groupby('sbin').switched_next.agg(['mean', 'size']).round(3))
# own group's mean punishment (do bystanders leave a punishing group?)
dec['others_p'] = (dec.grp_sum_p - dec.punishment) / (dec.members - 1).replace(0, np.nan)
dec['opbin'] = pd.cut(dec.others_p, [-1, 0, 1, 3, 30])
print('unpunished players, by groupmates punishment:')
print(dec[dec.punishment == 0].groupby('opbin').switched_next.agg(['mean', 'size']).round(3))
# by own contribution
dec['cbin'] = pd.cut(dec.contribution, [-1, 4, 9, 14, 20])
print(dec.groupby('cbin').switched_next.agg(['mean', 'size']).round(3))
# group size
print(dec.groupby('members').switched_next.agg(['mean', 'size']).round(3))
# logistic-ish OLS
X = np.column_stack([np.ones(len(dec)), dec.punishment, dec.p4, dec.sharediff.fillna(0), dec.contribution, dec.members])
b = np.linalg.lstsq(X, dec.switched_next.astype(float).values, rcond=None)[0]
print('LPM switch ~ 1, p, p4, sharediff, c, members:', b.round(4))

# persistence: contribution change k rounds later vs punishment now (under-contributors, same group throughout)
d = d.sort_values(['game', 'participant_code', 'round_number'])
g = d.groupby(['game', 'participant_code'])
for k in [1, 2, 3, 4]:
    d[f'c+{k}'] = g.contribution.shift(-k)
v = d[(d.player_no_input == 0) & (d.others_mean_c > d.contribution)].dropna(subset=['c+4'])
for k in [1, 2, 3, 4]:
    X = np.column_stack([np.ones(len(v)), v.contribution, v.others_mean_c - v.contribution, v.punishment])
    b = np.linalg.lstsq(X, (v[f'c+{k}'] - v.contribution).values, rcond=None)[0]
    print(f'k={k}: dc per punishment point {b[3]:.3f}')
