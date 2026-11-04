"""Human data: who switches groups at the decision points, and why."""
import numpy as np
import pandas as pd
d = pd.read_pickle('/tmp/pf5_human.pkl')
k = ['session', 'episode', 'participant_code']
# block of 4 rounds before each decision
d['block'] = d.round_number // 4
b = d.groupby(k + ['block']).agg(
    g=('group_id', 'last'), c=('contribution', 'mean'), p=('punishment', 'sum'),
    pay=('payoff', 'mean'), g_next=('g_next', 'last')).reset_index()
b = b[b.block < 5].dropna(subset=['g_next'])
b['switch'] = (b.g != b.g_next).astype(int)
# group-level per block: payoff of own group vs other, size, punishment total
gb = d.groupby(['session', 'episode', 'block', 'group_id']).agg(
    gpay=('payoff', 'mean'), gp=('punishment', 'mean'), gc=('contribution', 'mean'),
    n=('participant_code', 'nunique')).reset_index()
b = b.merge(gb, left_on=['session', 'episode', 'block', 'g'], right_on=['session', 'episode', 'block', 'group_id'])
o = gb.rename(columns={'group_id': 'og', 'gpay': 'opay', 'gp': 'op', 'gc': 'oc', 'n': 'on'})
b['og'] = 1 - b.g
b = b.merge(o, on=['session', 'episode', 'block', 'og'], how='left')
print('switch rate', b.switch.mean().round(3), 'by block', b.groupby('block').switch.mean().round(3).values)
print('\nby own punishment received in block')
print(b.groupby(pd.cut(b.p, [-1, 0, 3, 8, 20, 200]), observed=True).switch.agg(['mean', 'size']).round(3))
print('\nby group mean punishment per member-round minus other group')
b['dp'] = b.gp - b.op
print(b.groupby(pd.cut(b.dp, [-30, -2, -0.5, 0.5, 2, 30]), observed=True).switch.agg(['mean', 'size']).round(3))
b['dpay'] = b.gpay - b.opay
print('\nby own group payoff minus other group payoff')
print(b.groupby(pd.cut(b.dpay, [-50, -4, -1, 1, 4, 50]), observed=True).switch.agg(['mean', 'size']).round(3))
print('\nby own contribution')
print(b.groupby(pd.cut(b.c, [-1, 4, 9, 14, 20]), observed=True).switch.agg(['mean', 'size']).round(3))
from numpy.linalg import lstsq
bb = b.dropna(subset=['dpay', 'dp'])
X = np.c_[np.ones(len(bb)), bb.p / 4, bb.dp, bb.gc - bb.oc, bb.n]
print('LPM switch ~ 1 + own p/round + dgroup p + dgroup c + n:', lstsq(X, bb.switch.values, rcond=None)[0].round(3))
