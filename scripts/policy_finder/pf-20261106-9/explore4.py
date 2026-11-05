import pandas as pd, numpy as np
import warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
df = pd.read_pickle('/tmp/df.pkl').sort_values(['game','player_id','round_number'])
df['g_next'] = df.groupby(['game','player_id']).group_id.shift(-1)
df['block'] = df.round_number//4; df['pos'] = df.round_number%4
pp = df.pivot_table(index=['game','player_id','block'], columns='pos', values='punishment').add_prefix('p')
dec = df[df.pos==3].set_index(['game','player_id','block'])
dec = dec.join(pp)
dec['switch'] = (dec.g_next != dec.group_id).astype(int)
cg = df.groupby(['game','round_number','group_id']).common_good.first().unstack()
dec = dec.reset_index()
dec['cg_own'] = [cg.loc[(g,r)].get(gi, np.nan) for g,r,gi in zip(dec.game, dec.round_number, dec.group_id)]
dec['cg_oth'] = [cg.loc[(g,r)].get(1-gi, np.nan) for g,r,gi in zip(dec.game, dec.round_number, dec.group_id)]
dec['cg_oth'] = dec.cg_oth.fillna(0)
n = df.groupby(['game','round_number','group_id']).size().unstack().fillna(0)
dec['n_own'] = [n.loc[(g,r)].get(gi,0) for g,r,gi in zip(dec.game, dec.round_number, dec.group_id)]
dec = dec.dropna(subset=['switch'])
m = smf.logit('switch ~ p0 + p1 + p2 + p3 + I(cg_own/20) + I(cg_oth/20) + n_own + contribution', data=dec).fit(disp=0)
print(m.summary().tables[1])
# within-block contribution response timing
df['c_next'] = df.groupby(['game','player_id']).contribution.shift(-1)
s = df[(df.g_next==df.group_id)]
s['low'] = s.contribution <= 8
m2 = smf.ols('c_next ~ contribution + punishment*low + C(pos)', data=s).fit()
print(m2.summary().tables[1])
# persistence: c two rounds ahead
df['c_next2'] = df.groupby(['game','player_id']).contribution.shift(-2)
df['g_next2'] = df.groupby(['game','player_id']).group_id.shift(-2)
s2 = df[(df.g_next2==df.group_id)&(df.g_next==df.group_id)]
s2['low'] = s2.contribution <= 8
print(smf.ols('c_next2 ~ contribution + punishment*low', data=s2).fit().params)
