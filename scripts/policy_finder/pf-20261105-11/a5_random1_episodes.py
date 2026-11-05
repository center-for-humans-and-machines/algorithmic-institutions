# Per-episode punishment strength in random_1 pilots vs outcomes
import pandas as pd, numpy as np
d=pd.read_csv('experiments/pilot_random1_player_round_slim.csv')
r=d[(d.experiment_name=='random_1')&(d.player_no_input==0)].copy()
r['def']=20-r.contribution
rows=[]
for e,z in r.groupby('episode_id'):
    X=np.c_[np.ones(len(z)),z['def']]
    b=np.linalg.lstsq(X,z.punishment,rcond=None)[0]
    pos=z[z.punishment>0]
    rows.append(dict(ep=e,slope=b[1],icpt=b[0],thr=20-(pos['def'].min() if len(pos) else 20),
        c0=z[z.round_number==0].contribution.mean(), c_late=z[z.round_number>=10].contribution.mean(),
        p=z.punishment.mean(), pay=z.payoff.mean(), pay_late=z[z.round_number>=10].payoff.mean(),
        p_late=z[z.round_number>=10].punishment.mean()))
s=pd.DataFrame(rows)
s['sb']=pd.qcut(s.slope,4)
print(s.groupby('sb',observed=True)[['slope','c0','c_late','p','p_late','pay','pay_late']].mean().round(2))
print(s[['slope','c0','c_late','p','pay','pay_late']].corr().round(2))
X=np.c_[np.ones(len(s)),s.slope,s.c0]
print('c_late ~ 1, slope, c0', np.linalg.lstsq(X,s.c_late,rcond=None)[0].round(2))
print('pay ~ 1, slope, c0', np.linalg.lstsq(X,s.pay,rcond=None)[0].round(2))
print('pay_late ~ 1, slope, c0', np.linalg.lstsq(X,s.pay_late,rcond=None)[0].round(2))
h=d[(d.experiment_name=='trail_rounds_2')&(d.player_no_input==0)]
print('human mgr: c', h.contribution.mean().round(2),'p',h.punishment.mean().round(2),'pay',h.payoff.mean().round(2))
