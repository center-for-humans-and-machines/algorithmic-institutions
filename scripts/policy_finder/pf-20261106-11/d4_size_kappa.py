# v3: (a) v2 sweep: pool per opponent vs kappa/late/rate among good points; (b) switch logit with punishment x group size
import numpy as np, pandas as pd, json, sys
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
S = json.load(open('/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-11-v2_sweep/sweep.json'))
df = pd.DataFrame([dict(**x['params'], **{o: v['pool'] for o, v in x['pools'].items()}, **{'m'+o[-1]: v['members'] for o, v in x['pools'].items()}) for x in S['points']])
good = df[(df.delta < 0) & (df.delta > -3.5) & (df.rate > 2.5)]
print('good points', len(good))
import statsmodels.formula.api as smf
for o in ['opponent_1','opponent_2','opponent_3','opponent_4']:
    r = smf.ols(f'{o} ~ kappa + late + np.log(rate) + delta + floor', good).fit()
    print(o, r.params.round(2).to_dict(), 'se kappa', round(r.bse['kappa'],2))
from d1_arrays import load
d = load()
rng = np.random.default_rng(1); idx = rng.choice(len(d['c']), 30000, replace=False)
c=d['c'][idx].astype(float); p=d['p'][idx].astype(float); m=d['mine'][idx]
rows=[]
for t in [3,7,11,15,19]:
    blk=slice(t-3,t+1); inb=m[:,:,t]; n=inb.sum(1,keepdims=True)*np.ones((1,8))
    no=(~m[:,:,t]).sum(1,keepdims=True)*np.ones((1,8))
    sh_own=(c[:,:,t]*inb).sum(1,keepdims=True)/np.maximum(inb.sum(1,keepdims=True),1)
    sh_oth=(c[:,:,t]*~inb).sum(1,keepdims=True)/np.maximum((~inb).sum(1,keepdims=True),1)
    gap=(1.6*(sh_oth-sh_own))*np.ones((1,8))
    rows.append(pd.DataFrame(dict(t=t, n=n[inb], pm=p[:,:,blk].mean(2)[inb], c=c[:,:,blk].mean(2)[inb], gap=gap[inb], other_empty=(no[inb]==0), leave=(~m[:,:,t+1])[inb].astype(float))))
x=pd.concat(rows); x=x[~x.other_empty]
r=smf.logit('leave ~ pm*n + c + gap + C(t)', x).fit(disp=0); print(r.params.round(3))
x['nb']=pd.cut(x.n,[0,2,3,4,5,8]); x['pb']=pd.cut(x.pm,[-1,0.5,3,6,10,30])
print(x.pivot_table(index='nb',columns='pb',values='leave',aggfunc='mean',observed=True).round(3))
