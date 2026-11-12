# v4: paired comparison of final runs on the shared seed (seed 1000): per game, lead of each v3 final point
# vs the zero manager (v2 final gp, bar -1), same episode index and seat.
import sys, numpy as np
sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261112-1')
from v3_load import load
B='/opt/algorithmic-institutions/plots/simulation/policy_finder/'
def lead(d):
    C=d['C'].astype(float);P=d['P'].astype(float);G=d['G']
    own=(G==d['mine'][:,None,None]); oth=~own
    return ((1.6*C-P)*own).sum(1).mean(-1)-((1.6*C-P)*oth).sum(1).mean(-1)
def sub(d,m):
    return {k:v[m] for k,v in d.items()}
a=load(B+'pf-20261112-1-v2_sweep/final/per_round.parquet')
b=load(B+'pf-20261112-1-v3_sweep/final/per_round.parquet')
z=sub(a,a['point']=='pf-20261112-1-v2_gp'); lz=lead(z)
print('zero gp lead', lz.mean())
for d,tag in [(a,'v2'),(b,'v3')]:
  for p in np.unique(d['point']):
    x=sub(d,d['point']==p); lx=lead(x)
    same=(x['C']==z['C']).all((1,2))&(x['P']==z['P']).all((1,2))&(x['G']==z['G']).all((1,2))&(x['mine']==z['mine'])
    own=(x['G']==x['mine'][:,None,None])
    pun=(x['P']*own).sum((1,2))
    dd=lx-lz
    print(p, 'lead %.3f'%lx.mean(), 'identical %d'%same.sum(), 'punished games %d'%(pun>0).sum(),
          'paired diff %.3f se %.3f'%(dd.mean(), dd.std()/np.sqrt(len(dd))))
