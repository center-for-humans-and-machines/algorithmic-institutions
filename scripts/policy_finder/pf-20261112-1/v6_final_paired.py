# v6: pair every point of v5's final run (and v3/v4 finals) with the zero manager's final games (v1 final, seed 1000),
# slot by slot within each seat; prints the paired lead change, its se, and the share of games touched.
import sys, numpy as np
sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261112-1')
from v3_load import load
B='/opt/algorithmic-institutions/plots/simulation/policy_finder/'
def games(f):
    d=load(f); own=(d['G']==d['mine'][:,None,None]); pr=1.6*d['C'].astype(np.float32)-d['P']
    lead=((pr*own).sum(1)-(pr*~own).sum(1)).mean(1); po=(pr*own).sum(1).mean(1)
    out={}
    for nm in np.unique(d['point']):
        for s in (0,1):
            m=(d['point']==nm)&(d['mine']==s); out[(nm,s)]=(lead[m],po[m],(d['P'][m]*own[m]).sum((1,2)),d['C'][m])
    return out
z=games(B+'pf-20261112-1_sweep/final/per_round.parquet')
zk=[k for k in z if k[0]=='pf-20261112-1_gp']
for v in ['pf-20261112-1-v3','pf-20261112-1-v4','pf-20261112-1-v5']:
    g=games(B+v+'_sweep/final/per_round.parquet')
    for nm in sorted(set(k[0] for k in g)):
        dl=[];tc=[];same=[]
        for s in (0,1):
            a=g[(nm,s)]; b=z[('pf-20261112-1_gp',s)]; n=min(len(a[0]),len(b[0]))
            dl.append(a[0][:n]-b[0][:n]); tc.append(a[2][:n]>0); same.append((a[3][:n]==b[3][:n]).all((1,2))&(a[2][:n]==0))
        dl=np.concatenate(dl); tc=np.concatenate(tc); same=np.concatenate(same)
        print(nm, 'n',len(dl),'paired %.3f se %.3f'%(dl.mean(),dl.std()/np.sqrt(len(dl))),'touched %.2f'%tc.mean(),
              'untouched identical %.3f'%(same.sum()/max(1,(~tc).sum())), 'per touched %.2f'%(dl[tc].mean() if tc.any() else 0))
