# v4: are sweep games paired across versions? The sweeps share seeds [42,106] and batch layout, so the
# episode at the same point index, seat and episode index should replay identically when both rules are zero-equivalent.
import sys, numpy as np, json
sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261112-1')
from v3_load import load
B='/opt/algorithmic-institutions/plots/simulation/policy_finder/'
part=sys.argv[1] if len(sys.argv)>1 else '1'
def key(d):
    idx=np.array([int(p.split('_s')[-1]) for p in d['point']])
    return idx
res={}
for v in ['pf-20261112-1','pf-20261112-1-v2','pf-20261112-1-v3']:
    d=load(B+f'{v}_sweep/per_round_p{part}of2.parquet')
    own=(d['G']==d['mine'][:,None,None])
    d['pun']=(d['P']*own).sum((1,2))
    d['idx']=key(d)
    res[v]=d
    print(v, d['C'].shape, d['idx'][:5], d['mine'][:5])
a,b,c=res['pf-20261112-1'],res['pf-20261112-1-v2'],res['pf-20261112-1-v3']
print('same index order v2/v3', (b['idx']==c['idx']).all(), (b['mine']==c['mine']).all(), (a['idx']==c['idx']).all())
for x,y,n in [(b,c,'v2-v3'),(a,c,'v1-v3'),(a,b,'v1-v2')]:
    both0=(x['pun']==0)&(y['pun']==0)
    same=(x['C']==y['C']).all((1,2))&(x['G']==y['G']).all((1,2))
    print(n,'both unpunished',both0.sum(),'identical among them',(same&both0).sum(), 'identical overall', same.sum())
