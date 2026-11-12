# v5: does v4's sweep (seeds [42,100]) pair with v3's (seeds [42,106]) game by game? Compare per slot
# (row order in the file) when both are unpunished; also by matching the point index.
import sys, numpy as np
sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261112-1')
from v3_load import load
B='/opt/algorithmic-institutions/plots/simulation/policy_finder/'
for part in ['1','2']:
    a=load(B+f'pf-20261112-1-v3_sweep/per_round_p{part}of2.parquet')
    b=load(B+f'pf-20261112-1-v4_sweep/per_round_p{part}of2.parquet')
    print(part, a['C'].shape, b['C'].shape)
    n=min(len(a['C']),len(b['C']))
    pa=((a['P']*(a['G']==a['mine'][:,None,None])).sum((1,2))==0)[:n]
    pb=((b['P']*(b['G']==b['mine'][:,None,None])).sum((1,2))==0)[:n]
    same=(a['C'][:n]==b['C'][:n]).all((1,2))&(a['G'][:n]==b['G'][:n]).all((1,2))
    print('both unpunished',(pa&pb).sum(),'identical',(same&pa&pb).sum(),'mine equal',(a['mine'][:n]==b['mine'][:n]).mean())
    ia=np.array([int(p.split('_s')[-1]) for p in a['point'][:n]]); ib=np.array([int(p.split('_s')[-1]) for p in b['point'][:n]])
    print('point idx equal',(ia==ib).mean())
    # by blocks of 250
    blk=np.arange(n)//250
    r=[(same&pa&pb)[blk==k].sum()/max(1,(pa&pb)[blk==k].sum()) for k in range(blk.max()+1)]
    print('per-block identical share', np.round(r,2)[:40])
# match by (point index, seat, episode order within that point+seat) across parts
def keyed(v):
    out={}
    for part in ['1','2']:
        d=load(B+f'{v}_sweep/per_round_p{part}of2.parquet')
        idx=np.array([int(p.split('_s')[-1]) for p in d['point']])
        pun=(d['P']*(d['G']==d['mine'][:,None,None])).sum((1,2))
        for (i,s) in set(zip(idx,d['mine'])):
            m=np.where((idx==i)&(d['mine']==s))[0]
            out[(i,s)]=(d['C'][m],d['G'][m],pun[m])
    return out
A=keyed('pf-20261112-1-v3'); Bv=keyed('pf-20261112-1-v4')
tot=0; ide=0
for k in Bv:
    if k not in A: continue
    ca,ga,pa=A[k]; cb,gb,pb=Bv[k]; n=min(len(ca),len(cb))
    m=(pa[:n]==0)&(pb[:n]==0); s=(ca[:n]==cb[:n]).all((1,2))&(ga[:n]==gb[:n]).all((1,2))
    tot+=m.sum(); ide+=(m&s).sum()
print('keyed: both unpunished',tot,'identical',ide)
