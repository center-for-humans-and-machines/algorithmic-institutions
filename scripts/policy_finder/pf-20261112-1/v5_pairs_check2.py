# v5: hypothesis: batch k of a part uses seed (part's seed)+k, games pair by seed and slot within batch.
# v4 part2 starts at seed 100 -> v4 p2 batch 6+j == v3 p2 batch j (seed 106+j); v4 p2 batch k<6 == v3 p1 batch 58+k.
import sys, numpy as np
sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261112-1')
from v3_load import load
B='/opt/algorithmic-institutions/plots/simulation/policy_finder/'
a1=load(B+'pf-20261112-1-v3_sweep/per_round_p1of2.parquet'); a2=load(B+'pf-20261112-1-v3_sweep/per_round_p2of2.parquet')
b2=load(B+'pf-20261112-1-v4_sweep/per_round_p2of2.parquet')
def cmp(x,xs,y,ys,n):
    px=((x['P'][xs:xs+n]*(x['G'][xs:xs+n]==x['mine'][xs:xs+n,None,None])).sum((1,2))==0)
    py=((y['P'][ys:ys+n]*(y['G'][ys:ys+n]==y['mine'][ys:ys+n,None,None])).sum((1,2))==0)
    s=(x['C'][xs:xs+n]==y['C'][ys:ys+n]).all((1,2))
    return (px&py).sum(), (px&py&s).sum()
print('v4p2[6000:] vs v3p2[0:]', cmp(b2,6000,a2,0,51500))
print('v4p2[0:6000] vs v3p1[58000:]', cmp(b2,0,a1,58000,6000))
