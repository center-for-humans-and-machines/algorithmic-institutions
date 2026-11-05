# v4: v3 final (_gp games only): per opponent, per round: my members, my pool, their pool, punishment per head both sides
import numpy as np, sys
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from d1_arrays import load
d = load('/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-11-v3_sweep/final/', 'v3final')
keep=d['pt']==-1
c=d['c'][keep].astype(float); p=d['p'][keep].astype(float); m=d['mine'][keep]; opp=d['opp'][keep]
np.set_printoptions(linewidth=250)
for o in range(1,5):
    s=opp==o; cm=c[s]; pm=p[s]; mm=m[s]
    pool_m=(1.6*cm*mm).sum(1)-(pm*mm).sum(1); pool_o=(1.6*cm*~mm).sum(1)-(pm*~mm).sum(1)
    print(f'--- opp {o} games {s.sum()} pool mine {pool_m.mean():.1f} theirs {pool_o.mean():.1f}')
    print(' members  ', np.round(mm.sum(1).mean(0),2))
    print(' pool mine', np.round(pool_m.mean(0),0))
    print(' pool oth ', np.round(pool_o.mean(0),0))
    print(' c mine   ', np.round((cm*mm).sum(1).sum(0)/mm.sum(1).sum(0),1))
    print(' c oth    ', np.round((cm*~mm).sum(1).sum(0)/(~mm).sum(1).sum(0),1))
    print(' p/head m ', np.round((pm*mm).sum(1).sum(0)/mm.sum(1).sum(0),1))
    print(' p/head o ', np.round((pm*~mm).sum(1).sum(0)/(~mm).sum(1).sum(0),1))
