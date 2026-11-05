# v7: per-round decomposition of both groups' pool at a final's peak point (_gp): members, c/head, deduction
# usage: l1_decompose.py <final dir> <tag>   (arrays cached via d1_arrays.load)
import sys, numpy as np
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from d1_arrays import load
d = load(sys.argv[1], sys.argv[2])
gp = np.array(['_gp' in r for r in d['run']])
np.set_printoptions(linewidth=200)
for o in range(1,5):
    sel = gp & (d['opp']==o)
    c = d['c'][sel].astype(float); p = d['p'][sel].astype(float); m = d['mine'][sel]
    for name, mm in [('mine', m), ('theirs', ~m)]:
        n = mm.sum(1); cs = (c*mm).sum(1); ps = (p*mm).sum(1)
        pool = (1.6*cs - ps)
        print(o, name, 'pool', round(pool.mean(),2), 'gross', round(1.6*cs.mean(),2), 'ded', round(ps.mean(),2), 'n', round(n.mean(),2), 'c/h', round(cs.sum()/n.sum(),2), 'P(empty)', round((n==0).mean(),3))
        if name=='mine':
            print('  ded/rd', np.round(ps.mean(0),1)); print('  n/rd  ', np.round(n.mean(0),2)); print('  c/h/rd', np.round(cs.sum(0)/n.sum(0),1))
