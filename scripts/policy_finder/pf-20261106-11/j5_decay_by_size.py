# v6: causal check. Within each size bin at round 4 (after the first switch), regress my group's
# future pool (rounds 4-23, empty = 0) on log(decay) (exogenous: decay does not act in block 0),
# controlling log(rate), delta, mid, (kappa), opponent dummies. v3, v4, v5 sweeps.
import numpy as np, sys
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from d1_arrays import load
for tag in ['v3','v4','v5']:
    d = load(f'/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-11-{tag}_sweep/', tag)
    c=d['c'].astype(np.float32); p=d['p'].astype(np.float32); m=d['mine']
    pool=(1.6*c*m - p*m).sum(1)  # [game, round]
    fut=pool[:,4:].mean(1)
    n4=m[:,:,4].sum(1)
    P=d['params'].set_index(d['params'].name.str.extract(r'_s(\d+)')[0].astype(float).fillna(-1).astype(int))
    pt=d['pt']; ok=pt>=0
    def col(k): return P[k].reindex(pt).to_numpy() if k in P else np.zeros(len(pt))
    lr=np.log(col('rate')); ld=np.log(np.maximum(col('decay'),0.02)); de=col('delta'); mi=col('mid'); ka=col('kappa')
    O=np.stack([d['opp']==k for k in [2,3,4]],1).astype(float)
    print(f'=== {tag}')
    for lo,hi in [(1,1),(2,3),(4,5),(6,7),(8,8)]:
        s=ok&(n4>=lo)&(n4<=hi)
        X=np.column_stack([np.ones(s.sum()),ld[s],lr[s],de[s],mi[s],ka[s],O[s]])
        b,res,_,_=np.linalg.lstsq(X,fut[s],rcond=None)
        r=fut[s]-X@b; cov=np.linalg.pinv(X.T@X)*r.var()
        print(f'  size {lo}-{hi}: n={s.sum():6d} fut pool {fut[s].mean():6.1f}  dPool/dlog(decay) {b[1]:+.2f} ({np.sqrt(cov[1,1]):.2f})  dlog(rate) {b[2]:+.2f} ({np.sqrt(cov[2,2]):.2f})')
