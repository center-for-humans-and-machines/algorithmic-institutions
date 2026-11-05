# v6: at each switch, my group's size change vs my size, block punishment per head (both groups), decision-round mean c (both groups); v5 sweep
import numpy as np, sys
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from d1_arrays import load
tag = sys.argv[1] if len(sys.argv) > 1 else 'v5'
d = load(f'/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-11-{tag}_sweep/', tag)
c=d['c'].astype(np.float32); p=d['p'].astype(np.float32); m=d['mine']; opp=d['opp']
for be in [3,7,11,15,19]:
    bs=be-3
    nm=m[:,:,be].sum(1); no=8-nm; nn=m[:,:,be+1].sum(1)
    pm=(p[:,:,bs:be+1]*m[:,:,bs:be+1]).sum((1,2))/np.maximum(nm,1); po=(p[:,:,bs:be+1]*~m[:,:,bs:be+1]).sum((1,2))/np.maximum(no,1)
    cm=(c[:,:,be]*m[:,:,be]).sum(1)/np.maximum(nm,1); co=(c[:,:,be]*~m[:,:,be]).sum(1)/np.maximum(no,1)
    print(f'--- switch after r{be}')
    # transition table: E[new size | old size], P(new=0)
    for k in range(9):
        s=nm==k
        if s.sum()<50: continue
        print(f'  size {k}: n={s.sum():6d} E[new]={nn[s].mean():.2f} P(empty)={np.mean(nn[s]==0):.3f} P(8)={np.mean(nn[s]==8):.3f} mean p/head {pm[s].mean():.2f}')
    s=(nm>0)&(no>0)
    X=np.column_stack([np.ones(s.sum()),pm[s],po[s],cm[s],co[s],nm[s]])
    b=np.linalg.lstsq(X,(nn-nm)[s],rcond=None)[0]
    print(f'  d size: my p/head {b[1]:+.4f} their p/head {b[2]:+.4f} my c {b[3]:+.4f} their c {b[4]:+.4f} my n {b[5]:+.3f}')
