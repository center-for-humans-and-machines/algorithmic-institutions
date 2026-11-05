# v6: does the membership cost of my block punishment depend on my group size? d size ~ p/head * size bins, + c's; v5 sweep
import numpy as np, sys
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from d1_arrays import load
tag = sys.argv[1] if len(sys.argv) > 1 else 'v5'
d = load(f'/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-11-{tag}_sweep/', tag)
c=d['c'].astype(np.float32); p=d['p'].astype(np.float32); m=d['mine']
for be in [7,11,15]:
    bs=be-3
    nm=m[:,:,be].sum(1); no=8-nm; nn=m[:,:,be+1].sum(1)
    pm=(p[:,:,bs:be+1]*m[:,:,bs:be+1]).sum((1,2))/np.maximum(nm,1); po=(p[:,:,bs:be+1]*~m[:,:,bs:be+1]).sum((1,2))/np.maximum(no,1)
    cm=(c[:,:,be]*m[:,:,be]).sum(1)/np.maximum(nm,1); co=(c[:,:,be]*~m[:,:,be]).sum(1)/np.maximum(no,1)
    print(f'--- switch after r{be}: per size bin, d size on my p/head (block), my c, their c, their p/head')
    for lo,hi in [(2,3),(4,5),(6,7),(8,8)]:
        s=(nm>=lo)&(nm<=hi)
        X=np.column_stack([np.ones(s.sum()),pm[s],cm[s],co[s],po[s],nm[s]])
        b=np.linalg.lstsq(X,(nn-nm)[s],rcond=None)[0]
        print(f'  size {lo}-{hi}: n={s.sum()} my p/head {b[1]:+.4f}  my c {b[2]:+.3f} their c {b[3]:+.3f} their p {b[4]:+.4f}; per member: p {b[1]/nm[s].mean():+.4f} c {b[2]/nm[s].mean():+.3f}')
