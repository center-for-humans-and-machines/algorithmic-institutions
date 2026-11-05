# v5: what drives inflow into my group at a switch? per game and switch, count joiners vs my group's block punishment per head, its mean c in the block's last round, their block punishment and c; v4 sweep
import numpy as np, sys
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from d1_arrays import load
d = load('/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-11-v4_sweep/', 'v4')
c=d['c'].astype(np.float32); p=d['p'].astype(np.float32); m=d['mine']; opp=d['opp']
for be in [3,7,11]:
    bs=be-3
    nm=m[:,:,be].sum(1); no=8-nm
    pm=(p[:,:,bs:be+1]*m[:,:,bs:be+1]).sum((1,2))/np.maximum(nm,1); po=(p[:,:,bs:be+1]*~m[:,:,bs:be+1]).sum((1,2))/np.maximum(no,1)
    cm=(c[:,:,be]*m[:,:,be]).sum(1)/np.maximum(nm,1); co=(c[:,:,be]*~m[:,:,be]).sum(1)/np.maximum(no,1)
    inflow=((~m[:,:,be])&m[:,:,be+1]).sum(1)/np.maximum(no,1)   # share of the other group joining
    outflow=(m[:,:,be]&~m[:,:,be+1]).sum(1)/np.maximum(nm,1)
    s=(nm>0)&(no>0)
    X=np.column_stack([np.ones(s.sum()),pm[s],po[s],cm[s],co[s],nm[s]])
    bi=np.linalg.lstsq(X,inflow[s],rcond=None)[0]; bo=np.linalg.lstsq(X,outflow[s],rcond=None)[0]
    print(f'switch after r{be}: inflow share {inflow[s].mean():.3f}: my p/head {bi[1]:+.4f} their p/head {bi[2]:+.4f} my c {bi[3]:+.4f} their c {bi[4]:+.4f} my n {bi[5]:+.3f}')
    print(f'                 outflow share {outflow[s].mean():.3f}: my p/head {bo[1]:+.4f} their p/head {bo[2]:+.4f} my c {bo[3]:+.4f} their c {bo[4]:+.4f} my n {bo[5]:+.3f}')
