# v4: in the v3 sweep games, does a point of punishment work differently at round 0, 1, 4, 5, 8? next-round change and change to the block's end, plus leaving at the next switch
import numpy as np, sys
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from d1_arrays import load
d = load('/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-11-v3_sweep/', 'v3')
c=d['c'].astype(np.float32); p=d['p'].astype(np.float32); m=d['mine']; v=d['v']
n=m.sum(1,keepdims=True)
others=( (c*m).sum(1,keepdims=True)-c*m)/np.maximum(n-1,1)
gap=c-others   # negative = below norm
for t in [0,1,4,5,8,9]:
    blk_end = (t//4)*4+3
    s=m[:,:,t]&m[:,:,t+1]&(n[:,0,t]>1)[:,None]
    g=gap[:,:,t][s]; pp=p[:,:,t][s]; dc=(c[:,:,t+1]-c[:,:,t])[s]
    stay=m[:,:,blk_end]&m[:,:,t]
    leave=(m[:,:,t]&m[:,:,blk_end]&~m[:,:,blk_end+1])[s] if blk_end<23 else None
    # persistent: mean c over rounds blk_end+1..blk_end+4 for those who stay in group, minus c[t]
    X=np.c_[np.ones(len(g)),np.minimum(g,0),np.maximum(g,0),pp, c[:,:,t][s]]
    b=np.linalg.lstsq(X,dc,rcond=None)[0]
    lv=leave.astype(float)
    bl=np.linalg.lstsq(X,lv,rcond=None)[0]
    # longer-run: c at t+4 (if still in group)
    s4=s&m[:,:,t+4]
    X4=np.c_[np.ones(s4.sum()),np.minimum(gap[:,:,t][s4],0),np.maximum(gap[:,:,t][s4],0),p[:,:,t][s4],c[:,:,t][s4]]
    b4=np.linalg.lstsq(X4,(c[:,:,t+4]-c[:,:,t])[s4],rcond=None)[0]
    print(f't={t}: n={s.sum()} mean p={pp.mean():.2f}  dc next: per p {b[3]:.3f}  | dc to t+4 (stayers): per p {b4[3]:.3f} | P(leave at switch) per p {bl[3]:.4f} base {lv.mean():.3f}')
