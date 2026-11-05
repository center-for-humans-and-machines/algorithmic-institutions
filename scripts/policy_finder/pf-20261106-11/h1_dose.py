# v5: dose-response shape at rounds 0-1 in the v3+v4 sweeps: next-round and t+4 contribution change, and leaving at the first switch, by punishment bin, for members by shortfall bin
import numpy as np, sys
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from d1_arrays import load
for tag in ['v3','v4']:
    d = load(f'/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-11-{tag}_sweep/', tag)
    c=d['c'].astype(np.float32); p=d['p'].astype(np.float32); m=d['mine']
    n=m.sum(1,keepdims=True)
    others=((c*m).sum(1,keepdims=True)-c*m)/np.maximum(n-1,1)
    gap=c-others
    for t in [0,1]:
        s=m[:,:,t]&(n[:,0,t]>1)[:,None]
        print(f'== {tag} t={t}')
        for lo,hi in [(-20,-8),(-8,-4),(-4,-1.5)]:
            sel=s&(gap[:,:,t]>=lo)&(gap[:,:,t]<hi)
            row=[]
            for plo,phi in [(0,1),(1,5),(5,10),(10,15),(15,20),(20,29),(29,31)]:
                ss=sel&(p[:,:,t]>=plo)&(p[:,:,t]<phi)
                st=ss&m[:,:,t+1]; st4=ss&m[:,:,4]&m[:,:,3]
                if ss.sum()<200: row.append(f'p{plo}-{phi}: -'); continue
                dc=(c[:,:,t+1]-c[:,:,t])[st].mean(); dc4=(c[:,:,4]-c[:,:,t])[st4].mean()
                lv=(m[:,:,3]&~m[:,:,4])[ss].mean()
                row.append(f'p{plo}-{phi}: n{ss.sum()} dc{dc:+.2f} dc4{dc4:+.2f} lv{lv:.3f}')
            print(f' gap[{lo},{hi})', ' | '.join(row))
