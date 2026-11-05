"""Score candidate v7 designs on 79/83's block-adjusted pooled GP: design mean, share >= 655,
and the true value of the top-5 sweep picks under sweep noise (se 2.5), averaged over 2000 draws."""
import pickle, numpy as np, yaml, copy, sys
sys.path.insert(0,'src')
from aimanager.manager.rule import sobol_design
G=pickle.load(open('/tmp/pf7_gp79.pkl','rb')); gp,mu,sd=G['gp'],G['mu'],G['sd']
K=['mid_fine','curvature','window','gate_margin','net_floor','net_end','net_fine']
def f(P):
    X=np.array([[p[k] for k in K] for p in P],float); Z=X.copy(); Z[:,0]=np.log(Z[:,0]); Z[:,1]=np.log(Z[:,1])
    return gp.predict((Z-mu)/sd)
base=yaml.safe_load(open('configs/managers/rule_based/pf-20261105-7-v6.yml'))
D={
 'v6':base['sweep_config'],
 'B':dict(mid_fine=[6,8.5,'log'],curvature=[1.95,2.6,'log'],window=[5,7],gate_margin=[2.5,8],net_floor=2,net_end=[13,17],net_fine=[19,27]),
 'C':dict(mid_fine=[6.3,8,'log'],curvature=[2.0,2.5,'log'],window=[6,7],gate_margin=[3,8],net_floor=2,net_end=[14,16],net_fine=[21,26]),
 'B_nogate':dict(mid_fine=[6,8.5,'log'],curvature=[1.95,2.6,'log'],window=[5,7],gate_margin=8,net_floor=2,net_end=[13,17],net_fine=[19,27]),
 'B_floor':dict(mid_fine=[6,8.5,'log'],curvature=[1.95,2.6,'log'],window=[5,7],gate_margin=[2.5,8],net_floor=[1,3],net_end=[13,17],net_fine=[19,27]),
 'M':dict(mid_fine=[5.5,9,'log'],curvature=[1.85,2.8,'log'],window=[5,7],gate_margin=[2,8],net_floor=[1,3],net_end=[13,17],net_fine=[18,28]),
}
rng=np.random.default_rng(1)
for n,sc in D.items():
    r=copy.deepcopy(base); r['sweep_config']=sc
    P=sobol_design(r,256); y=f(P)
    top=[]; best=[]
    for _ in range(2000):
        o=np.argsort(-(y+rng.normal(0,2.5,len(y))))[:5]; top.append(y[o].mean()); best.append(y[o[1]])
    print(f'{n:9s} n{len(P)} mean {y.mean():.2f} sd {y.std():.2f} >=655 {np.mean(y>=655):.2f} max {y.max():.2f} top5 true {np.mean(top):.2f} 2nd-pick {np.mean(best):.2f}')
v3gp=dict(mid_fine=29.71*0.5**2.177,curvature=2.177,window=7,gate_margin=8,net_floor=2,net_end=16,net_fine=25.2)
print('v3_gp on GP', f([v3gp]))
