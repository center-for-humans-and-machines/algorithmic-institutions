"""fine = F*(shortfall/target)^gam in rounds < w, on surrogate v2."""
import torch as th, sys, itertools
sys.path.insert(0,'scripts/policy_finder/pf-20261105-7')
from harness import run
def mk(F,tg,gam,w):
    def r(C,P,t):
        c=C[...,-1]
        return F*((tg-c).clamp(min=0)/tg)**gam*float(t<w)
    return r
G=3000
res=[]
for tg,gam in itertools.product((13,15,17,20),(0.7,1,1.5,2.2)):
    for F in (24,30,36):
        m=run(mk(F,tg,gam,10),G=G,seed=5); res.append((m,tg,gam,F))
        print(f'tg {tg} gam {gam} F {F}: {m:.1f}',flush=True)
