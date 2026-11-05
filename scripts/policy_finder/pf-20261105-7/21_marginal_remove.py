"""Marginal value of extra fines under v1-best on surrogate v2: remove fines at round t0 to players in a c bin."""
import torch as th, sys
sys.path.insert(0,'scripts/policy_finder/pf-20261105-7')
from harness import run
def v1(C,P,t): return 2.08*(14-C[...,-1]).clamp(min=0)*float(t<9)*max(0,1-.18*t/9)
G=8000
base=run(v1,G=G,seed=3)
print('base',round(base,1))
bins=[(0,4),(5,9),(10,13),(14,16),(17,19)]
print('t0 ' + ' '.join(f'{a}-{b}'.rjust(14) for a,b in bins))
for t0 in (0,1,2,3,4,6,8):
    row=[]
    for a,b in bins:
        def r(C,P,t,a=a,b=b,t0=t0):
            f=v1(C,P,t)
            if t==t0:
                c=C[...,-1]; f=f*(1-((c>=a)&(c<=b)).float())
            return f
        # count affected: fraction
        m=run(r,G=G,seed=3)
        row.append(m-base)
    print(t0, ' '.join(f'{x:14.2f}' for x in row), flush=True)
