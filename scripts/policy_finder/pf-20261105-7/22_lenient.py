"""Leniency for repeat-fined players: fine shrinks with the number of earlier fines, or stops after K fines."""
import torch as th, sys
sys.path.insert(0,'scripts/policy_finder/pf-20261105-7')
from harness import run
def mk(tg=14,sl=2.08,w=9,esc=0.,K=99,decay=None):
    def r(C,P,t):
        c=C[...,-1]
        if t>=w: return th.zeros_like(c)
        npun=(P>0).float().sum(-1) if P.shape[-1]>0 else th.zeros_like(c)
        f=sl*(tg-c).clamp(min=0)
        f=f*(1+esc*npun).clamp(min=0)*(npun<K).float()
        if decay is not None: f=f*decay**npun
        return f
    return r
G=6000
run(mk(),G=G,seed=4,verbose=True,label='v1')
for esc in (-0.05,-0.1,-0.15,-0.2):
    for w in (9,14,20):
        run(mk(esc=esc,w=w),G=G,seed=4,verbose=True,label=f'esc {esc} w{w}')
for K in (2,3,4,6):
    for w in (9,14,20):
        run(mk(K=K,w=w),G=G,seed=4,verbose=True,label=f'K {K} w{w}')
for d in (0.9,0.8,0.7):
    for w in (9,14,20):
        run(mk(decay=d,w=w),G=G,seed=4,verbose=True,label=f'decay {d} w{w}')
