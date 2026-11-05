import torch as th, sys
sys.path.insert(0,'scripts/policy_finder/pf-20261105-7')
from harness import run
def mk(tg=14,sl=2.08,w=9,s0=None,tg2=None,sl2=0.,E=23,drop=None,kd=0.):
    def r(C,P,t):
        c=C[...,-1]
        if t<w:
            s=s0 if (s0 is not None and t==0) else sl
            return s*(tg-c).clamp(min=0)
        out=th.zeros_like(c)
        if t<E:
            if tg2 is not None: out=out+sl2*(tg2-c).clamp(min=0)
            if drop is not None and C.shape[-1]>1:
                d=C[...,-2]-c
                out=out+kd*d*(d>=drop).float()
        return out
    return r
run(mk(),verbose=True,label='v1')
for s0 in (1,1.5,3):
    run(mk(s0=s0),verbose=True,label=f's0 {s0}')
for tg2,sl2,E in [(10,2,18),(8,3,18),(12,1,18),(10,2,21),(6,3,21),(10,1,21)]:
    run(mk(tg2=tg2,sl2=sl2,E=E),verbose=True,label=f'maint tg2 {tg2} sl2 {sl2} E{E}')
for drop,kd,E in [(2,1,18),(2,2,18),(4,2,18),(4,3,21),(3,1,21),(6,2,21)]:
    run(mk(drop=drop,kd=kd,E=E),verbose=True,label=f'drop {drop} kd {kd} E{E}')
