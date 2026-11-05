import torch as th, sys
sys.path.insert(0,'scripts/policy_finder/pf-20261105-7')
from harness import run
def v1(tg,sl,w,ta=0.):
    return lambda C,P,t: sl*(tg-C[...,-1]).clamp(min=0)*float(t<w)*max(0,1-ta*t/w)
run(lambda C,P,t: th.zeros_like(C[...,-1]),verbose=True,label='zero')
run(v1(14,2.08,9,.18),verbose=True,label='v1 best')
for w in (9,12,16,20,23):
    run(v1(14,2.08,w,.18),verbose=True,label=f'v1 w{w}')
# floor: fine = a + sl*shortfall if c<tg
def floor(tg,a,sl,w):
    return lambda C,P,t: ((C[...,-1]<tg).float()*(a+sl*(tg-C[...,-1])))*float(t<w)
for a in (2,4,6):
    for sl in (1,2):
        run(floor(14,a,sl,9),verbose=True,label=f'floor a{a} sl{sl}')
# threshold-only big fines: only shortfall >= gap
def gap(tg,sl,w,gp):
    return lambda C,P,t: sl*(tg-C[...,-1]).clamp(min=0)*((tg-C[...,-1])>=gp).float()*float(t<w)
for gp in (2,4,6):
    for w in (9,16,22):
        run(gap(14,2.08,w,gp),verbose=True,label=f'gap{gp} w{w}')
