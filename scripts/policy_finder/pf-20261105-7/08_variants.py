"""Variants on the surrogate: fading slope, drop punishment, relative threshold, short windows."""
import importlib.util, numpy as np, sys
spec=importlib.util.spec_from_file_location("s","scripts/policy_finder/pf-20261105-7/07_surrogate_sim.py")
src=open(spec.origin).read().split("print(\"zero\"")[0]; g={}; exec(src,g); play=g["play"]
def fade(thr,s,T):
    return lambda C,P,t: np.maximum(0,thr-C[:,-1])*s*max(0,1-t/T)
def window(thr,s,T,base):
    return lambda C,P,t: (C[:,-1]<thr)*(base+s*(thr-C[:,-1]))*(t<T)
def drop(s,T):
    return lambda C,P,t: (s*np.maximum(0,C[:,-2]-C[:,-1]) if t>0 else 0*C[:,-1])*(t<T)
def rel(margin,s,T):
    def r(C,P,t):
        c=C[:,-1]; o=(c.sum(1,keepdims=True)-c)/3
        return s*np.maximum(0,np.minimum(20,o+margin)-c)*(t<T)
    return r
for a in [(20,0.5,4),(20,0.5,8),(20,0.3,8),(20,0.3,12),(15,1,4),(15,0.7,8),(20,0.25,23)]:
    print("window",a,play(window(*a,0)))
for a in [(20,1,8),(20,0.7,12),(20,0.5,16),(15,1,12),(20,1,12)]:
    print("fade",a,play(fade(*a)))
for a in [(15,0,4,4),(15,0,4,8),(12,0,6,4)]:
    print("flat",a,play(window(a[0],0,a[2],a[3])))
for a in [(1,8),(1,23),(2,23),(0.5,23)]:
    print("drop",a,play(drop(*a)))
for a in [(0,1,6),(3,1,6),(3,0.5,12),(5,0.5,8)]:
    print("rel",a,play(rel(*a)))
