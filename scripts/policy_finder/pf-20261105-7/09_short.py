"""Very short opening windows on the surrogate: T in 1..6, slopes, thresholds."""
import numpy as np
src=open("scripts/policy_finder/pf-20261105-7/07_surrogate_sim.py").read().split("print(\"zero\"")[0]; g={}; exec(src,g); play=g["play"]
def window(thr,s,T,base=0):
    return lambda C,P,t: (C[:,-1]<thr)*(base+s*(thr-C[:,-1]))*(t<T)
print("zero",play(lambda C,P,t:0*C[:,-1],n=3000))
for T in [1,2,3,4,6]:
  for thr,s in [(20,0.5),(20,1),(15,1),(15,2),(12,1.5),(20,1.5)]:
    print(T,thr,s,play(window(thr,s,T),n=3000))
