"""Cap on the fine, and a late phase for free riders, on the surrogate."""
import numpy as np
src=open("scripts/policy_finder/pf-20261105-7/07_surrogate_sim.py").read().split("print(\"zero\"")[0]; g={}; exec(src,g); play=g["play"]
def rule(thr,s,T,cap=30,late_thr=0,late_s=0):
    def r(C,P,t):
        c=C[:,-1]
        early=np.minimum(cap,s*np.maximum(0,thr-c))*(t<T)
        late=late_s*np.maximum(0,late_thr-c)*(t>=T)*(t<23)
        return early+late
    return r
print("zero",play(lambda C,P,t:0*C[:,-1],n=3000))
for a in [(20,0.5,3),(20,0.5,3,6),(20,1,3,6),(20,1,3,4),(20,2,3,5),(20,0.5,3,30,5,1),(20,0.5,3,30,10,0.5),(20,0.5,3,30,5,2)]:
    print(a,play(rule(*a),n=3000))
