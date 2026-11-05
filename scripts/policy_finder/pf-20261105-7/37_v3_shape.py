"""Candidate v3 family on sur3_all: fine = F*((20-c)/20)^gm for t<w; then scale*same for c<=floor until E."""
import sys; sys.argv=[sys.argv[0],'all']
exec(open('scripts/policy_finder/pf-20261105-7/35_net_sur3.py').read().split("run('zero'")[0])
def v3(F,gm,w,floor,E,sc): return mk(F,20,gm,w,floor,E,netF=F*sc)
run('ref s132 w11',mk(31.26,19,2.15,11))
for args in [(31,2.2,6,3,15,1),(31,2.2,6,3,15,0.6),(31,2.2,6,3,15,1.5),(31,2.2,8,3,15,0.8),(31,2.2,4,3,15,0.8),
             (38,2.2,6,3,15,0.8),(25,2.2,6,3,15,0.8),(31,1.6,6,3,15,0.8),(31,3.0,6,3,15,0.8),(31,2.2,6,0,15,1.0),(31,2.2,6,6,13,0.6),(31,2.2,6,3,19,0.6)]:
    run('v3 F%g gm%g w%d floor%d E%d sc%g'%args, v3(*args))
