"""Common-seed check of very short openings followed by the net (sur3_all, 3000 games)."""
import sys; sys.argv=[sys.argv[0],'all']
exec(open('scripts/policy_finder/pf-20261105-7/35_net_sur3.py').read().split("run('zero'")[0])
def v3(F,gm,w,floor,E,sc): return mk(F,20,gm,w,floor,E,netF=F*sc)
for w in [1,2,3,4,6]:
    for fl,sc in [(3,0.8),(6,0.6)]:
        run('v3 F38 gm2.5 w%d floor%d E16 sc%g'%(w,fl,sc), v3(38,2.5,w,fl,16,sc))
