"""Baselines for 35: the s132 schedule with windows 4-14 and no net; and nets with a fixed big fine."""
import sys; sys.argv=[sys.argv[0],'all']
exec(open('scripts/policy_finder/pf-20261105-7/35_net_sur3.py').read().split("run('zero'")[0])
for w in [4,6,8,11,14]: run(f's132 w{w} no net',mk(31.26,19,2.15,w))
for floor in [2,4]:
    for E in [12,14,16]:
        run(f's132 w6 floor{floor} E{E} net fine 30',mk(31.26,19,2.15,6,floor,E,netF=60))
        run(f's132 w6 floor{floor} E{E} net fine x0.6',mk(31.26,19,2.15,6,floor,E,netF=19))
