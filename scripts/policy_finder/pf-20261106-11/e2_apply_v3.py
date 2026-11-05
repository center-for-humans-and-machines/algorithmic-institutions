# v3: per-head punishment the v3 rule would deal on the v2 final (_gp) histories, by block, for a few param sets
import numpy as np, torch as th, yaml, sys
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from d1_arrays import load
from aimanager.manager.rule import run_rule
d = load('/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-11-v2_sweep/final/', 'v2final')
k = np.nonzero(d['pt'] == -1)[0][:2000]
C = th.tensor(d['c'][k]).float(); V = th.tensor(d['v'][k]).float(); P = th.tensor(d['p'][k]).float(); M = th.tensor(d['mine'][k]).float()
code = compile(yaml.safe_load(open('/opt/algorithmic-institutions/configs/managers/rule_based/pf-20261106-11-v3.yml'))['code'], 'v3', 'exec')
for prm in [dict(rate=6.9, delta=-2.0, decay=1.0, mid=1.0), dict(rate=10.0, delta=-2.0, decay=0.7, mid=0.5), dict(rate=4.0, delta=-3.0, decay=0.9, mid=1.0), dict(rate=15.0, delta=-1.0, decay=0.5, mid=0.0)]:
    out = []
    for t in range(24):
        x = dict(c=C[..., :t+1], valid=V[..., :t+1], p=P[..., :t], in_group=M[..., :t+1], t=th.tensor(float(t)))
        pu = run_rule(code, prm, x).clamp(0, 30).floor()
        out.append(float((pu * M[..., t]).sum() / M[..., t].sum()))
    print(prm, 'per head per round', round(np.mean(out), 2), 'rounds 0,1,4,5,8,16:', np.round([out[i] for i in [0, 1, 4, 5, 8, 16]], 2))
