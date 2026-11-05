"""Run the rule's actual code (run_rule) on every sweep design point on the surrogate."""
import sys, json, numpy as np, torch as th
sys.path.insert(0, 'scripts/policy_finder/pf-20261105-10')
from simsur import simulate
from aimanager.manager.rule import read_rule, sobol_design, run_rule
path = sys.argv[1]
rule, code = read_rule(path, 'exp1')
design = sobol_design(rule, 256)
out = []
for i, params in enumerate(design):
    def pol(c, p, t, params=params):
        C = th.tensor(c, dtype=th.float)
        inp = {'c': C, 'valid': th.ones_like(C), 'p': th.tensor(p, dtype=th.float), 't': th.tensor(float(t))}
        return run_rule(code, params, inp).numpy()
    s, se, cm, pm = simulate(pol, G=300, seed=1)
    out.append({'params': params, 'score': s, 'se': se, 'ptot': float(pm.sum())})
    print(i, {k: round(v, 2) for k, v in params.items()}, round(s, 1), flush=True)
json.dump(out, open('/tmp/pf10_design.json', 'w'))
