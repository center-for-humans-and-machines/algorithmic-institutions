"""Hand check of v7: round 0 shocks only the K0 lowest contributors giving <= hi (ties by index), rounds 4 and 8
everyone giving <= hi. My group = players 0-4 giving 3, 0, 5, 0, 20 (and a 2nd game with only 6 alone in it);
the other group gives 0."""
import torch as th
th.ones(2).sum(dim=-1, keepdim=True); th.tensor(3.) <= 20
from aimanager.manager.rule import load_rule, run_rule, dry_run_inputs
dry_run_inputs()
for prm in [{'D0': 20.0, 'K0': 1, 'D4': 29.5, 'D8': 10.0, 'hi': 5},
            {'D0': 25.0, 'K0': 2, 'D4': 15.0, 'D8': 0.0, 'hi': 4},
            {'D0': 25.0, 'K0': 3, 'D4': 15.0, 'D8': 0.0, 'hi': 7}]:
    code, params = load_rule('configs/managers/rule_based/pf-20261112-5-v7.yml', prm)
    cnow = th.tensor([[3., 0, 5, 0, 20, 0, 0, 0], [3., 0, 5, 0, 20, 0, 0, 0]]); val = th.ones(2, 8)
    print(prm)
    for t in range(24):
        T = t + 1
        g = th.zeros(2, 8, T); g[0, :5, :] = 1; g[1, 6, :] = 1
        inp = dict(c=cnow.unsqueeze(-1).repeat(1, 1, T), valid=val.unsqueeze(-1).repeat(1, 1, T),
                   in_group=g, p=th.zeros(2, 8, t), t=th.tensor(float(t)))
        out = run_rule(code, params, inp)
        if out.sum() > 0: print('  ', t, [[round(x, 2) for x in row] for row in out.tolist()])
