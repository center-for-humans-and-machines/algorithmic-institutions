"""Hand check of v5 timing, targeting and the newcomer weight (my group = players 0-3 now; player 1 joined at round 4
and player 2 at round 8, both from the other group; other group 4-7)."""
import torch as th
th.clamp(th.ones(2), min=1.0); th.ones(2).sum(dim=-1, keepdim=True); th.remainder(th.tensor(3.), 4.0); th.tensor(3.) < 20; th.cat([th.ones(1), th.ones(1)])
from aimanager.manager.rule import load_rule, run_rule, dry_run_inputs
dry_run_inputs()
for prm in [{'D0': 0.0, 'D4': 24.5, 'D8': 10.0, 's': 0.5, 'hi': 7},
            {'D0': 12.0, 'D4': 30.0, 'D8': 0.0, 's': 0.0, 'hi': 3}]:
    code, params = load_rule('configs/managers/rule_based/pf-20261112-5-v5.yml', prm)
    cnow = th.tensor([[0., 7, 3, 20, 0, 5, 5, 5]]); val = th.ones(1, 8)
    print(prm)
    for t in range(24):
        T = t + 1
        g = th.zeros(1, 8, T); g[0, 0, :] = 1; g[0, 3, :] = 1; g[0, 1, 4:] = 1; g[0, 2, 8:] = 1
        inp = dict(c=cnow.unsqueeze(-1).repeat(1, 1, T), valid=val.unsqueeze(-1).repeat(1, 1, T),
                   in_group=g, p=th.zeros(1, 8, t), t=th.tensor(float(t)))
        out = run_rule(code, params, inp)
        if out.sum() > 0: print('  ', t, [round(x, 2) for x in out[0].tolist()])
