"""Hand check of v4 timing and targeting on one made-up state (my group = players 0-3, other group 4-7)."""
import torch as th
th.clamp(th.ones(2), min=1.0); th.ones(2).sum(dim=-1, keepdim=True); th.remainder(th.tensor(3.), 4.0); th.tensor(3.) < 20; th.floor(th.tensor(3.))
from aimanager.manager.rule import load_rule, run_rule, dry_run_inputs
dry_run_inputs()
for prm in [{'D0': 0.0, 'D4': 24.5, 'D8': 10.0, 'D12': 30.0, 'hi': 7},
            {'D0': 12.0, 'D4': 0.0, 'D8': 0.0, 'D12': 0.0, 'hi': 2}]:
    code, params = load_rule('configs/managers/rule_based/pf-20261112-5-v4.yml', prm)
    cnow = th.tensor([[0., 7, 8, 20, 0, 5, 5, 5]]); grp = th.tensor([[1., 1, 1, 1, 0, 0, 0, 0]]); val = th.ones(1, 8)
    print(prm)
    for t in range(24):
        T = t + 1
        inp = dict(c=cnow.unsqueeze(-1).repeat(1, 1, T), valid=val.unsqueeze(-1).repeat(1, 1, T),
                   in_group=grp.unsqueeze(-1).repeat(1, 1, T), p=th.zeros(1, 8, t), t=th.tensor(float(t)))
        out = run_rule(code, params, inp)
        if out.sum() > 0: print('  ', t, out.tolist())
