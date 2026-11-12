"""Hand check of v6 timing and the lo..hi band (my group = players 0-3: contributions 0, 2, 3, 20 every round; the
other group 4-7 gives 0, 5, 5, 5)."""
import torch as th
th.ones(2).sum(dim=-1, keepdim=True); th.tensor(3.) <= 20  # warm torch lazy imports (no builtins in the rule scope)
from aimanager.manager.rule import load_rule, run_rule, dry_run_inputs
dry_run_inputs()
for prm in [{'D0': 20.0, 'D4': 29.5, 'D8': 10.0, 'lo': 3, 'hi': 5},
            {'D0': 0.0, 'D4': 15.0, 'D8': 0.0, 'lo': 0, 'hi': 4}]:
    code, params = load_rule('configs/managers/rule_based/pf-20261112-5-v6.yml', prm)
    cnow = th.tensor([[0., 2, 3, 20, 0, 5, 5, 5]]); val = th.ones(1, 8)
    print(prm)
    for t in range(24):
        T = t + 1
        g = th.zeros(1, 8, T); g[0, :4, :] = 1
        inp = dict(c=cnow.unsqueeze(-1).repeat(1, 1, T), valid=val.unsqueeze(-1).repeat(1, 1, T),
                   in_group=g, p=th.zeros(1, 8, t), t=th.tensor(float(t)))
        out = run_rule(code, params, inp)
        if out.sum() > 0: print('  ', t, [round(x, 2) for x in out[0].tolist()])
