"""Hand check of the rule's timing and targeting on one made-up game state."""
import torch as th
th.clamp(th.ones(2), min=1.0); th.ones(2).sum(dim=-1, keepdim=True); th.remainder(th.tensor(3.), 4.0); th.tensor(3.) < 20
from aimanager.manager.rule import load_rule, run_rule, dry_run_inputs
dry_run_inputs()  # warms torch lazy imports, as validate-rule and the sim do
code, params = load_rule('configs/managers/rule_based/pf-20261112-5.yml', {'k': 1.0, 'thr': 1.0, 'w': 2, 'end': 0.5})
cnow = th.tensor([[0., 10, 10, 20, 5, 5, 5, 5]])
grp = th.tensor([[1., 1, 1, 1, 0, 0, 0, 0]])
val = th.tensor([[1., 1, 1, 1, 1, 1, 1, 1]])
for t in [0, 1, 2, 3, 4, 19, 20, 22, 23]:
    T = t + 1
    inp = dict(c=cnow.unsqueeze(-1).repeat(1, 1, T), valid=val.unsqueeze(-1).repeat(1, 1, T),
               in_group=grp.unsqueeze(-1).repeat(1, 1, T), p=th.zeros(1, 8, t), t=th.tensor(float(t)))
    print(t, run_rule(code, params, inp).tolist())
