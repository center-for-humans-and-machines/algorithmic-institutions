import torch as th, yaml
from aimanager.manager.rule import load_rule, run_rule
th.ones(2).sum(-1, keepdim=True)  # warm up torch arg parsing (a fresh process imports once, which the no-builtins scope blocks)
cfg = 'configs/managers/rule_based/pf-20261106-9.yml'
params = dict(k=0.5, s0=2.0, quiet=1, t_end=22, w=0.5)
code, P = load_rule(cfg, params, 'exp2_multi')
def run(t, other_p):
    c = th.tensor([[0,5,10,15,20,20,20,20]], dtype=th.float).unsqueeze(-1).repeat(1,1,t+1)
    valid = th.ones_like(c)
    ing = th.tensor([[1,1,1,1,0,0,0,0]], dtype=th.float).unsqueeze(-1).repeat(1,1,t+1)
    p = th.zeros(1,8,t); p[:, 4:] = other_p
    return run_rule(code, P, dict(c=c, valid=valid, p=p, in_group=ing, t=th.tensor(float(t))))
for t, op in [(0,0),(1,0),(3,0),(5,4),(20,0),(23,0)]:
    print(t, op, run(t, op).tolist())
