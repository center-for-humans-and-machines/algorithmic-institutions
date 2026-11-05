# Sanity check of v3's rule on hand-made histories (fresh process: catches the import trap).
# Player 0: c=0 always (punished every round -> given up after N strikes); player 1: c=10;
# player 2: c=20 (never punished); player 3: c=12 then 20 from round 3 (two strikes, then clean).
import torch as th
from aimanager.manager.rule import run_rule, load_rule
params = dict(T=13.0, A=1.7, B=2.0, S=11, N=5)
code, prm = load_rule('configs/managers/rule_based/pf-20261105-11-v3.yml', params, 'exp1')
c = th.tensor([[0., 10, 20, 12]]).unsqueeze(-1).repeat(1, 1, 24)
c[0, 3, 3:] = 20
P = th.zeros(1, 4, 24)
for t in range(24):
    cc = c[..., :t + 1]
    inp = dict(c=cc, valid=th.ones_like(cc), p=P[..., :t], t=th.tensor(float(t)))
    out = run_rule(code, prm, inp)
    P[..., t] = th.floor(th.clamp(out, 0, 30))
    if t in (0, 1, 4, 5, 6, 11, 12, 23):
        print(t, out.tolist())
