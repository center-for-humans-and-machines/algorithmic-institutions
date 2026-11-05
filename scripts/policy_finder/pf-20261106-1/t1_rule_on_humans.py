"""Run the rule on the human histories (group 0's view) to see what it would deal."""
import sys; sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-1')
sys.path.insert(0,'/opt/algorithmic-institutions/src')
from load import load
import numpy as np, torch as th
from aimanager.manager.rule import read_rule, run_rule
df = load()
games = sorted(df.game.unique())
def arr(col):
    return th.tensor(np.stack([df[df.game==g].pivot(index='player_id',columns='round_number',values=col).values for g in games]),dtype=th.float)
_ = th.rand(2) > 3.0; th.clamp(th.rand(2), min=0.0)  # warm torch's lazy scalar import outside the builtin-free scope
C, P, G, V = arr('contribution'), arr('punishment'), arr('group_id'), 1-arr('player_no_input')
rule = read_rule('/opt/algorithmic-institutions/configs/managers/rule_based/pf-20261106-1.yml', 'exp2_multi')
import yaml; code = compile(yaml.safe_load(open('/opt/algorithmic-institutions/configs/managers/rule_based/pf-20261106-1.yml'))['code'], 'rule', 'exec')
for params in [dict(thresh=10,slope=0.5,dec_mult=0.5,margin=1.0), dict(thresh=10,slope=0.5,dec_mult=0.5,margin=6.0), dict(thresh=6,slope=0.2,dec_mult=1,margin=1)]:
    tot=[]; byc={}
    for t in range(24):
        inp=dict(c=C[...,:t+1], valid=V[...,:t+1], p=P[...,:t], in_group=(G[...,:t+1]==0).float(), t=th.tensor(float(t)))
        out = run_rule(code, params, inp).clamp(0,30).floor()
        m = (G[...,t]==0)
        tot.append(out[m].mean().item())
    print(params, 'mean per member by round', np.round(tot,2), 'overall', np.mean(tot).round(2), 'humans', P[G==0].mean().item())
