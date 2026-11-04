"""v3 version of replay.py: Replay the rule on the human games (inputs from the data, punishments as dealt by humans)
and report what it would punish, by contribution bin and round, vs the human managers."""
import sys, pandas as pd, numpy as np, torch as th
from aimanager.manager.rule import load_rule, rule_inputs, run_rule
params = dict(rel=0.97, slope=2.8, cap=21.0, fresh=3.0, last_round=6, newcomer=0.7)
for a in sys.argv[1:]:
    k, v = a.split('='); params[k] = float(v)
code, params = load_rule('configs/managers/rule_based/pf-20261104-9-v3.yml', params)
d = pd.read_csv('experiments/2group_8agent_50ep.csv')
d = d.drop_duplicates(['session','episode','round_number','participant_code'])
d['game'] = d.session.astype(str)+'_'+d.episode.astype(str)
games = sorted(d.game.unique())
def arr(col):
    return th.tensor(np.stack([d[d.game==g].pivot(index='player_id', columns='round_number', values=col).values for g in games]), dtype=th.float)
C, V, P, G = arr('contribution'), 1-arr('player_no_input'), arr('punishment'), arr('group_id')
out = th.zeros_like(C)
# warm torch's lazy imports once with builtins (a fresh process trips on them without)
_inp = rule_inputs(C[..., :2], V[..., :2].bool(), P[..., :1], G[..., :2].long(), 0)
exec(code, {'th': th, **{k: th.tensor(float(v)) for k, v in params.items()}, **_inp})
for t in range(24):
    for grp in (0,1):
        inp = rule_inputs(C[..., :t+1], V[..., :t+1].bool(), P[..., :t], G[..., :t+1].long(), grp)
        raw = run_rule(code, params, inp).clamp(0,30).floor()
        out[..., t] = th.where(G[..., t]==grp, raw, out[..., t])
out = out*V
r = pd.DataFrame({'c':C.flatten(), 'rule':out.flatten(), 'human':P.flatten(), 'v':V.flatten(),
                  't':th.arange(24).repeat(C.shape[0]*C.shape[1])})
r = r[r.v==1]
r['cb'] = pd.cut(r.c,[-1,0,4,8,12,16,20])
print(params)
print(r.groupby('cb')[['rule','human']].mean().round(2))
print('share punished rule %.2f human %.2f' % ((r.rule>0).mean(), (r.human>0).mean()))
print('mean per member-round rule %.2f human %.2f' % (r.rule.mean(), r.human.mean()))
print(r.groupby(r.t//4)[['rule','human']].mean().round(2).T)
