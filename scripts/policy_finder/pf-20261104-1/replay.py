"""Open-loop replay of the rule on recorded games: per member-round punishment it
would deal at several param sets, and its correlation with what was dealt.
Usage: PYTHONPATH=src python3 replay.py"""
import pandas as pd, numpy as np, torch as th
from aimanager.manager.rule import load_rule, rule_inputs, run_rule
RULE = 'configs/managers/rule_based/pf-20261104-1.yml'
def tensors(d, game, pid, gcol, vcol):
    d = d.sort_values([game, pid, 'round_number'])
    games = d[game].unique(); B = len(games)
    arr = lambda col: th.tensor(d[col].astype(float).values.reshape(B, 8, 24), dtype=th.float)
    return arr('contribution'), arr(vcol), arr('punishment'), arr(gcol).long()
h = pd.read_csv('experiments/2group_8agent_50ep.csv').drop_duplicates(['session','episode','round_number','participant_code'])
h['game'] = h.session.astype(str)+'_'+h.episode.astype(str); h['v'] = 1-h.player_no_input
s = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
s = s[s.run.str.contains('ah_vs_ah')]
sets = {'zeroish': dict(slope=0.1,tol=3,cap=5,t_stop=23,pre_switch=1),
        'light': dict(slope=0.3,tol=2,cap=4,t_stop=15,pre_switch=0.3),
        'mid': dict(slope=0.7,tol=1,cap=8,t_stop=19,pre_switch=0.5),
        'harsh': dict(slope=2,tol=0,cap=15,t_stop=23,pre_switch=1)}
for name, (C,V,P,G) in [('human', tensors(h,'game','player_id','group_id','v')), ('sim ah_vs_ah', tensors(s,'episode','participant_code','agent_group','contribution_valid'))]:
    print('==', name, 'actual punishment per member-round', P.mean().item().__round__(2))
    for k, prm in sets.items():
        code, params = load_rule(RULE, prm)
        out = []
        for t in range(24):
            o = th.zeros(C.shape[:2])
            for g in (0,1):
                inp = rule_inputs(C[...,:t+1], V[...,:t+1], P[...,:t], G[...,:t+1], g)
                r = run_rule(code, params, inp).clamp(0,30).floor()
                o = th.where(G[...,t]==g, r, o)
            out.append(o)
        R = th.stack(out, -1)
        print(f'  {k:8s} mean {R.mean():.2f}  share punished {(R>0).float().mean():.2f}  corr with actual {np.corrcoef(R.flatten(), P.flatten())[0,1]:.2f}')
