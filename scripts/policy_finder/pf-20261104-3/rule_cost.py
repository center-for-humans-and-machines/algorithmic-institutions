"""Apply the rule to zero-managed histories (zero_vs_ah, group 0) and report what it
would spend: mean punishment per group-round, share of members punished, by c bin."""
import sys, pandas as pd, numpy as np, torch as th, yaml
from aimanager.manager.rule import load_rule, rule_inputs, run_rule
_w = th.rand(2, 3); _w.sum(-1, keepdim=True); th.clamp(_w, min=1.0); th.clamp(_w, max=th.tensor(1.0))  # torch lazy-import warm-up
rule_path = 'configs/managers/rule_based/pf-20261104-3.yml'
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d[d.run.str.contains('zero_vs_ah') & (d.episode < 300)]
d = d.sort_values(['episode', 'participant_code', 'round_number'])
pids = sorted(d.participant_code.unique(), key=lambda s: int(s.split('_')[-1]))
E = d.episode.nunique()
def arr(col):
    return th.tensor(d.pivot_table(index=['episode', 'participant_code'], columns='round_number', values=col).values.reshape(E, 8, 24))
C, V, G, P = arr('contribution').float(), arr('contribution_valid').float(), arr('agent_group'), arr('punishment').float()
settings = [dict(ref_frac=0.6, k=0.3, pmax=3, t_stop=21), dict(ref_frac=0.8, k=0.5, pmax=5, t_stop=21),
            dict(ref_frac=1.0, k=1.0, pmax=10, t_stop=23), dict(ref_frac=0.5, k=0.15, pmax=2, t_stop=20)]
for s in settings:
    code, params = load_rule(rule_path, s)
    tot, npun, nmem, by = 0, 0, 0, []
    for t in range(24):
        inp = rule_inputs(C[..., :t+1], V[..., :t+1], P[..., :t], G[..., :t+1], 0)
        pun = th.floor(run_rule(code, params, inp).clamp(0, 30)) * inp['in_group'][..., -1] * inp['valid'][..., -1]
        tot += pun.sum().item(); npun += (pun > 0).sum().item(); nmem += inp['in_group'][..., -1].sum().item()
        by.append(pd.DataFrame({'c': C[..., t].flatten().numpy(), 'p': pun.flatten().numpy(), 'm': inp['in_group'][..., -1].flatten().numpy()}))
    b = pd.concat(by); b = b[b.m == 1]
    print(s, 'pun/group-round %.2f, share punished %.2f' % (tot / (E * 24), npun / nmem))
    print('   mean p by c bin:', b.groupby(pd.cut(b.c, [-1, 0, 4, 9, 14, 20]), observed=True).p.mean().round(2).to_dict())
