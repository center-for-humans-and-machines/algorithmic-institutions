"""v7: per-episode scores of every earlier final point; paired differences where two
points share seeds (identical round-0 contributions)."""
import pandas as pd, numpy as np, glob
P = '/opt/algorithmic-institutions/plots/simulation/policy_finder/'
eps = {}
r0 = {}
for f in sorted(glob.glob(P + '*/final/per_round.parquet')):
    d = pd.read_parquet(f)
    d = d[d.contribution_valid.astype(bool)]
    for run, s in d.groupby('run'):
        name = run.split('managed by ')[1]
        sc = s.groupby(['episode', 'round_number']).payoff.mean().groupby('episode').sum()
        eps[name] = sc
        r0[name] = tuple(s[s.round_number == 0].sort_values(['episode', 'participant_code']).contribution.values[:40])
groups = {}
for k, v in r0.items():
    groups.setdefault(v, []).append(k)
for g in groups.values():
    print('seed group:', g)
    if len(g) > 1:
        base = max(g, key=lambda k: eps[k].mean())
        for k in g:
            dd = eps[k] - eps[base]
            print(f'  {k:28s} {eps[k].mean():7.2f}  diff vs {base}: {dd.mean():6.2f} se {dd.std() / np.sqrt(len(dd)):.2f}')
