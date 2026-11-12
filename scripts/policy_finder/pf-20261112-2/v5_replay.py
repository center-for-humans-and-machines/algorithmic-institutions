"""v5: replay the v5 rule (fresh process, real rule engine) on never-punish games (v4 sweep
index 3, both seats, 1000 games): in how many games and rounds would it fire first, for a grid
of (lead, gap, last_block), and does its race measure match a pandas computation."""
import pandas as pd, numpy as np, torch as th, glob, pyarrow.parquet as pq, sys
from aimanager.manager.rule import read_rule, run_rule
rule, code = read_rule('configs/managers/rule_based/pf-20261112-2-v5.yml')
v, idx = 'pf-20261112-2-v4', 3
ds = []
for f in sorted(glob.glob(f'plots/simulation/policy_finder/{v}_sweep/per_round*.parquet')):
    runs = [r for r in set(pq.read_table(f, columns=['run']).column('run').to_pylist()) if f'_s{idx:03d}' in r or r.endswith(f'_s{idx}')]
    if runs:
        ds.append(pq.read_table(f, filters=[('run', 'in', runs)]).to_pandas())
d = pd.concat(ds)
print(d.run.unique())
r = d.run.str.extract(r'by (\S+)_vs_(\S+)$')
d['seat'] = np.where(r[0].str.startswith('pf'), 0, 1)
d['pidx'] = d.groupby(['seat', 'episode']).participant_code.transform(lambda s: pd.factorize(s, sort=True)[0])
d = d.sort_values(['seat', 'episode', 'pidx', 'round_number'])
games = d[['seat', 'episode']].drop_duplicates().reset_index(drop=True)
n = len(games)
def arr(col):
    return th.tensor(d[col].astype(float).values.reshape(n, 8, 24), dtype=th.float32)
C, V, P, GID = arr('contribution'), arr('contribution_valid'), arr('punishment'), arr('group_id')
seat = th.tensor(games.seat.values, dtype=th.float32)[:, None, None]
ING = (GID == seat).float()
assert (P * ING).sum() == 0, 'my group was fined'
grid = [(L, S, lb) for L in [0, 5, 10, 15, 20] for S in [6, 8, 10, 12] for lb in [2, 6]]
first = {g: th.full((n,), 99.) for g in grid}
nfine = {g: th.zeros(n) for g in grid}
races = []
for t in range(24):
    inp = dict(c=C[..., :t + 1], valid=V[..., :t + 1], p=P[..., :t], in_group=ING[..., :t + 1], t=th.tensor(float(t)))
    for g in grid:
        out = run_rule(code, dict(lead=float(g[0]), gap=float(g[1]), fine=1.0, last_block=float(g[2])), inp)
        fired = (out.clamp(0, 30).floor() > 0).any(-1)
        first[g] = th.where(fired & (first[g] == 99), th.full_like(first[g], t), first[g])
        nfine[g] += (out.floor() > 0).float().sum(-1)
    # race via a big lead -> extract by bisection is awkward; recompute in torch the same way for the check
res = []
for g in grid:
    f = first[g]
    res.append(dict(lead=g[0], gap=g[1], last_block=g[2], fired=(f < 99).float().mean().item(), t_first_med=f[f < 99].median().item() if (f < 99).any() else np.nan, fines_per_game=nfine[g].mean().item()))
print(pd.DataFrame(res).round(3).to_string())
print('=== race check: rule race vs the race with the true current-round punishments of ah')
code2 = compile(rule['code'].replace('punishment = fine * target * ahead * window', 'punishment = race.unsqueeze(-1) + 0 * fine * gap * lead * last_block * c_now'), 'r', 'exec')
diffs = []
for t in [0, 2, 3, 5, 9, 14, 19]:
    inp = dict(c=C[..., :t + 1], valid=V[..., :t + 1], p=P[..., :t], in_group=ING[..., :t + 1], t=th.tensor(float(t)))
    race = run_rule(code2, dict(lead=1., gap=1., fine=1., last_block=1.), inp)[:, 0]
    b0 = 4 * (t // 4)
    me, ot = ING * V, (1 - ING) * V
    pcm = (1.6 * (C * me).sum(1) - (P * ING).sum(1)) / me.sum(1).clamp(min=1)
    pco = (1.6 * (C * ot).sum(1) - (P * (1 - ING)).sum(1)) / ot.sum(1).clamp(min=1)
    true = (pcm - pco)[:, b0:t + 1].mean(1)
    print(t, 'rule race mean %.2f, true %.2f, corr %.3f, mean |diff| %.2f' % (race.mean(), true.mean(), np.corrcoef(race, true)[0, 1], (race - true).abs().mean()))
