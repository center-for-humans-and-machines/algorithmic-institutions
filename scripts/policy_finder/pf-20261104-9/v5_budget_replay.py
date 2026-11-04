"""Static replay on v3's final run: how much of v3's actual punishment a per-round budget
n_mine x max(0, seen gap - margin) would allow, by round, for several margins; and the leave rate after
round 3 of members in episodes where the budget would have bound in round 0 vs not."""
import pandas as pd, numpy as np
fn = '/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261104-9-v3_sweep/final/per_round.parquet'
d = pd.read_parquet(fn, columns=['run', 'episode', 'round_number', 'group_id', 'participant_code', 'contribution', 'punishment', 'contribution_valid'])
d = d[d.run == [x for x in d.run.unique() if '_gp' in x][0]]
d['ci'] = d.contribution * d.contribution_valid; d['pi'] = d.punishment * d.contribution_valid
g = d.groupby(['episode', 'round_number', 'group_id']).agg(nv=('contribution_valid', 'sum'), c=('ci', 'sum'), p=('pi', 'sum')).reset_index()
w = g.pivot_table(index=['episode', 'round_number'], columns='group_id', values=['nv', 'c', 'p']).fillna(0)
w.columns = [f'{a}{b}' for a, b in w.columns]; w = w.reset_index()
w['pm1_last'] = (w.p1 / w.nv1.clip(lower=1)).groupby(w.episode).shift(1).fillna(0)
cgo = np.where(w.nv1 > 0, 1.6 * w.c1 / w.nv1.clip(lower=1) - w.pm1_last, 0)
w['gap'] = 1.6 * w.c0 / w.nv0.clip(lower=1) - cgo
for mg in (-15, -10, -6, -3, 0):
    bud = w.nv0 * np.maximum(0, w.gap - mg)
    allowed = np.minimum(w.p0, bud)
    w[f'k{mg}'] = allowed / w.p0.where(w.p0 > 0)
    r = w.groupby('round_number').apply(lambda x: np.minimum(x.p0, x.nv0 * np.maximum(0, x.gap - mg)).sum() / max(x.p0.sum(), 1e-9))
    print(f'margin {mg:4d}: share of v3 punishment kept by round', ' '.join(f'{v:.2f}' for v in r.values[:11]), '| overall', round(float(np.minimum(w.p0, bud).sum() / w.p0.sum()), 2))
# leave after round 3 by whether the round-0 budget at margin -6 binds
e0 = w[w.round_number == 0].set_index('episode')
mem = d[(d.round_number == 3) & (d.group_id == 0)][['episode', 'participant_code']]
nxt = d[d.round_number == 4][['episode', 'participant_code', 'group_id']]
mm = mem.merge(nxt, on=['episode', 'participant_code'])
mm['leave'] = mm.group_id != 0
mm['gap0'] = e0.loc[mm.episode, 'gap'].values
for lo, hi in ((-99, -4), (-4, 0), (0, 4), (4, 99)):
    s = (mm.gap0 >= lo) & (mm.gap0 < hi)
    print(f'round-0 seen gap {lo}..{hi}: share of episodes {(e0.gap.between(lo, hi - 1e-9)).mean():.2f}, my members leaving after round 3 {mm.leave[s].mean():.3f}')
