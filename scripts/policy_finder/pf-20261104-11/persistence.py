"""Does a punishment's lift persist? OLS of c(t+k)-c(t) on p(t), with controls, for stayers.
ah_vs_ah sims, low/mid givers (c<=10)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d[d.run.str.endswith('ah_vs_ah')].sort_values(['episode', 'participant_code', 'round_number'])
d['gm'] = d.groupby(['episode', 'round_number', 'group_id']).contribution.transform('mean')
g = d.groupby(['episode', 'participant_code'])
d['c_prev'] = g.contribution.shift(1)
for k in [1, 2, 3, 4, 6]:
    d[f'c{k}'] = g.contribution.shift(-k)
    d[f'g{k}'] = g.group_id.shift(-k)
base = d[(d.contribution_valid == 1) & d.c_prev.notna() & (d.round_number <= 16)]
for lo, hi in [(0, 5), (6, 10), (11, 20)]:
    b = base[(base.contribution >= lo) & (base.contribution <= hi)]
    out = []
    for k in [1, 2, 3, 4, 6]:
        x = b[(b[f'g{k}'] == b.group_id)]
        y = x[f'c{k}'] - x.contribution
        X = np.column_stack([np.ones(len(x)), x.punishment, x.contribution, x.c_prev, x.gm, x.round_number])
        beta = np.linalg.lstsq(X, y.values, rcond=None)[0]
        out.append(f'k={k}: {beta[1]:+.3f}')
    print(f'c {lo}-{hi}:', ', '.join(out))

# nonlinearity: dummies by punishment bin, c 0-10, k=4
b = base[(base.contribution <= 10)]
x = b[b.g4 == b.group_id]
bins = [0, 2, 4, 7, 12, 31]
D = [((x.punishment > bins[i]) & (x.punishment <= bins[i + 1])).astype(float) for i in range(len(bins) - 1)]
X = np.column_stack([np.ones(len(x))] + D + [x.contribution, x.c_prev, x.gm, x.round_number])
beta = np.linalg.lstsq(X, (x.c4 - x.contribution).values, rcond=None)[0]
print('k=4 lift by bin', [f'{bins[i]+1}-{bins[i+1]}: {beta[1+i]:+.2f}' for i in range(len(bins) - 1)])

# spillover: unpunished players, change at t+2 vs punishment others in group got at t
d['gp_sum'] = d.groupby(['episode', 'round_number', 'group_id']).punishment.transform('sum')
d['others_p'] = d.gp_sum - d.punishment
b = d[(d.contribution_valid == 1) & d.c_prev.notna() & (d.round_number <= 16) & (d.punishment == 0)]
x = b[b.g2 == b.group_id]
X = np.column_stack([np.ones(len(x)), x.others_p, x.contribution, x.c_prev, x.gm, x.round_number])
beta = np.linalg.lstsq(X, (x.c2 - x.contribution).values, rcond=None)[0]
print('spillover per point of others punishment, k=2:', round(beta[1], 4))
for lo, hi in [(0, 10), (11, 20)]:
    y = x[(x.contribution >= lo) & (x.contribution <= hi)]
    X = np.column_stack([np.ones(len(y)), y.others_p, y.contribution, y.c_prev, y.gm, y.round_number])
    print(lo, hi, round(np.linalg.lstsq(X, (y.c2 - y.contribution).values, rcond=None)[0][1], 4))
