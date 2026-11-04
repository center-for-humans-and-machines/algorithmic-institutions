"""v5: per-round profile of v4's final runs (group 0 = ours): members, contribution per
member, punishment per member, pool; plus the ah-vs-zero reference for comparison."""
import sys, numpy as np
sys.path.insert(0, 'scripts/policy_finder/pf-20261104-5')
from v3_load import load
def prof(A, m, g):
    gg = A['g'][m] == g
    cc, pp = A['c'][m], A['p'][m]
    n = gg.sum(2)
    pool = np.where(n > 0, (1.6 * cc * gg).sum(2) - (pp * gg).sum(2), 0)
    cm = (cc * gg).sum(2).sum(0) / np.maximum(gg.sum(2).sum(0), 1)
    pm = (pp * gg).sum(2).sum(0) / np.maximum(gg.sum(2).sum(0), 1)
    return pool.mean(), n.mean(0), cm, pm, pool.mean(0)
A = load('plots/simulation/policy_finder/pf-20261104-5-v4_sweep/final/per_round.parquet')
for run in np.unique(A['run']):
    m = A['run'] == run
    for g in [0, 1]:
        s, n, cm, pm, pl = prof(A, m, g)
        print(run.split('by ')[1], 'g', g, 'score %.1f' % s)
        print('  n ', np.round(n, 2)); print('  cm', np.round(cm, 1)); print('  pm', np.round(pm, 1)); print('  pl', np.round(pl, 0))
R = load('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
for run in np.unique(R['run']):
    m = R['run'] == run
    for g in [0, 1]:
        s, n, cm, pm, pl = prof(R, m, g)
        print(run, 'g', g, 'score %.1f' % s, 'games', m.sum())
        print('  n ', np.round(n, 2)); print('  cm', np.round(cm, 1)); print('  pm', np.round(pm, 1))
