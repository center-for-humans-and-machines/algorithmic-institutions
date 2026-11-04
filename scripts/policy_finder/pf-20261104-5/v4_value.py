"""v4: player-level value of a punishment in v3's sweep (part 1). For a member of our group
(group 0) at round t with contribution c_t, Y = 1.6 * sum of its contributions to our group
over rounds t+1..23 (0 for rounds it is elsewhere). Compare by dose at t, by contribution
class, by phase (t), and newcomer (joined in this block) vs incumbent; also stay rate at the
next decision. Net value per game = mean Y(dose) - mean Y(0) - dose."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'scripts/policy_finder/pf-20261104-5')
from v3_load import load
A = load('plots/simulation/policy_finder/pf-20261104-5-v3_sweep/per_round_p1of2.parquet')
c, p, g, v = A['c'], A['p'], A['g'] == 0, A['v']
G = c.shape[0]
contrib = 1.6 * c * g  # [G,24,8]
fut = np.cumsum(contrib[:, ::-1], 1)[:, ::-1]  # sum from t..23
rows = []
for t in range(0, 20):
    Y = fut[:, t + 1] if t < 23 else 0
    nxt_dec = 4 * (t // 4) + 3
    if nxt_dec == t: nxt_dec = t + 4
    stay = g[:, min(nxt_dec + 1, 23)]
    newc = (g[:, t] & ~g[:, 4 * (t // 4) - 1]) if t >= 4 else np.zeros_like(g[:, t])
    m = g[:, t] & v[:, t]
    rows.append(pd.DataFrame(dict(t=t, c=c[:, t][m], pun=p[:, t][m], Y=Y[m], stay=stay[m], new=newc[m],
                                  recv=p[:, :t].sum(1)[m])))
D = pd.concat(rows)
D['cb'] = pd.cut(D.c, [-1, 2, 5, 8, 11, 20], labels=['0-2', '3-5', '6-8', '9-11', '12+'])
D['db'] = pd.cut(D.pun, [-1, 0, 4, 9, 15, 22, 30], labels=['0', '1-4', '5-9', '10-15', '16-22', '23-30'])
D['ph'] = pd.cut(D.t, [-1, 0, 2, 3, 6, 7, 11, 19], labels=['r0', 'r1-2', 'r3', 'r4-6', 'r7', 'r8-11', 'r12-19'])
D['grp'] = np.where(D.t < 4, 'start', np.where(D.new, 'new', 'old'))
pd.set_option('display.width', 250)
for ph in D.ph.cat.categories:
    for grp in ['start', 'new', 'old']:
        S = D[(D.ph == ph) & (D.grp == grp)]
        if len(S) < 5000: continue
        T = S.groupby(['cb', 'db'], observed=True).agg(n=('Y', 'size'), Y=('Y', 'mean'), stay=('stay', 'mean'), pun=('pun', 'mean'))
        base = T.xs('0', level='db')
        T['net'] = T.Y - base.Y.reindex(T.index.get_level_values(0)).values - T.pun
        T['dstay'] = T.stay - base.stay.reindex(T.index.get_level_values(0)).values
        T = T[T.n >= 300]
        print(f'== {ph} {grp} (n={len(S)})')
        print(T[['n', 'Y', 'net', 'dstay']].unstack('db').round(1).to_string())
