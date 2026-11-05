"""v2: under v1's best rule, who keeps getting punished in rounds 4-11, and does it work?"""
import pandas as pd, numpy as np
f = pd.read_parquet('plots/simulation/policy_finder/pf-20261105-8_sweep/final/per_round.parquet')
f = f[f.run.str.endswith('_gp')]
f['player'] = f.participant_code.str.split('_').str[0].astype(int)
E = f.episode.nunique()
def arr(col):
    return f.pivot_table(index=['episode', 'player'], columns='round_number', values=col).values.reshape(E, 4, 24)
C, P, V = arr('contribution'), arr('punishment'), arr('contribution_valid').astype(bool)
# number of previous rounds punished, per player-round
npun = np.concatenate([np.zeros((E, 4, 1)), np.cumsum(P > 0, axis=2)[:, :, :-1]], axis=2)
rows = []
for t in range(1, 23):
    for i in range(4):
        oth = np.delete(C[:, :, t], i, axis=1).mean(1)
        rows.append(pd.DataFrame(dict(t=t, c=C[:, i, t], p=P[:, i, t], cprev=C[:, i, t-1], pprev=P[:, i, t-1],
                                      npun=npun[:, i, t], dnext=C[:, i, t+1] - C[:, i, t], oth=oth,
                                      v=V[:, i, t] & V[:, i, t+1])))
d = pd.concat(rows); d = d[d.v]
pun = d[(d.t >= 4) & (d.t <= 11) & (d.p > 0)]
print('punished in rounds 4-11: n', len(pun), 'mean p', pun.p.mean().round(2))
print(pun.groupby(pd.cut(pun.npun, [-1, 0, 1, 2, 4, 8, 30])).agg(n=('p', 'size'), p=('p', 'mean'), c=('c', 'mean'), dnext=('dnext', 'mean')).round(2))
print('punished share by previous number of punishments (rounds 4-11, c below 13.75)')
low = d[(d.t >= 4) & (d.t <= 11) & (d.c < 13.75)]
print(low.groupby(pd.cut(low.npun, [-1, 0, 1, 2, 4, 8, 30])).agg(n=('p', 'size'), c=('c', 'mean'), cprev=('cprev', 'mean'), dnext=('dnext', 'mean')).round(2))
# fraction of total punishment points by player 'type': total punished rounds in game
tot = (P > 0).sum(2).ravel(); pts = P.sum(2).ravel()
for lo, hi in [(0, 0), (1, 1), (2, 3), (4, 6), (7, 12)]:
    m = (tot >= lo) & (tot <= hi)
    print(f'players punished {lo}-{hi} rounds: share {m.mean():.2f}, share of points {pts[m].sum()/pts.sum():.2f}, mean c rounds 12-23 {C[:, :, 12:].mean(2).ravel()[m].mean():.2f}')
# response at round 0
r0 = d[d.t == 1]
print('round 1 change by p at round 1', r0.groupby(pd.cut(r0.p, [-1, 0, 5, 10, 20, 30])).dnext.mean().round(2).to_dict())
