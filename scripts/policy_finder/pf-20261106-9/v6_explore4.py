# v6: exogenous newcomer punishment. In v5's sweep the slope newcomers face, k1*(1+w*q), varies by
# design point. Joiners at round 4: their contribution in rounds 8-11 (if they stayed) and whether
# they stay, by bin of their arrival contribution relative to the incumbents and by tercile of the
# point's slope (k1*(1+w*q), q the other's mean punishment per member so far); also the incumbents'
# rounds 8-11 contribution (spillover) and the per-game pool in rounds 4-11.
import numpy as np, json
d = np.load('/tmp/pf-20261106-9-v5_sweep.npz')
c, p, g, ours, opp, pt = d['c'].astype(float), d['p'].astype(float), d['g'], d['ours'], d['opp'], d['pt']
s = json.load(open('plots/simulation/policy_finder/pf-20261106-9-v5_sweep/sweep.json'))
K1 = np.array([q['params']['k1'] for q in s['points']]); W = np.array([q['params']['w'] for q in s['points']])
mine = g == ours[:, None, None]
oth = ~mine
q = (p[:, :, :4] * oth[:, :, :4]).sum((1, 2)) / oth[:, :, :4].sum((1, 2))
slope = K1[pt] * (1 + W[pt] * q)
j = mine[:, :, 4] & ~mine[:, :, 3]
inc = mine[:, :, 4] & mine[:, :, 3]
incmean4 = (c[:, :, 4] * inc).sum(1, keepdims=True) / np.maximum(inc.sum(1, keepdims=True), 1)
sh = incmean4 - c[:, :, 4]
st = mine[:, :, 4:12].all(-1)
late = c[:, :, 8:12].mean(-1)
tb = np.quantile(slope, [1/3, 2/3])
ter = np.digitize(slope, tb)
hasj = j.any(1)
pool = (1.6 * c * mine - p * mine).sum(1)
print('slope terciles', tb)
for o in [1, 2, 3, 4, 0]:
    om = (opp == o) if o else np.ones_like(opp, bool)
    print(f'== opp {o or "all"}')
    for lo, hi in [(-20, 0), (0, 3), (3, 20)]:
        row = []
        for tt in range(3):
            m = j & om[:, None] & (ter == tt)[:, None] & (sh >= lo) & (sh < hi)
            row.append(f'{late[m & st].mean():5.2f}/{mine[:, :, 8][m].mean():.2f}')
        print(f'  joiner short [{lo},{hi}): late/stay8 by tercile  ' + '  '.join(row))
    row = []
    for tt in range(3):
        m = om & (ter == tt) & hasj
        I = inc & m[:, None]
        row.append(f'inc late {late[I & st].mean():5.2f} pool4-11 {pool[m][:, 4:12].mean():5.1f} pool12-23 {pool[m][:, 12:].mean():5.1f}')
    print('   ' + ' | '.join(row))
