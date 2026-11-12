"""v2: natural experiment in the v1 sweep. Before a point's first punishing round its group is run exactly
like `zero`, so at round t the groups whose first punishment falls at t (treated) and those whose first
punishment comes later (control) have exchangeable histories. Compare, member by member and by contribution
bin, the next-round contribution and (at decision rounds) leaving, and at group level the next round's
contribution sum and the members after the next decision."""
import numpy as np, json
D = np.load('/tmp/v1sweep.npz')
c, p, g, v, mg, name = D['c'], D['p'], D['g'], D['v'], D['mine_g'], D['name']
pts = {q['name']: q['params'] for q in json.load(open('plots/simulation/policy_finder/pf-20261112-6_sweep/sweep.json'))['points']}
mine = g == mg[:, None, None]
# first round in which the rule's group received any punishment (24 = never)
gp = (p * mine).sum(1)  # [N,24]
first = np.where((gp > 0).any(1), (gp > 0).argmax(1), 24)
# policy-implied first punishable round (independent of outcome): t_start..22 within window
def first_allowed(q):
    for t in range(int(q['t_start']), 23):
        if t >= 20 or t % 4 < q['w']:
            return t
    return 24
fa = np.array([first_allowed(pts[n]) if pts[n]['k'] * pts[n]['thr'] >= 1 else 24 for n in name])
bins = [(0, 0), (1, 2), (3, 5), (6, 9), (10, 20)]
print('t | treated n | member next-round dc treated vs control by c-bin (only members punished>0 in treated)')
res = []
for t in range(0, 23):
    tr = (fa == t); co = (fa > t)
    if tr.sum() < 500: continue
    m_t = mine[:, :, t] & (v[:, :, t] == 1) & (g[:, :, t + 1] == g[:, :, t]) if t not in (3, 7, 11, 15, 19) else mine[:, :, t] & (v[:, :, t] == 1)
    line = []
    for lo, hi in bins:
        cb = (c[:, :, t] >= lo) & (c[:, :, t] <= hi) & m_t
        T = cb & tr[:, None] & (p[:, :, t] > 0); C = cb & co[:, None]
        if T.sum() < 100: line.append('   .   '); continue
        dT = (c[:, :, t + 1] - c[:, :, t])[T].mean(); dC = (c[:, :, t + 1] - c[:, :, t])[C].mean()
        pm = p[:, :, t][T].mean()
        s = f'{dT - dC:+.2f}@{pm:.1f}'
        if t in (3, 7, 11, 15, 19):
            lT = (g[:, :, t + 1] != g[:, :, t])[T].mean(); lC = (g[:, :, t + 1] != g[:, :, t])[C].mean()
            s += f' L{lT - lC:+.2f}'
        line.append(s)
    # group level: members at the end (mean over rounds t+1..23) and pool from t onwards, treated vs control
    n_mem = mine.sum(1)  # [N,24]
    pool = (1.6 * c * mine).sum(1) - (p * mine).sum(1)
    ot = ~mine
    pool_o = (1.6 * c * ot).sum(1) - (p * ot).sum(1)
    def grp(sel):
        return (n_mem[sel][:, t + 1:].mean(), pool[sel][:, t:].sum(1).mean() / 24, pool_o[sel][:, t:].sum(1).mean() / 24)
    gT, gC = grp(tr), grp(co)
    print(f'{t:2d} {tr.sum():6d} ' + ' | '.join(line) + f' || mem {gT[0]-gC[0]:+.2f} own {gT[1]-gC[1]:+.2f} other {gT[2]-gC[2]:+.2f}')
