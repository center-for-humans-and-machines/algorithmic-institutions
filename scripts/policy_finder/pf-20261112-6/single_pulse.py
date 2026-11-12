"""v2: cleanest natural experiment in the v1 sweep: points with w=1 punish only in the first round of a
block, so a group whose first punishment is at block start t (t%4==0, t<20) gets one pulse and then nothing
until t+4. Control: groups with no punishment through t+3 (first allowed round > t+3). Matched on the
member's exact contribution at t; outcomes: contribution at t+1..t+3, leaving at the decision t+3; group:
members in the next block, pools of both groups over t..t+3."""
import numpy as np, json
D = np.load('/tmp/v1sweep.npz')
c, p, g, v, mg, name = D['c'], D['p'], D['g'], D['v'], D['mine_g'], D['name']
pts = {q['name']: q['params'] for q in json.load(open('plots/simulation/policy_finder/pf-20261112-6_sweep/sweep.json'))['points']}
mine = g == mg[:, None, None]
def first_allowed(q):
    if q['k'] * q['thr'] < 1: return 24
    for t in range(int(q['t_start']), 23):
        if t >= 20 or t % 4 < q['w']: return t
    return 24
fa = np.array([first_allowed(pts[n]) for n in name])
w = np.array([pts[n]['w'] for n in name])
# pool every group-round
own_pool = (1.6 * c * mine).sum(1) - (p * mine).sum(1)
oth_pool = (1.6 * c * ~mine).sum(1) - (p * ~mine).sum(1)
nm = mine.sum(1)
acc = {}
for t in (0, 4, 8, 12, 16):
    tr = (fa == t) & (w == 1); co = fa > t + 3
    m0 = mine[:, :, t] & (v[:, :, t] == 1)
    stay = (g[:, :, t + 1:t + 4] == g[:, :, t:t + 1]).all(-1)
    for cv in range(0, 21):
        cm = m0 & (c[:, :, t] == cv)
        for pb, (lo, hi) in enumerate([(1, 1), (2, 3), (4, 7), (8, 30)]):
            T = cm & tr[:, None] & (p[:, :, t] >= lo) & (p[:, :, t] <= hi); C = cm & co[:, None]
            if T.sum() < 20 or C.sum() < 20: continue
            for h in (1, 2, 3):
                TT, CC = T & stay, C & stay
                d = c[:, :, t + h][TT].mean() - c[:, :, t + h][CC].mean()
                acc.setdefault((pb, h), []).append((d, TT.sum()))
            L = (g[:, :, t + 4] != g[:, :, t + 3])[T].mean() - (g[:, :, t + 4] != g[:, :, t + 3])[C].mean()
            acc.setdefault((pb, 'L'), []).append((L, T.sum()))
            acc.setdefault((pb, 'cbin', min(cv // 5, 2)), []).append((c[:, :, t + 1:t + 4][T & stay].mean() - c[:, :, t + 1:t + 4][C & stay].mean(), (T & stay).sum()))
    gT = own_pool[tr][:, t:t + 4].mean() - own_pool[co][:, t:t + 4].mean()
    oT = oth_pool[tr][:, t:t + 4].mean() - oth_pool[co][:, t:t + 4].mean()
    mT = nm[tr][:, t + 4].mean() - nm[co][:, t + 4].mean()
    pT = (p * mine)[tr][:, :, t].sum(1).mean()
    print(f't={t}: treated {tr.sum()} groups, pulse {pT:.1f} pts/group; own pool/round t..t+3 {gT:+.2f}, '
          f'other {oT:+.2f}; members next block {mT:+.3f}')
lab = ['p=1', 'p=2-3', 'p=4-7', 'p=8+']
for k in sorted(acc, key=str):
    a = np.array(acc[k]); print(lab[k[0]], k[1:], f'{np.average(a[:,0], weights=a[:,1]):+.3f}', int(a[:, 1].sum()))
