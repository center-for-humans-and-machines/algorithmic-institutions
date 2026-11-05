"""v5: natural experiment in the v4 sweep (S drawn independently of T, A, B).
Groups that 'graduated' (every valid player >= T in some round <= r0) before any S matters:
do the fines of rounds r0+1..S still pay for them? Compare the payoff of rounds r0+1..23 across S."""
import json, numpy as np, sys
tag = sys.argv[1] if len(sys.argv) > 1 else 'v4'
ver = {'v4': 'pf-20261105-11-v4', 'v3': 'pf-20261105-11-v3', 'v2': 'pf-20261105-11-v2'}[tag]
z = np.load(f'/tmp/pf5/{tag}.npz', allow_pickle=True)
c, p, v = z['c'].astype(float), z['p'].astype(float), z['v'].astype(bool)
run = z['run']
pts = {f"{ver}_s{i:03d}": q['params'] for i, q in enumerate(json.load(open(f'plots/simulation/policy_finder/{ver}_sweep/sweep.json'))['points'])}
names = [q for q in json.load(open(f'plots/simulation/policy_finder/{ver}_sweep/sweep.json'))['points']]
key = np.array([r.split(' managed by ')[-1] for r in run])
print('example run', run[0], key[0], list(pts)[:2])
par = {k: np.array([pts[x][k] if x in pts else np.nan for x in key]) for k in ['T', 'A', 'B', 'S']}
if tag == 'v3':
    Nn = np.array([pts[x]['N'] for x in key])
n = v.sum(1)
cv = np.where(v, c, 0); pv = np.where(v, p, 0)
pay = (20 - cv - pv + (1.6 * cv.sum(1, keepdims=True) - pv.sum(1, keepdims=True)) / np.maximum(n, 1)[:, None, :])
rpay = np.where(v, pay, 0).sum(1) / np.maximum(n, 1)  # [E, 24]
print('mean total', rpay.sum(1).mean().round(1))
T = par['T'][:, None, None]
comply = (~v | (c >= T)).all(1)  # [E, 24] everyone valid at/above T
sel = (par['T'] > 12) & (par['T'] < 15) & (par['B'] > 1.3) & (par['B'] < 2.6) & (par['A'] < 4.5)
if tag == 'v3': sel &= Nn >= 8
for r0 in [2, 3, 4]:
    grad = comply[:, :r0 + 1].any(1)
    print(f'--- graduated by round {r0}: share {grad[sel].mean():.2f}')
    for g in [True, False]:
        row = []
        for S in range(r0 + 1, 13):
            m = sel & (grad == g) & (par['S'] == S)
            if m.sum() < 300: continue
            row.append(f'S{S}:{rpay[m, r0 + 1:].sum(1).mean():6.1f}±{rpay[m, r0+1:].sum(1).std() / np.sqrt(m.sum()):.1f}(bill {2*pv[m][:, :, r0+1:].sum((1, 2)).mean() / 4:4.1f},n{m.sum()})')
        print('grad' if g else 'not ', ' '.join(row))
