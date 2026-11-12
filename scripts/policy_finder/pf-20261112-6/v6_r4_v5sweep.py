"""v6: round-4 natural experiment on v5's sweep (every point punishes only at round 4: floor(amt) to members with
lo <= c4 <= hi). Given c4 the dose is random over the design, so E[dose | c4] is a 21-value table. Treatment per
target class: dose and dose^2/10 minus their design expectation. Outcomes: rest-of-game lead (rounds 4-23, /24),
own members at rounds 8/12/20, own group's contributions per round 5-23."""
import numpy as np, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from load_sweep import load
b = 4
ver = 'pf-20261112-6-v5'
base = f'plots/simulation/policy_finder/{ver}_sweep/'
f = '/tmp/v5sweep.npz'
if not os.path.exists(f): np.savez(f, **load(base))
D = np.load(f)
c, p, g, v, mg, name = D['c'], D['p'], D['g'], D['v'], D['mine_g'], D['name']
pts = {q['name']: q['params'] for q in json.load(open(base + 'sweep.json'))['points']}
keys = np.array(sorted(pts)); PAR = np.array([[pts[k][x] for x in ['lo', 'hi', 'amt']] for k in keys])
pid = np.searchsorted(keys, name)
mine = g == mg[:, None, None]; size = mine.sum(1)
here = mine[:, :, b] & (v[:, :, b] > 0)
cv = c[:, :, b].astype(int)
P = PAR[pid]
dose = np.where((cv >= P[:, 0:1]) & (cv <= P[:, 1:2]), np.floor(P[:, 2:3]), 0.) * here
act = p[:, :, b] * mine[:, :, b]
print('games', len(c), 'dose check', np.mean(np.abs(dose - act).max(1) < 1e-6),
      'punish elsewhere', (p * mine)[:, :, np.r_[0:4, 5:24]].sum())
vals = np.arange(21)
E1 = np.array([np.mean(np.where((x >= PAR[:, 0]) & (x <= PAR[:, 1]), np.floor(PAR[:, 2]), 0)) for x in vals])
E2 = np.array([np.mean(np.where((x >= PAR[:, 0]) & (x <= PAR[:, 1]), np.floor(PAR[:, 2]) ** 2 / 10, 0)) for x in vals])
print('E[dose|c]', dict(zip(vals[:9], E1[:9].round(2))))
op = (1.6 * c * mine - p * mine).sum(1); oo = (1.6 * c * ~mine - p * ~mine).sum(1)
Y = dict(lead=(op[:, b:] - oo[:, b:]).sum(1) / 24, s8=size[:, 8].astype(float), s12=size[:, 12].astype(float),
         s20=size[:, 20].astype(float), ownc=(c * mine)[:, :, 5:].sum(1).mean(1))
seat = mg.astype(float)

def fit(classes, sel=None, title='', doses=(8, 13, 18, 23, 28), extra=None):
    s = np.ones(len(c), bool) if sel is None else sel
    R, E = [], []
    for lab, cm in classes:
        cm = cm & here
        a1 = (act * cm).sum(1); e1 = (E1[cv] * here * cm).sum(1)
        a2 = (act ** 2 / 10 * cm).sum(1); e2 = (E2[cv] * here * cm).sum(1)
        R += [a1 - e1, a2 - e2]; E += [e1, e2]
    X = np.column_stack([np.ones(len(c))] + R + E + [seat, size[:, b]] + ([] if extra is None else extra))[s]
    print('==', title, 'games', s.sum())
    for k, y in Y.items():
        y = y[s]; bb, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ bb
        V = np.linalg.pinv(X.T @ X) * (e ** 2).mean()
        for i, (lab, _) in enumerate(classes):
            line = f'  {k:5s} {lab:18s}'
            for dd in doses:
                gv = np.zeros(len(bb)); gv[1 + 2 * i] = dd; gv[2 + 2 * i] = dd ** 2 / 10
                line += f' d{dd} {gv @ bb:+.2f}({np.sqrt(gv @ V @ gv):.2f})'
            print(line)

if __name__ == '__main__':
    cls = [(f'c{a}-{z}', (cv >= a) & (cv <= z)) for a, z in [(0, 0), (1, 2), (3, 3), (4, 4), (5, 5), (6, 7)]]
    fit(cls, title='by value')
    cls2 = [(f'c{a}-{z}', (cv >= a) & (cv <= z)) for a, z in [(0, 2), (3, 5), (6, 7)]]
    fit(cls2, title='grouped')
    fit(cls2, sel=size[:, b] <= 4, title='own size <= 4')
    fit(cls2, sel=size[:, b] >= 5, title='own size >= 5')
