"""v6: non-parametric dose curve on v5's sweep (src 0 of the pooled data): per target class and dose bin, the
count of members hit in that bin minus its design expectation given c4 (fraction of v5 points with lo<=c4<=hi
and floor(amt) in the bin). Coefficient = total rest-of-game lead per member hit at that dose."""
import numpy as np, json
D = dict(np.load('/tmp/v6_pool3.npz'))
s0 = D['src'] == 0
here = D['here'].astype(bool)[s0]; c4 = D['c4'][s0].astype(int); act = D['act'][s0]; size4 = D['size4'][s0]
pts = json.load(open('plots/simulation/policy_finder/pf-20261112-6-v5_sweep/sweep.json'))['points']
PAR = np.array([[q['params'][k] for k in ['lo', 'hi', 'amt']] for q in pts if '_s' in q['name']])
print('design points', len(PAR))
bins = [(8, 11), (12, 15), (16, 19), (20, 23), (24, 28)]
def pbin(a, z):
    return np.array([np.mean((x >= PAR[:, 0]) & (x <= PAR[:, 1]) & (np.floor(PAR[:, 2]) >= a) & (np.floor(PAR[:, 2]) <= z)) for x in range(21)])
PB = [pbin(a, z) for a, z in bins]
def run(classes, sel, title):
    cols = []
    for _, cm in classes:
        cm = cm & here
        for (a, z), pb in zip(bins, PB):
            cols.append((((act >= a) & (act <= z)) * cm).sum(1) - (pb[c4] * cm).sum(1))
            cols.append((pb[c4] * cm).sum(1))
    X = np.column_stack([np.ones(len(c4))] + cols + [size4, D['seat'][s0]]).astype(float)[sel]
    print('==', title, sel.sum())
    for k in ['lead', 's8', 'ownc']:
        y = D[k][s0][sel]; bb, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ bb
        V = np.linalg.pinv(X.T @ X) * (e ** 2).mean()
        for i, (lab, _) in enumerate(classes):
            line = f'  {k:5s} {lab:8s}'
            for j, (a, z) in enumerate(bins):
                q = 1 + 2 * (i * len(bins) + j)
                line += f' {a}-{z}: {bb[q]:+.2f}({np.sqrt(V[q, q]):.2f})'
            print(line)
cl = [('c0', c4 == 0), ('c1-2', (c4 >= 1) & (c4 <= 2)), ('c3-5', (c4 >= 3) & (c4 <= 5)), ('c6-7', (c4 >= 6) & (c4 <= 7))]
run(cl, np.ones(len(c4), bool), 'all v5 games')
run(cl, size4 <= 4, 'size <= 4')
run(cl, size4 >= 5, 'size >= 5')
if True:
    new = D['new4'].astype(bool)[s0]
    cl2 = [('c0-2 old', (c4 <= 2) & ~new), ('c0-2 new', (c4 <= 2) & new), ('c3-5 old', (c4 >= 3) & (c4 <= 5) & ~new),
           ('c3-5 new', (c4 >= 3) & (c4 <= 5) & new), ('c6-7', (c4 >= 6) & (c4 <= 7))]
    run(cl2, np.ones(len(c4), bool), 'newcomer split, all v5 games')
