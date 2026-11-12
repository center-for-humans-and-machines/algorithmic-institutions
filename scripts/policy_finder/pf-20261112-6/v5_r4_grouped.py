"""v5: grouped round-4 classes (0-2, 3-5, 6-7, 8-10, 11+), total effect of one member hit with 5/9/13/16 points,
and splits of the 3-5 / 0-2 classes by the target's trend (c4 vs own mean of rounds 0-3) and by the others' mean."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import v5_r4_pooled as M  # reuses prep (prints its own results first)
b = 4
M.classes = [(0, 2), (3, 5), (6, 7), (8, 10), (11, 20)]
def fit2(mask_list, title, doses=(5, 9, 13, 16)):
    """mask_list: list of (label, fn(S)->[n,8] bool) target classes."""
    Xs, ys = [], {k: [] for k in ['lead', 's8', 'ownc']}
    for j, S in enumerate([M.A, M.B]):
        R, E = [], []
        for lab, fn in mask_list:
            cm = fn(S) & S['mine'][:, :, b]
            for f in (lambda d: d, lambda d: d ** 2 / 10):
                P = (f(S['act']) * cm).sum(1); Ex = (f(S['Ds']) * cm[:, None]).sum(2).mean(1)
                R.append(P - Ex); E.append(Ex)
        Xs.append(np.column_stack(R + E + [np.full(len(P), j), S['size'][:, b]]))
        for k in ys: ys[k].append(S[k])
    X = np.column_stack([np.ones(sum(len(x) for x in Xs)), np.concatenate(Xs)])
    print('==', title)
    for k in ys:
        y = np.concatenate(ys[k]); bb, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ bb
        V = np.linalg.pinv(X.T @ X) * (e ** 2).mean()
        for i, (lab, _) in enumerate(mask_list):
            line = f'  {k:5s} {lab:22s}'
            for dd in doses:
                gv = np.zeros(len(bb)); gv[1 + 2 * i] = dd; gv[2 + 2 * i] = dd ** 2 / 10
                line += f' d{dd} {gv @ bb:+.2f}({np.sqrt(gv @ V @ gv):.2f})'
            print(line)
cv = lambda S: S['c'][:, :, b]
fit2([(f'c{a}-{z}', (lambda a, z: lambda S: (cv(S) >= a) & (cv(S) <= z))(a, z)) for a, z in M.classes], 'grouped')
def past(S): return S['c'][:, :, :4].mean(2)
def oth(S):
    m = S['mine'][:, :, b]; s = (S['c'][:, :, b] * m).sum(1, keepdims=True); n = m.sum(1, keepdims=True)
    return (s - S['c'][:, :, b]) / np.maximum(n - 1, 1)
lowmid = lambda S: (cv(S) >= 3) & (cv(S) <= 5)
low = lambda S: cv(S) <= 2
fit2([('c3-5 past<=c4+1', lambda S: lowmid(S) & (past(S) <= cv(S) + 1)), ('c3-5 past>c4+1', lambda S: lowmid(S) & (past(S) > cv(S) + 1)),
      ('c0-2 past<=3', lambda S: low(S) & (past(S) <= 3)), ('c0-2 past>3', lambda S: low(S) & (past(S) > 3)),
      ('c6+', lambda S: cv(S) >= 6)], 'split by own past mean (rounds 0-3)')
fit2([('c3-5 others<10', lambda S: lowmid(S) & (oth(S) < 10)), ('c3-5 others>=10', lambda S: lowmid(S) & (oth(S) >= 10)),
      ('c0-2 others<10', lambda S: low(S) & (oth(S) < 10)), ('c0-2 others>=10', lambda S: low(S) & (oth(S) >= 10)),
      ('c6+', lambda S: cv(S) >= 6)], 'split by mates mean at round 4')
for S in (M.A, M.B):
    m = S['mine'][:, :, b]; cc = cv(S)
    print('per game own members by class:', [round(float(((cc >= a) & (cc <= z) & m).sum(1).mean()), 2) for a, z in M.classes], 'size', S['size'][:, b].mean())
