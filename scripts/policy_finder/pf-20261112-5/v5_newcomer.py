"""v5: per-player answer to the round-4 shock, newcomers vs stayers, from the natural-experiment pairs (identical until
round 4, punishment differs at round 4). Per player of my group at round 4 with dp>0 in one side and 0 in the other:
treated-minus-control mean contribution in rounds 5-7, being in my group at round 8, mean contribution rounds 8-23
(wherever they are), and in my group rounds 8-23 share."""
import numpy as np, itertools
V = ['v1', 'v2', 'v3', 'v4']
D = {v: np.load(f'/tmp/pf/{v}.npz') for v in V}
D = {v: dict(c=z['c'], p=z['p'], m=z['mine'].astype(bool), point=z['point'], seat=z['seat']) for v, z in D.items()}
rows = []
for a, b in itertools.combinations(V, 2):
    A, B = D[a], D[b]
    for k in np.unique(A['point']):
        for s in (0, 1):
            ia = np.where((A['point'] == k) & (A['seat'] == s))[0]; ib = np.where((B['point'] == k) & (B['seat'] == s))[0]
            if len(ia) != len(ib) or len(ia) == 0: continue
            cA, cB, pA, pB, mA, mB = A['c'][ia], B['c'][ib], A['p'][ia], B['p'][ib], A['m'][ia], B['m'][ib]
            same = (pA[:, :, :4] == pB[:, :, :4]).all((1, 2)) & (cA[:, :, :5] == cB[:, :, :5]).all((1, 2)) & (mA[:, :, :5] == mB[:, :, :5]).all((1, 2))
            diff4 = (pA[:, :, 4] != pB[:, :, 4]).any(1)
            for e in np.where(same & diff4)[0]:
                for i in range(8):
                    if not mA[e, i, 4]: continue
                    x, y = int(pA[e, i, 4]), int(pB[e, i, 4])
                    if (x > 0) == (y > 0): continue
                    T, C = (B, A) if y > 0 else (A, B)
                    cT, cC = (cB, cA) if y > 0 else (cA, cB); mT, mC = (mB, mA) if y > 0 else (mA, mB)
                    rows.append((max(x, y), cA[e, i, 4], int(not mA[e, i, 3]), mA[e, :, 4].sum(),
                                 cT[e, i, 5:8].mean() - cC[e, i, 5:8].mean(), int(mT[e, i, 8]) - int(mC[e, i, 8]),
                                 cT[e, i, 8:].mean() - cC[e, i, 8:].mean(), mT[e, i, 8:].mean() - mC[e, i, 8:].mean(),
                                 # rest of my group (others) contribution rounds 5-7
                                 (cT[e, :, 5:8] * mT[e, :, 5:8]).sum() - (cC[e, :, 5:8] * mC[e, :, 5:8]).sum()))
R = np.array(rows, float)
print('n treated players', len(R))
for lab, sel in [('newcomer', R[:, 2] == 1), ('stayer', R[:, 2] == 0)]:
    for band in [(0, 2), (3, 5), (6, 20)]:
        s = sel & (R[:, 1] >= band[0]) & (R[:, 1] <= band[1])
        if s.sum() < 30: continue
        m = R[s].mean(0); se = R[s].std(0) / np.sqrt(s.sum())
        print(f'{lab:9s} c{band[0]}-{band[1]:<3d} n={s.sum():5d} dose {m[0]:.1f}  dc5-7 {m[4]:+.2f}({se[4]:.2f})  in@8 {m[5]:+.3f}({se[5]:.3f})  dc8-23 {m[6]:+.2f}({se[6]:.2f})  in8-23 {m[7]:+.3f}({se[7]:.3f})  groupC5-7 {m[8]:+.1f}({se[8]:.1f})')
