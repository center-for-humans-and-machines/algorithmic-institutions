"""All four sweeps pooled: does punishment lift a member as a function of its ABSOLUTE contribution
(not only relative to the others' mean)? For my-group member-rounds in rounds 0-10 (not pre-decision),
bin by own c, others'-mean o, punishment p; outcome: c three rounds later (members still in my group
then; same-phase only when t%4==0) and the share still in my group at t+4."""
import numpy as np
vers = ['pf-20261104-9', 'pf-20261104-9-v2', 'pf-20261104-9-v3', 'pf-20261104-9-v4']
rows = []
for v in vers:
    z = np.load(f'/tmp/pf/{v}.npz'); C = z['C'].astype(np.float32); V = z['V']; P = z['P'].astype(np.float32); G = z['G']
    m = (G == 0) & (V == 1)
    k = m.sum(2, keepdims=True)
    o = ((C * m).sum(2, keepdims=True) - C * m) / np.maximum(k - m, 1)
    for t in [0, 1, 2, 4, 5, 6, 8, 9, 10]:
        sel = m[..., t] & (k[..., 0, t] >= 2)[..., None]
        ci = C[..., t][sel]; oi = o[..., t][sel]; pi = P[..., t][sel]
        h = 3 if t % 4 == 0 else (4 - t % 4) + 0  # look within the phase or right after
        t2 = min(t + 3, 23)
        stay = (G[..., t + 4 - t % 4] == 0)[sel] if t + 4 - t % 4 < 24 else np.ones_like(ci, bool)
        c2 = C[..., t2][sel]; g2 = (G[..., t2] == 0)[sel] & (V[..., t2] == 1)[sel]
        rows.append(np.column_stack([np.full(ci.shape, t), ci, oi, pi, c2, g2, stay]))
R = np.concatenate(rows)
np.save('/tmp/pf/abs_rows.npy', R)
t, ci, oi, pi, c2, g2, stay = R.T
print('n', len(R))
cb = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 18), (18, 21)]
pb = [(0, 1), (1, 6), (6, 13), (13, 22), (22, 31)]
for early in (True, False):
    print('=== rounds', '0 only' if early else '4-10')
    for lo, hi in cb:
        line = f'c {lo:2d}-{hi:2d}: '
        base = None
        for plo, phi in pb:
            s = (ci >= lo) & (ci < hi) & (pi >= plo) & (pi < phi) & ((t == 0) if early else (t >= 4))
            # control o: report o mean too
            s2 = s & (g2 == 1)
            if s2.sum() < 200:
                line += f' p{plo}-{phi}: {"--":>22s}'; continue
            d = c2[s2].mean() - ci[s2].mean()
            line += f' p{plo}-{phi}: d{d:+5.2f} o{oi[s2].mean():5.1f} st{stay[s].mean():.2f} n{s2.sum()//1000:3d}k'
        print(line)
