"""v3: same as v3_dose_v2.py on v1 sweep (v1 params as controls). dose-response (rule-driven variation in p), by the punished
player's contribution class and round group. Regress own c(t+k)-c(t) and others' mean change on
p(t), with c dummies, c(t-1), others' mean, t, earlier punishments, others' p and v2 rule params."""
import numpy as np
z = np.load('/tmp/v1sweep.npz')
sub = np.arange(0, z['C'].shape[0], 2)
C, P, V = z['C'][sub].astype(np.float32), z['P'][sub].astype(np.float32), z['V'][sub]
R = z['R'][sub]; prm = z['prm']
G = C.shape[0]
npun = np.concatenate([np.zeros((G, 4, 1)), np.cumsum(P > 0, 2)[:, :, :-1]], 2)
K = (1, 3, 6, 10)
rows = []
for t in range(0, 20):
    for i in range(4):
        o = [j for j in range(4) if j != i]
        g = np.where(V[:, i, t] & V[:, :, t].all(1))[0]
        rec = [np.full(len(g), t), C[g, i, t], C[g, i, t - 1] if t > 0 else C[g, i, t], C[g][:, o, t].mean(1), P[g, i, t],
               npun[g, i, t], P[g][:, o, t].mean(1), prm[R[g], 0], np.log(prm[R[g], 1]), np.log(prm[R[g], 3]), prm[R[g], 2] - t]
        for k in K:
            tk = min(t + k, 23); rec += [C[g, i, tk], C[g][:, o, tk].mean(1)]
        rows.append(np.column_stack(rec).astype(np.float32))
D = np.concatenate(rows)
cols = ['t', 'c', 'cprev', 'oth', 'p', 'npun', 'pothers', 'st', 'dose', 'lcu', 'hz'] + sum([[f'own{k}', f'oth{k}'] for k in K], [])
ix = {c: j for j, c in enumerate(cols)}
CBINS = [(0, 3), (3, 8), (8, 12), (12, 15), (15, 18), (18, 21)]
def reg(M, label):
    cd = np.stack([(M[:, ix['c']] >= lo) & (M[:, ix['c']] < hi) for lo, hi in CBINS], 1).astype(float)
    X = np.column_stack([cd, M[:, ix['cprev']], M[:, ix['oth']], M[:, ix['t']], M[:, ix['npun']], M[:, ix['pothers']],
                         M[:, ix['st']], M[:, ix['dose']], M[:, ix['lcu']], np.clip(M[:, ix['hz']], -1, 12), M[:, ix['p']]])
    out = []
    for k in K:
        b1 = np.linalg.lstsq(X, M[:, ix[f'own{k}']] - M[:, ix['c']], rcond=None)[0][-1]
        b2 = np.linalg.lstsq(X, M[:, ix[f'oth{k}']] - M[:, ix['oth']], rcond=None)[0][-1]
        out.append(f'k{k}: {b1:+.3f}/{b2:+.3f}')
    print(f'{label:16s} n={len(M):7d} meanp={M[:, ix["p"]].mean():5.2f} sdp={M[:, ix["p"]].std():5.2f} | ' + ' | '.join(out), flush=True)
print('own/others-mean change per punishment point')
for tl, th_ in [(0, 0), (1, 3), (4, 8)]:
    for lo, hi in CBINS:
        m = (D[:, ix['t']] >= tl) & (D[:, ix['t']] <= th_) & (D[:, ix['c']] >= lo) & (D[:, ix['c']] < hi)
        if m.sum() > 1000: reg(D[m], f't{tl}-{th_} c{lo}-{hi}')
