"""On the v1 sweep games, in the rule's group: exits vs own and others' punishment by own contribution level,
contribution response of the punished and spillover onto unpunished members."""
import numpy as np, pandas as pd
z = np.load('/tmp/v1_compact.npz'); c, p, m, v = [z[k].astype(np.float32) for k in ['c', 'p', 'mine', 'v']]
G, N, T = c.shape
# --- exits at decisions: player in my group in round r (r=3,7,..,19), left by r+1
rows = []
for r in [3, 7, 11, 15, 19]:
    blk = slice(r - 3, r + 1)
    inblk = m[:, :, blk].min(-1)  # in my group the whole block
    left = (m[:, :, r] == 1) & (m[:, :, r + 1] == 0)
    ownp = p[:, :, blk].sum(-1); ownc = c[:, :, blk].mean(-1)
    grp_p = (p[:, :, blk] * m[:, :, blk]).sum((1, 2))[:, None] - ownp
    lastp = p[:, :, r]
    sel = inblk == 1
    rows.append(pd.DataFrame(dict(r=r, left=left[sel], ownp=ownp[sel], lastp=lastp[sel], ownc=ownc[sel],
                                  othp=np.broadcast_to(grp_p, ownp.shape)[sel],
                                  size=np.broadcast_to(m[:, :, r].sum(1, keepdims=True), ownp.shape)[sel])))
X = pd.concat(rows)
X['cb'] = pd.cut(X.ownc, [-1, 2, 5, 10, 21]); X['pb'] = pd.cut(X.ownp, [-1, 0, 4, 12, 30, 200])
print('exit rate from my group: rows own block contribution, cols own block punishment total')
print(X.pivot_table(index='cb', columns='pb', values='left', aggfunc='mean', observed=True).round(3))
print(X.pivot_table(index='cb', columns='pb', values='left', aggfunc='size', observed=True))
U = X[X.ownp == 0]
print('unpunished members: exit vs others-in-group block punishment')
print(U.groupby(pd.cut(U.othp, [-1, 0, 5, 15, 40, 1000]), observed=True).left.agg(['mean', 'size']).round(3))
print('exit by decision round (all)', X.groupby('r').left.mean().round(3).values)
# --- contribution response: member in my group at t and t+1 (no decision between), t not in block end
rows = []
for t in range(23):
    if t in (3, 7, 11, 15, 19): continue
    stay = (m[:, :, t] == 1) & (m[:, :, t + 1] == 1) & (v[:, :, t] == 1) & (v[:, :, t + 1] == 1)
    othp = (p[:, :, t] * m[:, :, t]).sum(1, keepdims=True) - p[:, :, t]
    rows.append(pd.DataFrame(dict(t=t, c0=c[:, :, t][stay], dc=(c[:, :, t + 1] - c[:, :, t])[stay], ownp=p[:, :, t][stay],
                                  othp=np.broadcast_to(othp, stay.shape)[stay])))
Y = pd.concat(rows)
Y['cb'] = pd.cut(Y.c0, [-1, 0, 2, 5, 10, 15, 21]); Y['pb'] = pd.cut(Y.ownp, [-1, 0, 2, 5, 10, 20, 30])
print('next-round change in contribution: rows own c, cols own punishment')
print(Y.pivot_table(index='cb', columns='pb', values='dc', aggfunc='mean', observed=True).round(2))
U = Y[Y.ownp == 0]
print('unpunished: dc vs others punishment, by own c')
print(U.pivot_table(index='cb', columns=pd.cut(U.othp, [-1, 0, 5, 15, 40, 1000]), values='dc', aggfunc='mean', observed=True).round(2))
