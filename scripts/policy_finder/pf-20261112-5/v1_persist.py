"""v1 games: persistence of the contribution answer, and spillover on the group.
For a member of the rule's group with contribution c_t in a bin, punished p_t at round t (t not a decision
round, t<=18), followed while it stays: its contribution at t+1..t+4 minus c_t; and the other members' mean
contribution change at t+1..t+2. Compared with unpunished members of the same c bin and similar others' mean."""
import numpy as np, pandas as pd
z = np.load('/tmp/v1_compact.npz'); c, p, m, v = [z[k].astype(np.float32) for k in ['c', 'p', 'mine', 'v']]
rows = []
for t in range(1, 19):
    if t % 4 == 3: continue
    ok = (m[:, :, t] == 1) & (v[:, :, t] == 1)
    n = m[:, :, t].sum(1, keepdims=True)
    oth = ((c[:, :, t] * m[:, :, t]).sum(1, keepdims=True) - c[:, :, t]) / np.maximum(n - 1, 1)
    d = {f'd{k}': c[:, :, t + k] - c[:, :, t] for k in range(1, 5)}
    st = {f's{k}': m[:, :, t + k] for k in range(1, 5)}
    # others' mean change at t+1
    nxt = ((c[:, :, t + 1] * m[:, :, t + 1] * m[:, :, t]).sum(1, keepdims=True) - c[:, :, t + 1] * m[:, :, t + 1])
    cnt = (m[:, :, t + 1] * m[:, :, t]).sum(1, keepdims=True) - m[:, :, t + 1]
    othd = nxt / np.maximum(cnt, 1) - oth
    sel = ok & (n > 1)
    rows.append(pd.DataFrame(dict(t=t, c0=c[:, :, t][sel], p0=p[:, :, t][sel], oth=np.broadcast_to(oth, sel.shape)[sel],
                                  othd=np.broadcast_to(othd, sel.shape)[sel],
                                  **{k: x[sel] for k, x in d.items()}, **{k: x[sel] for k, x in st.items()})))
X = pd.concat(rows)
X['pb'] = pd.cut(X.p0, [-1, 0, 3, 8, 15, 30])
for lo, hi in [(0, 0), (1, 4), (5, 9), (10, 14)]:
    Y = X[(X.c0 >= lo) & (X.c0 <= hi) & (X.oth >= 6) & (X.oth <= 12)]
    print(f'c_t in [{lo},{hi}], others mean 6-12: change in own c at +k (stayers through t+k), others mean change at +1')
    out = {}
    for k in range(1, 5):
        Z = Y[Y[f's{k}'] == 1]
        out[f'd{k}'] = Z.groupby('pb', observed=True)[f'd{k}'].mean()
    out['othd1'] = Y.groupby('pb', observed=True).othd.mean()
    out['n'] = Y.groupby('pb', observed=True).size()
    print(pd.DataFrame(out).round(2))
