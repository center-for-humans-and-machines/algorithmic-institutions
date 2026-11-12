"""v3 sweep natural experiment at a block start r0 (0, 4, 8, 12): points whose window punishes round r0 (treated) vs
points that do not (control). Members of my group at r0 with c<=hi are punished D in treated points.
Per D bin: the punished-eligible member's contribution change (rounds r0+1..r0+3 minus r0), exit at the next
decision, and the group-level: my members' mean contribution r0+1..23, members at end, other group's contribution."""
import numpy as np, pandas as pd, sys
z = np.load('/tmp/pf/v3.npz'); c, p, m, v = [z[k].astype(np.float32) for k in ['c', 'p', 'mine', 'v']]
P = pd.DataFrame(z['params'], columns=['D', 'hi', 'first', 'nblocks', 'w'], index=z['param_idx'])
pt = z['point'].astype(int); par = P.loc[pt].reset_index(drop=True)
for r0 in [0, 4, 8, 12]:
    b = r0 // 4
    treat = ((par['first'] <= b) & (par['first'] + par['nblocks'] > b)).values
    # control: not open in this block, and same history before r0 (no punishment open earlier) - require first > b
    ctrl = (par['first'] > b).values
    # but treated points may have been treated earlier: require first == b for clean comparison when b > 0
    clean_t = treat & (par['first'] == b).values
    hi = par.hi.values[:, None]
    elig = (m[:, :, r0] == 1) & (v[:, :, r0] == 1) & (c[:, :, r0] <= hi)
    dc = c[:, :, r0 + 1:r0 + 4].mean(2) - c[:, :, r0]
    dec = r0 + 3
    stay = m[:, :, dec + 1] if dec + 1 < 24 else m[:, :, 23]
    rows = []
    Db = pd.cut(par.D, [2, 5, 10, 16, 24.1]).astype(str).values
    for name, sel in [('ctrl', ctrl), ('treat', clean_t)]:
        for db in np.unique(Db):
            s = sel & (Db == db)
            e = elig & s[:, None]
            pool = 1.6 * c * v - p
            lead = ((pool * m).sum(1) - (pool * (1 - m)).sum(1))[s][:, r0:].mean(1)
            rows.append(dict(arm=name, D=db, n=e.sum(), dc=dc[e].mean(), stay=stay[e].mean(),
                             c_mine=((c * m)[s][:, :, r0 + 1:].sum((1, 2)) / m[s][:, :, r0 + 1:].sum((1, 2))).mean(),
                             c_oth=((c * (1 - m))[s][:, :, r0 + 1:].sum((1, 2)) / (1 - m)[s][:, :, r0 + 1:].sum((1, 2))).mean(),
                             mem_end=m[s][:, :, 23].sum(1).mean(), lead_rest=lead.mean(), lead_se=lead.std() / np.sqrt(s.sum())))
    print('round', r0); print(pd.DataFrame(rows).round(3).to_string())
