# AH switching in v1's sweep: who leaves our group / joins it, vs own punishment in the block,
# our vs their punishment per head and pool per head.
import numpy as np, pandas as pd
d = np.load('/tmp/v1sweep.npz')
c, p, v, g, ours, opp = d['c'].astype(np.float32), d['p'].astype(np.float32), d['v'], d['g'], d['ours'], d['opp']
ing = (g == ours[:, None, None])
rows = []
for t in [3, 7, 11, 15, 19]:
    blk = slice(t - 3, t + 1)
    stay_in = ing[:, :, blk].all(2)
    stay_out = (~ing[:, :, blk]).all(2)
    nin = ing[:, :, t].sum(1); nout = 8 - nin
    pool_in = ((1.6 * c[:, :, t] - p[:, :, t]) * ing[:, :, t]).sum(1)
    pool_out = ((1.6 * c[:, :, t] - p[:, :, t]) * ~ing[:, :, t]).sum(1)
    pin = (p[:, :, blk] * ing[:, :, blk]).sum((1, 2)) / np.maximum(ing[:, :, blk].sum((1, 2)), 1)
    pout = (p[:, :, blk] * ~ing[:, :, blk]).sum((1, 2)) / np.maximum((~ing[:, :, blk]).sum((1, 2)), 1)
    for side, m in [('in', stay_in), ('out', stay_out)]:
        moved = (ing[:, :, t + 1] != ing[:, :, t])
        e, i = np.nonzero(m)
        rows.append(pd.DataFrame(dict(side=side, t=t, opp=opp[e], moved=moved[e, i], ownp=p[e, i, blk].sum(1),
            plast=p[e, i, t], ownc=c[e, i, blk].mean(1), n_own=np.where(side == 'in', nin, nout)[e],
            dpool=np.where(side == 'in', pool_in / np.maximum(nin, 1) - pool_out / np.maximum(nout, 1),
                           pool_out / np.maximum(nout, 1) - pool_in / np.maximum(nin, 1))[e],
            dpun=np.where(side == 'in', pin - pout, pout - pin)[e])))
D = pd.concat(rows, ignore_index=True)
print('leave rate overall', D.groupby('side').moved.mean().round(3).to_dict())
D['ob'] = pd.cut(D.ownp, [-1, 0, 4, 10, 20, 40, 200])
print('leave by own block punishment\n', D.groupby(['side', 'ob'], observed=True).moved.agg(['mean', 'size']).round(3).unstack(0).to_string())
D['db'] = pd.cut(D.dpun, [-40, -3, -1, -0.25, 0.25, 1, 3, 40])
print('leave by own group punishment per head minus other\n', D.groupby('db', observed=True).moved.agg(['mean', 'size']).round(3).to_string())
D['pb'] = pd.cut(D.dpool, [-200, -40, -20, -8, 8, 20, 40, 200])
print('leave by own pool per head minus other (decision round)\n', D.groupby('pb', observed=True).moved.agg(['mean', 'size']).round(3).to_string())
# logit-ish OLS (linear probability) 
X = np.column_stack([np.ones(len(D)), D.ownp, D.plast, D.dpun, D.dpool / 20, D.n_own, D.ownc])
b = np.linalg.lstsq(X, D.moved.astype(float), rcond=None)[0]
print('LPM: const %.3f ownp %.4f plast %.4f dpun %.4f dpool/20 %.4f n_own %.4f ownc %.4f' % tuple(b))
