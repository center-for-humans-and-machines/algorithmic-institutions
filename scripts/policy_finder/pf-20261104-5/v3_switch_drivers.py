"""v3: in v2's sweep (part 1), what drives a switch at a decision? Per player at decision
rounds 3/7/11/15/19: own group's common_good (per-member share), the other group's, the
punishment received in the decision round and in the 3 rounds before, group sizes."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'scripts/policy_finder/pf-20261104-5')
from v3_load import load
A = load('plots/simulation/policy_finder/pf-20261104-5-v2_sweep/per_round_p1of2.parquet')
c, p, g, v, cg = A['c'], A['p'], A['g'], A['v'], A['cg']
N = c.shape[0]; print('games', N)
# sanity: common_good vs pool / valid members
gi = (g == 0)
pool0 = (1.6 * c * gi).sum(2) - (p * gi * v).sum(2)
nv0 = (gi & v).sum(2)
cg0 = np.where(gi, cg, np.nan); cg0 = np.nanmax(np.where(gi, cg, -1e9), 2)
m = nv0 > 0
print('cg*nvalid vs pool corr', np.corrcoef((cg0 * nv0)[m], pool0[m])[0, 1], 'mean abs diff', np.abs(cg0 * nv0 - pool0)[m].mean())
rows = []
for r in [3, 7, 11, 15, 19]:
    for gg in [0, 1]:
        ing = g[:, r] == gg
        n_own = ing.sum(1, keepdims=True) * np.ones((1, 8))
        n_oth = 8 - n_own
        cg_own = np.array([np.nanmean(np.where(ing[i], cg[i, r], np.nan)) if ing[i].any() else 0 for i in range(N)])
        cg_oth = np.array([np.nanmean(np.where(~ing[i], cg[i, r], np.nan)) if (~ing[i]).any() else 0 for i in range(N)])
        sw = g[:, r + 1] != g[:, r]
        pprev = p[:, r - 3:r].sum(1)
        cm_own = (c[:, r] * ing).sum(1) / np.maximum(ing.sum(1), 1)
        df = pd.DataFrame(dict(sw=sw[ing], cg=np.repeat(cg_own[:, None], 8, 1)[ing], cgo=np.repeat(cg_oth[:, None], 8, 1)[ing],
                               pd=p[:, r][ing], pp=pprev[ing], n=n_own[ing], c=c[:, r][ing], r=r, gid=gg,
                               cmo=np.repeat(cm_own[:, None], 8, 1)[ing]))
        rows.append(df)
D = pd.concat(rows); D['gap'] = D.cg - D.cgo
print('switch rate', D.sw.mean().round(3), len(D))
print(D.groupby(pd.cut(D.gap, [-99, -10, -5, -2, 2, 5, 10, 99]), observed=True).sw.agg(['mean', 'size']).round(3))
print(D.groupby(pd.cut(D.pd, [-1, 0, 2, 5, 10, 20, 30]), observed=True).sw.agg(['mean', 'size']).round(3))
print(D.groupby(pd.cut(D.pp, [-1, 0, 5, 15, 30, 90]), observed=True).sw.agg(['mean', 'size']).round(3))
print(D.groupby('n').sw.agg(['mean', 'size']).round(3))
print(D.groupby('r').sw.agg(['mean', 'size']).round(3))
X = np.c_[np.ones(len(D)), D.cg, D.cgo, D.pd, D.pp, D.n, D.r, D.c]
b = np.linalg.lstsq(X, D.sw.values.astype(float), rcond=None)[0]
print('LPM: const cg cg_o p_dec p_prev3 n round contrib'); print(np.round(b, 4))
D.to_parquet('/tmp/v3_dec.parquet')
