"""v7: value of punishing in block b (1..4), by our size at the block's start, in v6's sweep.
Natural experiment via t_stop: A = t_stop in [4b-4, 4b-1] (nothing in block b or later),
B = t_stop in [4b, 4b+3] (block b punished, nothing after). Outcome: our mean pool over rounds
4b..23 (pool from b on), n_next, contribution per member after block b. Controls: none other
than n-bin (Sobol design makes other params balanced); report B - A with se."""
import numpy as np, pandas as pd
D = pd.read_parquet('scripts/policy_finder/pf-20261104-5/v7_blocks.parquet')
W = D.pivot_table(index=['run', 'episode'], columns='block', values='pool').reset_index()
for b in range(1, 5):
    X = D[D.block == b].copy()
    # pool from block b on
    poolcols = W[[j for j in range(b, 6)]].mean(1)
    X = X.merge(W[['run', 'episode']].assign(pool_from=poolcols.values), on=['run', 'episode'])
    X['grp'] = np.where(X.t_stop.between(4 * b - 4, 4 * b - 1), 'A', np.where(X.t_stop.between(4 * b, 4 * b + 3), 'B', None))
    X = X[X.grp.notna()]
    X['nb'] = pd.cut(X.n, [-1, 0, 2, 4, 6, 7, 8])
    print(f'--- block {b}: rows A {sum(X.grp=="A")} B {sum(X.grp=="B")}')
    rows = []
    for nb, Y in X.groupby('nb', observed=True):
        r = {'nbin': str(nb), 'share': len(Y) / len(X)}
        for v in ['P', 'pool_from', 'n_next', 'cm_rest', 'pool']:
            a, bb = Y[Y.grp == 'A'][v], Y[Y.grp == 'B'][v]
            r[v] = round(bb.mean() - a.mean(), 2)
            r[v + '_se'] = round(np.sqrt(a.var() / len(a) + bb.var() / len(bb)), 2)
        rows.append(r)
    print(pd.DataFrame(rows).to_string())
