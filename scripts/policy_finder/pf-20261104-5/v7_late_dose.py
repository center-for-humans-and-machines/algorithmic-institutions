"""v7: dose response of block-b punishment, by our size, in v6's sweep.
Within each n-bin, regress pool_from (our pool over rounds 4b..23) on the design's dose
proxy h = mean block-b punishment per member of the design (over all its games in the bin's
B arm), separately in arm B (block b punished, t_stop in [4b, 4b+3]) and arm A (t_stop in
[4b-4, 4b-1], no punishment in b): the B slope minus the A slope is the marginal value of
block-b dose, the A slope absorbing the design's earlier dose. Uses k, thr, w spline-free:
proxy h_d = k*relu(thr - cm_block)-ish is replaced by design means."""
import numpy as np, pandas as pd
D = pd.read_parquet('scripts/policy_finder/pf-20261104-5/v7_blocks.parquet')
W = D.pivot_table(index=['run', 'episode'], columns='block', values='pool')
for b in [1, 2, 3]:
    X = D[D.block == b].set_index(['run', 'episode'])
    X['pool_from'] = W[[j for j in range(b, 6)]].mean(1)
    X = X.reset_index()
    X['nb'] = pd.cut(X.n, [0, 2, 4, 6, 7, 8])
    X = X[X.nb.notna()]
    X['ppm'] = X.P / X.n
    B = X[X.t_stop.between(4 * b, 4 * b + 3)]
    A = X[X.t_stop.between(4 * b - 4, 4 * b - 1)]
    rows = []
    for nb in B.nb.cat.categories:
        Bb, Ab = B[B.nb == nb], A[A.nb == nb]
        h = Bb.groupby('run').ppm.mean()
        # dose proxy for A designs: fit h on params from B designs, predict for A
        prm = X.drop_duplicates('run').set_index('run')[['k', 'thr', 'w']]
        F = np.c_[np.ones(len(h)), np.log(prm.loc[h.index].k), prm.loc[h.index].thr, prm.loc[h.index].w]
        beta = np.linalg.lstsq(F, h.values, rcond=None)[0]
        def proxy(Y):
            q = prm.loc[Y.run]
            return np.c_[np.ones(len(Y)), np.log(q.k), q.thr, q.w] @ beta
        out = {'nbin': str(nb), 'nB': len(Bb), 'nA': len(Ab), 'h_mean': round(h.mean(), 2)}
        for arm, Y in [('B', Bb), ('A', Ab)]:
            x = proxy(Y)
            Z = np.c_[np.ones(len(Y)), x, Y.n]
            co, res, *_ = np.linalg.lstsq(Z, Y.pool_from.values, rcond=None)
            r = Y.pool_from.values - Z @ co
            se = np.sqrt(np.linalg.pinv(Z.T @ Z)[1, 1] * r.var())
            out[arm + '_slope'] = round(co[1], 2); out[arm + '_se'] = round(se, 2)
        rows.append(out)
    print(f'--- block {b}: pool_from per point of block-b punishment per member (design proxy)')
    print(pd.DataFrame(rows).to_string())
