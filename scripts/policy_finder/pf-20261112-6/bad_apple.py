"""v3: is a persistent low contributor worth keeping? Reference sims, zero-managed groups (zero_vs_zero both
groups, zero_vs_ah the zero group). At decisions 3/7/11/15, groups holding exactly one member with c <= 3 at d-1
and d: regress the rest-of-game lead (own pool - other pool, mean over rounds d+1..23) and the remaining members'
contributions on whether that member moved out, controlling for the pre-decision state."""
import numpy as np, pandas as pd
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet',
                    columns=['run', 'episode', 'round_number', 'participant_code', 'group_id', 'contribution', 'punishment'])
rows = []
for runkey, mgs in [('zero_vs_zero', [0, 1]), ('zero_vs_ah', [0]), ('ah_vs_zero', [1])]:
    s = d[d.run.str.endswith(runkey)].sort_values(['episode', 'participant_code', 'round_number'])
    n = len(s) // 192
    c = s.contribution.to_numpy().reshape(n, 8, 24).astype(float)
    g = s.group_id.to_numpy().reshape(n, 8, 24)
    p = s.punishment.to_numpy().reshape(n, 8, 24).astype(float)
    for mg in mgs:
        mine = g == mg
        pool_o = (1.6 * c * mine - p * mine).sum(1); pool_x = (1.6 * c * ~mine - p * ~mine).sum(1)
        for dec in [3, 7, 11, 15]:
            low = (c[:, :, dec] <= 3) & (c[:, :, dec - 1] <= 3) & mine[:, :, dec]
            one = low.sum(1) == 1
            idx = np.where(one)[0]
            j = low[idx].argmax(1)
            moved = (g[idx, j, dec + 1] != mg).astype(float)
            # rest members (not j) in group at dec who stayed: their mean contribution next block
            stay_rest = mine[idx, :, dec] & mine[idx, :, dec + 1]
            stay_rest[np.arange(len(idx)), j] = False
            nr = stay_rest.sum(1)
            pre_rest = np.where(nr > 0, (c[idx, :, dec] * stay_rest).sum(1) / np.maximum(nr, 1), np.nan)
            post_rest = np.where(nr > 0, (c[idx, :, dec + 1:dec + 5].mean(2) * stay_rest).sum(1) / np.maximum(nr, 1), np.nan)
            lead = (pool_o[idx, dec + 1:] - pool_x[idx, dec + 1:]).mean(1)
            size = mine[idx, :, dec].sum(1)
            sc = (c[idx, :, dec] * mine[idx, :, dec]).sum(1); so = (c[idx, :, dec] * ~mine[idx, :, dec]).sum(1)
            for k in range(len(idx)):
                rows.append((runkey, dec, moved[k], lead[k], pre_rest[k], post_rest[k], size[k], sc[k], so[k], c[idx[k], j[k], dec]))
R = pd.DataFrame(rows, columns=['run', 'dec', 'moved', 'lead', 'pre_rest', 'post_rest', 'size', 'sc', 'so', 'cj'])
print(len(R), R.moved.mean())
for y in ['lead', 'post_rest']:
    Q = R.dropna()
    X = np.column_stack([np.ones(len(Q)), Q.moved, Q.pre_rest, Q['size'], Q.sc, Q.so, Q.cj] + [(Q.dec == k).astype(float) for k in [7, 11, 15]])
    b, *_ = np.linalg.lstsq(X, Q[y], rcond=None)
    r = Q[y] - X @ b
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (r ** 2).mean()))
    print(f'{y}: moved {b[1]:+.3f} (se {se[1]:.3f}); pre_rest {b[2]:+.3f}; size {b[3]:+.3f}')
    for dec in [3, 7, 11, 15]:
        q = Q[Q.dec == dec]
        Xd = np.column_stack([np.ones(len(q)), q.moved, q.pre_rest, q['size'], q.sc, q.so, q.cj])
        b, *_ = np.linalg.lstsq(Xd, q[y], rcond=None)
        r = q[y] - Xd @ b
        se = np.sqrt(np.diag(np.linalg.pinv(Xd.T @ Xd) * (r ** 2).mean()))
        print(f'   dec {dec} n {len(q)}: moved {b[1]:+.3f} ({se[1]:.3f})')
