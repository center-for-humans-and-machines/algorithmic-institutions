# v5: v3 sweep (t_end up to 22): does a hit in round 20 (after the last decision: membership is fixed) raise
# contributions in rounds 21-23? player level, my group in round 20, by c20 band x own p20 band, and others hit.
import pandas as pd, numpy as np, pyarrow.parquet as pq, re
base = '/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-5-v3_sweep/'
cols = ['episode', 'participant_code', 'round_number', 'punishment', 'contribution', 'group_id', 'run']
out = []
for f in ['per_round_p1of2.parquet', 'per_round_p2of2.parquet']:
    d = pq.read_table(base + f, columns=cols, read_dictionary=['run'], filters=[('round_number', '>=', 16)]).to_pandas()
    for k in ['punishment', 'contribution', 'group_id', 'round_number']: d[k] = d[k].astype('int8')
    cats = d.run.cat.categories
    seat = np.array([0 if re.search(r'managed by pf-', r) else 1 for r in cats])
    d['seat'] = seat[d.run.cat.codes.values]
    d['run'] = d.run.cat.codes
    d = d.sort_values(['run', 'episode', 'participant_code', 'round_number'])
    n = len(d) // 64
    P = d.punishment.values.reshape(n, 8, 8).astype(float)
    C = d.contribution.values.reshape(n, 8, 8).astype(float)
    G = d.group_id.values.reshape(n, 8, 8)
    S = d.seat.values.reshape(n, 8, 8)[:, :, 0]
    del d
    M = (G == S[:, :, None])[:, :, 4]  # round 20 (index 4)
    nh = ((P[:, :, 4] >= 5) * M).sum(1, keepdims=True) - (P[:, :, 4] >= 5)
    x = pd.DataFrame({'c19': C[:, :, 3].ravel(), 'c20': C[:, :, 4].ravel(), 'p20': P[:, :, 4].ravel(),
                      'c2123': C[:, :, 5:].mean(-1).ravel(), 'nh': np.broadcast_to(nh, M.shape).ravel(), 'mine': M.ravel()})
    out.append(x[x.mine])
X = pd.concat(out, ignore_index=True)
pd.set_option('display.width', 250)
X['d'] = X.c2123 - X.c20
X['cb'] = pd.cut(X.c20, [-1, 2, 5, 8, 11, 20]); X['pb'] = pd.cut(X.p20, [-1, 0, 5, 15, 29, 30])
print('c21-23 minus c20, own hit'); print(X.pivot_table(index='cb', columns='pb', values='d', aggfunc='mean', observed=True).round(2))
print(X.pivot_table(index='cb', columns='pb', values='d', aggfunc='size', observed=True))
U = X[X.p20 == 0].copy(); U['nhb'] = U.nh.clip(0, 2)
print('unpunished: c21-23 minus c20 by others hit 5+'); print(U.pivot_table(index='cb', columns='nhb', values='d', aggfunc='mean', observed=True).round(2))
