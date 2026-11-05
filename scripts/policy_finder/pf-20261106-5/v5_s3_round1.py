# v5: v4 sweep, my group's starting members: effect of the round-1 hit by round-1 contribution and whether
# the player improved after round 0; outcome = stay through decision 1, c in rounds 9-11 (stayers), and the player's
# value to my pool in rounds 2-11 (1.6*c - p while in my group, 0 otherwise)
import pandas as pd, numpy as np, pyarrow.parquet as pq, re
base = '/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-5-v4_sweep/'
cols = ['episode', 'participant_code', 'round_number', 'punishment', 'contribution', 'group_id', 'run']
out = []
for f in ['per_round_p1of2.parquet', 'per_round_p2of2.parquet']:
    d = pq.read_table(base + f, columns=cols, read_dictionary=['run'], filters=[('round_number', '<', 12)]).to_pandas()
    for k in ['punishment', 'contribution', 'group_id', 'round_number']: d[k] = d[k].astype('int8')
    cats = d.run.cat.categories
    seat = np.array([0 if re.search(r'managed by pf-', r) else 1 for r in cats])
    opp = np.array([int(re.search(r'opponent_(\d)', r).group(1)) for r in cats])
    d['seat'] = seat[d.run.cat.codes.values]; d['opp'] = opp[d.run.cat.codes.values]
    d['run'] = d.run.cat.codes
    d = d.sort_values(['run', 'episode', 'participant_code', 'round_number'])
    n = len(d) // 96
    P = d.punishment.values.reshape(n, 8, 12).astype(float)
    C = d.contribution.values.reshape(n, 8, 12).astype(float)
    G = d.group_id.values.reshape(n, 8, 12)
    S = d.seat.values.reshape(n, 8, 12)[:, :, 0]; O = d.opp.values.reshape(n, 8, 12)[:, :, 0]
    del d
    M = (G == S[:, :, None])
    val = ((1.6 * C - P) * M)[:, :, 2:].sum(-1)
    x = pd.DataFrame({'c0': C[:, :, 0].ravel(), 'c1': C[:, :, 1].ravel(), 'p0': P[:, :, 0].ravel(), 'p1': P[:, :, 1].ravel(),
                      'c911': C[:, :, 9:12].mean(-1).ravel(), 'stay4': M[:, :, 4].ravel(), 'stay12': M[:, :, 2:12].all(-1).ravel(),
                      'val': val.ravel(), 'opp': O.ravel(), 'mine': M[:, :, 0].ravel()})
    out.append(x[x.mine])
X = pd.concat(out, ignore_index=True)
X.to_parquet('/tmp/v5/r1.parquet')
pd.set_option('display.width', 250)
X['c1b'] = pd.cut(X.c1, [-1, 2, 5, 8, 10, 12, 20])
X['p1b'] = pd.cut(X.p1, [-1, 0, 10, 20, 29, 30])
X['rose'] = np.where(X.c1 >= X.c0 + 3, 'rose3+', 'not')
X['hit0'] = X.p0 >= 10
for v in ['stay4', 'c911', 'val']:
    for h in [True, False]:
        print(v, 'hit0 >= 10' if h else 'hit0 < 10', '| rows c1 band, cols p1 band')
        Z = X[X.hit0 == h]
        if v == 'c911': Z = Z[Z.stay12]
        print(Z.pivot_table(index=['c1b'], columns='p1b', values=v, aggfunc='mean', observed=True).round(2))
print(X[X.hit0].pivot_table(index='c1b', columns='p1b', values='val', aggfunc='size', observed=True))
