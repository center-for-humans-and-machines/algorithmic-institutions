# v5: v4 sweep: do unpunished members (c0 >= 12, p0 = 0) contribute more when others in my group were punished in round 0?
# group-level: total punishment dealt to the others in round 0 and the number of others hit 10+
import pandas as pd, numpy as np, pyarrow.parquet as pq, re
base = '/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-5-v4_sweep/'
cols = ['episode', 'participant_code', 'round_number', 'punishment', 'contribution', 'group_id', 'run']
out = []
for f in ['per_round_p1of2.parquet', 'per_round_p2of2.parquet']:
    d = pq.read_table(base + f, columns=cols, read_dictionary=['run']).to_pandas()
    for k in ['punishment', 'contribution', 'group_id', 'round_number']: d[k] = d[k].astype('int8')
    cats = d.run.cat.categories
    seat = np.array([0 if re.search(r'managed by pf-', r) else 1 for r in cats])
    d['seat'] = seat[d.run.cat.codes.values]
    d['run'] = d.run.cat.codes
    d = d.sort_values(['run', 'episode', 'participant_code', 'round_number'])
    n = len(d) // 192
    P = d.punishment.values.reshape(n, 8, 24).astype(float)
    C = d.contribution.values.reshape(n, 8, 24).astype(float)
    G = d.group_id.values.reshape(n, 8, 24)
    S = d.seat.values.reshape(n, 8, 24)[:, :, 0]
    del d
    M = (G == S[:, :, None])
    M0 = M[:, :, 0]
    tot_p0 = (P[:, :, 0] * M0).sum(1, keepdims=True) - P[:, :, 0]
    nhit = ((P[:, :, 0] >= 10) * M0).sum(1, keepdims=True) - (P[:, :, 0] >= 10)
    oth_c0 = ((C[:, :, 0] * M0).sum(1, keepdims=True) - C[:, :, 0]) / np.maximum(M0.sum(1, keepdims=True) - 1, 1)
    x = pd.DataFrame({'c0': C[:, :, 0].ravel(), 'p0': P[:, :, 0].ravel(), 'c1': C[:, :, 1].ravel(),
                      'c911': C[:, :, 9:12].mean(-1).ravel(), 'c2023': C[:, :, 20:24].mean(-1).ravel(),
                      'stay4': M[:, :, 4].ravel(), 'stay12': M[:, :, 1:12].all(-1).ravel(), 'stayall': M.all(-1).ravel(),
                      'totp': np.broadcast_to(tot_p0, M0.shape).ravel(), 'nhit': np.broadcast_to(nhit, M0.shape).ravel(),
                      'othc0': oth_c0.ravel(), 'mine': M0.ravel()})
    out.append(x[x.mine])
X = pd.concat(out, ignore_index=True)
pd.set_option('display.width', 250)
U = X[(X.p0 == 0) & (X.c0 >= 12)].copy()
U['nhitb'] = U.nhit.clip(0, 3)
U['othb'] = pd.cut(U.othc0, [-1, 5, 8, 11, 20])
for v in ['c1', 'stay4']:
    print(v, 'unpunished c0>=12, by others c0 mean x number of others hit 10+')
    print(U.pivot_table(index='othb', columns='nhitb', values=v, aggfunc='mean', observed=True).round(2))
for v in ['c911', 'c2023']:
    print(v, '(stayers)'); print(U[U.stay12 if v == 'c911' else U.stayall].pivot_table(index='othb', columns='nhitb', values=v, aggfunc='mean', observed=True).round(2))
print(U.pivot_table(index='othb', columns='nhitb', values='c1', aggfunc='size', observed=True))
