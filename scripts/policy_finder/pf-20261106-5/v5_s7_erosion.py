# v5: v4 sweep: at block starts 4, 8, 12 (my group), does a hit pay where the group's contribution is low (norm eroding)
# and cost where it is high? group-episode level: mean c at the block start (band) x punishment per member dealt then (band);
# outcomes: mean c per member over the rest of the game (rounds t+1..23), members at t+4, and pool per round t..23
import pandas as pd, numpy as np, pyarrow.parquet as pq, re
base = '/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-5-v4_sweep/'
cols = ['episode', 'participant_code', 'round_number', 'punishment', 'contribution', 'group_id', 'run', 'contribution_valid']
rows = []
for f in ['per_round_p1of2.parquet', 'per_round_p2of2.parquet']:
    d = pq.read_table(base + f, columns=cols, read_dictionary=['run']).to_pandas()
    for k in ['punishment', 'contribution', 'group_id', 'round_number']: d[k] = d[k].astype('int8')
    cats = d.run.cat.categories
    seat = np.array([0 if re.search(r'managed by pf-', r) else 1 for r in cats])
    opp = np.array([int(re.search(r'opponent_(\d)', r).group(1)) for r in cats])
    d['seat'] = seat[d.run.cat.codes.values]; d['opp'] = opp[d.run.cat.codes.values]
    d['run'] = d.run.cat.codes
    d = d.sort_values(['run', 'episode', 'participant_code', 'round_number'])
    n = len(d) // 192
    P = d.punishment.values.reshape(n, 8, 24).astype(float)
    C = d.contribution.values.reshape(n, 8, 24).astype(float)
    V = d.contribution_valid.values.reshape(n, 8, 24).astype(float)
    G = d.group_id.values.reshape(n, 8, 24)
    S = d.seat.values.reshape(n, 8, 24)[:, :, 0]; O = d.opp.values.reshape(n, 8, 24)[:, 0, 0]
    del d
    M = (G == S[:, :, None]).astype(float)
    nm = M.sum(1); cm = (C * M * V).sum(1) / np.maximum((M * V).sum(1), 1); pm = (P * M).sum(1) / np.maximum(nm, 1)
    pool = (1.6 * C * M).sum(1) - (P * M).sum(1)
    for t in [4, 8, 12]:
        rows.append(pd.DataFrame({'t': t, 'opp': O, 'cm': cm[:, t], 'pm': pm[:, t], 'n': nm[:, t],
                                  'c_rest': (cm[:, t + 1:] * (nm[:, t + 1:] > 0)).mean(1), 'n_next': nm[:, min(t + 4, 23)],
                                  'pool_rest': pool[:, t:].mean(1)}))
X = pd.concat(rows, ignore_index=True)
pd.set_option('display.width', 250)
X['cb'] = pd.cut(X.cm, [-1, 7, 9, 10.5, 12, 14, 20]); X['pb'] = pd.cut(X.pm, [-1, 0, 2, 5, 10, 30])
for v in ['pool_rest', 'c_rest', 'n_next']:
    for t in [4, 8, 12]:
        print(v, 'block start', t, '| rows: group mean c then, cols: punishment per member then')
        print(X[X.t == t].pivot_table(index='cb', columns='pb', values=v, aggfunc='mean', observed=True).round(2))
print(X[X.t == 8].pivot_table(index='cb', columns='pb', values='n', aggfunc='size', observed=True))
