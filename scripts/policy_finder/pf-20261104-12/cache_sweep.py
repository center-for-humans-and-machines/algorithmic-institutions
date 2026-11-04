"""Turn a sweep's per_round parquet files into arrays [point, episode, player, round]
(c, p, g, v as int8) plus the points' params, saved to /tmp/<tag>.npz."""
import pyarrow.parquet as pq, sys, glob, json, re, numpy as np, pandas as pd
sweep = sys.argv[1]; tag = sys.argv[2]
fs = sorted(glob.glob(f'{sweep}/per_round*.parquet'))
S = json.load(open(f'{sweep}/sweep.json'))
names = [p['name'] for p in S['points']]
arrs = {k: [] for k in 'cpgv'}; runs = []
for f in fs:
    d = pq.read_table(f, columns=['episode','participant_code','round_number','punishment','contribution','agent_group','contribution_valid','run'],
                      read_dictionary=['run','participant_code']).to_pandas()
    for k in ('episode','round_number','punishment','contribution','agent_group'): d[k] = d[k].astype(np.int16)
    cats = d.participant_code.cat.categories
    d['pid'] = np.array([int(x.split('_')[0]) for x in cats], dtype=np.int8)[d.participant_code.cat.codes.to_numpy()]
    d.drop(columns='participant_code', inplace=True)
    d = d.sort_values(['run','episode','pid','round_number'])
    R = list(d.run.cat.categories[np.sort(pd.unique(d.run.cat.codes))]); E = d.episode.nunique()
    sh = (len(R), E, 8, 24)
    arrs['c'].append(d.contribution.to_numpy().astype(np.int8).reshape(sh))
    arrs['p'].append(d.punishment.to_numpy().astype(np.int8).reshape(sh))
    arrs['g'].append(d.agent_group.to_numpy().astype(np.int8).reshape(sh))
    arrs['v'].append(d.contribution_valid.to_numpy().astype(np.int8).reshape(sh))
    runs += [re.search(r'by (\S+)_vs_ah', r).group(1) for r in R]
    del d
P = {p['name']: p for p in S['points']}
keys = list(S['points'][0]['params'])
params = np.array([[P[r]['params'][k] for k in keys] for r in runs])
pool = np.array([P[r]['pool'] for r in runs])
np.savez(f'/tmp/{tag}.npz', **{k: np.concatenate(v) for k, v in arrs.items()}, params=params, keys=keys, pool=pool, runs=runs)
print(len(runs), keys)
