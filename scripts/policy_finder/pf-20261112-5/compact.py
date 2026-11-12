"""Compact a sweep's (or the reference sims') per-round data into arrays [game, player, round], from the rule's
seat's view. Usage: compact.py <sweep_dir> <out.npz> <param names comma-separated>
Saves c, p, v (valid), g (group), mine (1 if in the rule's group), point, seat, params."""
import pandas as pd, numpy as np, json, glob, re, sys
import pyarrow.parquet as pq
base, outf, names = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
sw = json.load(open(base + '/sweep.json'))
par = {}
for p in sw['points']:
    mm = re.search(r'_s(\d+)$', p['name'])
    par[int(mm.group(1)) if mm else -1] = p['params']
out = {k: [] for k in ['c', 'p', 'v', 'g', 'point', 'seat']}
files = sorted(glob.glob(base + '/per_round_p*of*.parquet')) or [base + '/per_round.parquet']
for f in files:
    t = pq.read_table(f, columns=['run', 'episode', 'participant_code', 'round_number', 'punishment',
                                  'contribution', 'agent_group', 'contribution_valid'],
                      read_dictionary=['run', 'participant_code'])
    d = t.to_pandas(); del t
    d['pl'] = np.array([int(s.split('_')[0]) for s in d.participant_code.cat.categories], np.int8)[d.participant_code.cat.codes.values]
    d.drop(columns='participant_code', inplace=True)
    for c_ in ['episode', 'round_number']: d[c_] = d[c_].astype(np.int16)
    for c_ in ['punishment', 'contribution', 'agent_group']: d[c_] = d[c_].astype(np.int8)
    for r, x in d.groupby('run', observed=True):
        m = re.search(r'managed by (.*)$', r).group(1)
        seat = 0 if m.startswith('pf-') else 1
        mm = re.search(r'_s(\d+)', m)
        pt = int(mm.group(1)) if mm else -1
        x = x.sort_values(['episode', 'pl', 'round_number'])
        E = x.episode.nunique(); sh = (E, 8, 24)
        out['c'].append(x.contribution.values.reshape(sh).astype(np.int8))
        out['p'].append(x.punishment.values.reshape(sh).astype(np.int8))
        out['v'].append(x.contribution_valid.values.reshape(sh).astype(np.int8))
        out['g'].append(x.agent_group.values.reshape(sh).astype(np.int8))
        out['point'].append(np.full(E, pt, np.int16)); out['seat'].append(np.full(E, seat, np.int8))
    del d
o = {k: np.concatenate(v) for k, v in out.items()}
o['mine'] = (o['g'] == o['seat'][:, None, None]).astype(np.int8)
ks = sorted(par)
P = np.array([[par[i][k] for k in names] for i in ks]); idx = np.array(ks)
np.savez_compressed(outf, params=P, param_idx=idx, **o)
print({k: v.shape for k, v in o.items()})
