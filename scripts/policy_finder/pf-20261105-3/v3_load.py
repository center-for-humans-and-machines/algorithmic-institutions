"""v3: load a sweep (default: v2's) into arrays [run, episode, player, round], cached in /tmp.
participant_code is '<player>_<episode>' (see v2 entry 4)."""
import os, sys, json, numpy as np, pandas as pd
import pyarrow.parquet as pq
ROOT = '/opt/algorithmic-institutions/plots/simulation/policy_finder/'

def load(version='pf-20261105-3-v2'):
    cache = f'/tmp/v3_{version}.npz'
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    base = ROOT + version + '_sweep/'
    C, P, V, runs = [], [], [], []
    for f in sorted(x for x in os.listdir(base) if x.startswith('per_round')):
        pf = pq.ParquetFile(base + f)
        cols = ['run', 'episode', 'participant_code', 'round_number', 'punishment', 'contribution', 'contribution_valid']
        t = pf.read(columns=cols).to_pandas()
        t['run'] = t.run.astype('category')
        t['pt'] = t.participant_code.str.split('_').str[0].astype(np.int8)
        t.drop(columns='participant_code', inplace=True)
        for r, s in t.groupby('run', observed=True):
            s = s.sort_values(['episode', 'pt', 'round_number'], kind='mergesort')
            sh = (s.episode.nunique(), 4, 24)
            C.append(s.contribution.to_numpy(np.float32).reshape(sh))
            P.append(s.punishment.to_numpy(np.float32).reshape(sh))
            V.append(s.contribution_valid.to_numpy(bool).reshape(sh))
            runs.append(r)
        del t
    sw = json.load(open(base + 'sweep.json'))
    byname = {p['name']: p for p in sw['points']}
    names = [r.split(' managed by ')[1] for r in runs]
    pool = np.array([byname[n]['pool'] for n in names])
    out = dict(c=np.stack(C), p=np.stack(P), v=np.stack(V), runs=np.array(names), pool=pool,
               params=np.array([json.dumps(byname[n]['params']) for n in names]))
    np.savez(cache, **out)
    return out

if __name__ == '__main__':
    a = load(*sys.argv[1:])
    print(a['c'].shape, a['runs'][:3], a['pool'][:3])
