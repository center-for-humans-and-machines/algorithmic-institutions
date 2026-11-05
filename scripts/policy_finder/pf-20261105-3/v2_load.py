"""v2: load version 1's sweep into arrays [run, episode, player, round] (cached in /tmp)."""
import os, numpy as np, pandas as pd
import pyarrow.parquet as pq
CACHE = '/tmp/v2_sweep_arrays_fixed.npz'
base = '/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261105-3_sweep/'

def load():
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}
    C, P, V, runs = [], [], [], []
    for f in ['per_round_p1of2.parquet', 'per_round_p2of2.parquet']:
        pf = pq.ParquetFile(base + f)
        cols = ['run', 'episode', 'participant_code', 'round_number', 'punishment', 'contribution', 'contribution_valid']
        t = pf.read(columns=cols).to_pandas()
        t['run'] = t.run.astype('category')
        t['pt'] = t.participant_code.str.split('_').str[0].astype(np.int8)  # '<player>_<episode>'
        t.drop(columns='participant_code', inplace=True)
        for r, s in t.groupby('run', observed=True):
            s = s.sort_values(['episode', 'pt', 'round_number'], kind='mergesort')
            ne = s.episode.nunique()
            sh = (ne, 4, 24)
            C.append(s.contribution.to_numpy(np.float32).reshape(sh))
            P.append(s.punishment.to_numpy(np.float32).reshape(sh))
            V.append(s.contribution_valid.to_numpy(bool).reshape(sh))
            runs.append(r)
        del t
    out = dict(c=np.stack(C), p=np.stack(P), v=np.stack(V), runs=np.array(runs))
    np.savez(CACHE, **out)
    return out

if __name__ == '__main__':
    a = load()
    print(a['c'].shape, a['runs'][:3])
