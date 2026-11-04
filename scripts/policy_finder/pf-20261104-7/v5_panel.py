"""v5: load a sweep's per_round parquet files into arrays [episodes, 8 players, 24 rounds]
(c, p, valid, in my group = agent_group 0) plus the design point index; cached in /tmp."""
import os, sys, numpy as np, pandas as pd, pyarrow.parquet as pq

def load(version):
    cache = f'/tmp/v5_panel_{version}.npz'
    if os.path.exists(cache):
        z = np.load(cache); return {k: z[k] for k in z.files}
    B = f'plots/simulation/policy_finder/{version}_sweep/'
    files = sorted(f for f in os.listdir(B) if f.startswith('per_round'))
    out = {k: [] for k in ['c', 'p', 'v', 'g', 'point']}
    for fn in files:
        f = pq.ParquetFile(B + fn)
        parts = []
        for i in range(f.num_row_groups):
            d = f.read_row_group(i, columns=['run', 'episode', 'participant_code', 'round_number', 'punishment',
                                              'contribution', 'agent_group', 'contribution_valid']).to_pandas()
            parts.append(pd.DataFrame({
                'point': d.run.str.extract(r'_s(\d+)_vs')[0].astype(np.int16).values,
                'episode': d.episode.values.astype(np.int32),
                'pl': d.participant_code.str.split('_').str[0].astype(np.int8).values,
                'r': d.round_number.values.astype(np.int8), 'c': d.contribution.values.astype(np.int8),
                'p': d.punishment.values.astype(np.int8), 'g': (d.agent_group.values == 0),
                'v': d.contribution_valid.values}))
        d = pd.concat(parts).sort_values(['point', 'episode', 'pl', 'r'])
        n = len(d) // (8 * 24)
        sh = (n, 8, 24)
        for k in ['c', 'p', 'v', 'g']:
            out[k].append(d[k].values.reshape(sh))
        out['point'].append(d.point.values.reshape(sh)[:, 0, 0])
    r = {k: np.concatenate(v) for k, v in out.items()}
    np.savez(cache, **r)
    return r

if __name__ == '__main__':
    r = load(sys.argv[1])
    print({k: v.shape for k, v in r.items()})
    print('mean my-group size by round', r['g'].sum(1).mean(0).round(2))
