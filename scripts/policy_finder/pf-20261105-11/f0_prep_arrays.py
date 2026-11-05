"""v6 (copy of e0 adding v5 and its final): per_round parquets -> arrays [episodes, 4, 24] (c, p, valid) + run name, to /tmp/pf6/<tag>.npz."""
import numpy as np, pandas as pd, glob, os
os.makedirs('/tmp/pf6', exist_ok=True)
B = 'plots/simulation/policy_finder/'
srcs = {
    'v2': sorted(glob.glob(B + 'pf-20261105-11-v2_sweep/per_round_p*.parquet')),
    'v3': sorted(glob.glob(B + 'pf-20261105-11-v3_sweep/per_round_p*.parquet')),
    'v4': sorted(glob.glob(B + 'pf-20261105-11-v4_sweep/per_round_p*.parquet')),
    'v4f': [B + 'pf-20261105-11-v4_sweep/final/per_round.parquet'],
    'v3f': [B + 'pf-20261105-11-v3_sweep/final/per_round.parquet'],
    'v5': sorted(glob.glob(B + 'pf-20261105-11-v5_sweep/per_round_p*.parquet')),
    'v5f': [B + 'pf-20261105-11-v5_sweep/final/per_round.parquet'],
    'v1': sorted(glob.glob(B + 'pf-20261105-11_sweep/per_round_p*.parquet')),
    'ref': ['plots/simulation/29_EXP1_reference/per_round.parquet'],
}
for tag, files in srcs.items():
    C, P, V, R = [], [], [], []
    for f in files:
        d = pd.read_parquet(f, columns=['run', 'episode', 'participant_code', 'round_number', 'punishment',
                                        'contribution', 'contribution_valid'])
        d['pl'] = d.participant_code.str.split('_').str[0].astype(int)
        d = d.sort_values(['run', 'episode', 'pl', 'round_number'])
        n = len(d) // 96
        C.append(d.contribution.values.reshape(n, 4, 24).astype(np.int8))
        P.append(d.punishment.values.reshape(n, 4, 24).astype(np.int8))
        V.append(d.contribution_valid.values.reshape(n, 4, 24))
        R.append(d.run.values[::96])
        del d
    np.savez_compressed(f'/tmp/pf6/{tag}.npz', c=np.concatenate(C), p=np.concatenate(P), v=np.concatenate(V),
                        run=np.concatenate(R))
    print(tag, sum(len(x) for x in C), flush=True)
