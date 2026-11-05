"""v3: turn the sweep / reference per_round parquets into arrays [episodes, 4, 24] (c, p, valid)
plus the run name per episode, saved to /tmp/pf3/<tag>.npz for the surrogate."""
import numpy as np, pandas as pd, glob, sys
srcs = {
    'v1': sorted(glob.glob('plots/simulation/policy_finder/pf-20261105-11_sweep/per_round_p*.parquet')),
    'v2': sorted(glob.glob('plots/simulation/policy_finder/pf-20261105-11-v2_sweep/per_round_p*.parquet')),
    'v1f': ['plots/simulation/policy_finder/pf-20261105-11_sweep/final/per_round.parquet'],
    'v2f': ['plots/simulation/policy_finder/pf-20261105-11-v2_sweep/final/per_round.parquet'],
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
    np.savez_compressed(f'/tmp/pf3/{tag}.npz', c=np.concatenate(C), p=np.concatenate(P), v=np.concatenate(V),
                        run=np.concatenate(R))
    print(tag, sum(len(x) for x in C))
