"""v7: per_round parquets -> arrays [episodes, 4, 24] (c, p, valid) + run name, to /tmp/pf7/<tag>.npz.
Also params per run from sweep.json."""
import numpy as np, pandas as pd, glob, os, json
os.makedirs('/tmp/pf7', exist_ok=True)
B = 'plots/simulation/policy_finder/'
srcs = {f'v{k}': sorted(glob.glob(B + f'pf-20261105-11-v{k}_sweep/per_round_p*.parquet')) for k in [2, 3, 4, 5, 6]}
srcs['v4f'] = [B + 'pf-20261105-11-v4_sweep/final/per_round.parquet']
srcs['ref'] = ['plots/simulation/29_EXP1_reference/per_round.parquet']
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
    np.savez_compressed(f'/tmp/pf7/{tag}.npz', c=np.concatenate(C), p=np.concatenate(P), v=np.concatenate(V),
                        run=np.concatenate(R))
    print(tag, sum(len(x) for x in C), flush=True)
