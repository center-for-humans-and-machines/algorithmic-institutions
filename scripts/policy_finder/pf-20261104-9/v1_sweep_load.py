"""Load v1's sweep per-round data compactly (row group by row group) into a dense
numpy cache: arrays [point, episode, player, round] for c, valid, p, group."""
import pyarrow.parquet as pq, numpy as np, os, re
base = '/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261104-9_sweep/'
out = '/tmp/v1sweep.npz'
NP, NE = 256, 500
C = np.zeros((NP, NE, 8, 24), np.int8); V = np.zeros_like(C); P = np.zeros_like(C); G = np.zeros_like(C)
if not os.path.exists(out):
    for k in (1, 2):
        f = pq.ParquetFile(base + f'per_round_p{k}of2.parquet')
        for i in range(f.metadata.num_row_groups):
            t = f.read_row_group(i, columns=['episode', 'participant_code', 'round_number', 'punishment',
                                             'contribution', 'contribution_valid', 'group_id', 'run'])
            run = t.column('run').dictionary_encode()
            names = run.combine_chunks().dictionary.to_pylist()
            idx = run.combine_chunks().indices.to_numpy()
            ptmap = np.array([int(re.search(r'_s(\d+)_vs_ah', n).group(1)) for n in names])
            pt = ptmap[idx]
            pc = t.column('participant_code').to_pylist()
            pl = np.array([int(s.split('_')[0]) for s in pc])
            e = t.column('episode').to_numpy(); r = t.column('round_number').to_numpy()
            C[pt, e, pl, r] = t.column('contribution').to_numpy()
            V[pt, e, pl, r] = t.column('contribution_valid').to_numpy()
            P[pt, e, pl, r] = t.column('punishment').to_numpy()
            G[pt, e, pl, r] = t.column('group_id').to_numpy()
            print(k, i, flush=True)
    np.savez(out, C=C, V=V, P=P, G=G)
