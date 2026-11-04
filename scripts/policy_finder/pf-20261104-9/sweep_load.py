"""Load a sweep's per-round data into dense int8 arrays [point, episode, player, round]
(c, valid, p, group). Usage: sweep_load.py <version> <out.npz>. Group 0 is the rule."""
import pyarrow.parquet as pq, numpy as np, os, re, sys, glob
ver, out = sys.argv[1], sys.argv[2]
base = f'/opt/algorithmic-institutions/plots/simulation/policy_finder/{ver}_sweep/'
files = sorted(glob.glob(base + 'per_round_p*of*.parquet'))
NP, NE = 256, 500
C = np.zeros((NP, NE, 8, 24), np.int8); V = np.zeros_like(C); P = np.zeros_like(C); G = np.zeros_like(C)
for fn in files:
    f = pq.ParquetFile(fn)
    for i in range(f.metadata.num_row_groups):
        t = f.read_row_group(i, columns=['episode', 'participant_code', 'round_number', 'punishment',
                                         'contribution', 'contribution_valid', 'group_id', 'run'])
        run = t.column('run').combine_chunks().dictionary_encode()
        names = run.dictionary.to_pylist(); idx = run.indices.to_numpy()
        pt = np.array([int(re.search(r'_s(\d+)_vs_ah', n).group(1)) for n in names])[idx]
        pl = np.array([int(s.split('_')[0]) for s in t.column('participant_code').to_pylist()])
        e = t.column('episode').to_numpy(); r = t.column('round_number').to_numpy()
        C[pt, e, pl, r] = t.column('contribution').to_numpy()
        V[pt, e, pl, r] = t.column('contribution_valid').to_numpy()
        P[pt, e, pl, r] = t.column('punishment').to_numpy()
        G[pt, e, pl, r] = t.column('group_id').to_numpy()
        print(fn[-16:], i, flush=True)
np.savez(out, C=C, V=V, P=P, G=G)
