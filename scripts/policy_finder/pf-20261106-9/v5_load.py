# v5: load the v4 sweep (or, given files, the v4 final) per-round parquet into arrays [games, 8, 24].
# Rows are stored contiguous by run, episode, player, round (checked via round_number).
import pyarrow.parquet as pq, pyarrow.compute as pc, numpy as np, re, sys
base = 'plots/simulation/policy_finder/pf-20261106-9-v4_sweep/'
files = sys.argv[1:] or [base + 'per_round_p1of2.parquet', base + 'per_round_p2of2.parquet']
out = '/tmp/v4sweep.npz' if len(sys.argv) < 2 else '/tmp/v4final.npz'
C, P, V, G, OURS, PT, OPP = [], [], [], [], [], [], []
for fn in files:
    f = pq.ParquetFile(fn)
    for t in f.iter_batches(batch_size=192 * 4096, columns=['run', 'round_number', 'punishment', 'contribution', 'contribution_valid', 'group_id']):
        rn = t['round_number'].to_numpy()
        n = len(rn) // 192
        assert (rn.reshape(n, 8, 24) == np.arange(24)).all()
        C.append(t['contribution'].to_numpy().reshape(n, 8, 24).astype(np.int8))
        P.append(t['punishment'].to_numpy().reshape(n, 8, 24).astype(np.int8))
        V.append(t['contribution_valid'].to_numpy(zero_copy_only=False).reshape(n, 8, 24))
        G.append(t['group_id'].to_numpy().reshape(n, 8, 24).astype(np.int8))
        runs = t['run'].take(np.arange(0, len(rn), 192)).to_pylist()
        cache = {}
        for r in runs:
            if r not in cache:
                m = re.search(r'managed by (.*)$', r).group(1)
                s = re.search(r'_(s\d+|gp)_', m + '_').group(1)
                cache[r] = (0 if m.startswith('pf-') else 1, -1 if s == 'gp' else int(s[1:]),
                            int(re.search(r'opponent_(\d+)', m).group(1)))
            o, s, op = cache[r]
            OURS.append(o); PT.append(s); OPP.append(op)
        del t
np.savez(out, c=np.concatenate(C), p=np.concatenate(P), v=np.concatenate(V), g=np.concatenate(G),
         ours=np.array(OURS, np.int8), pt=np.array(PT, np.int16), opp=np.array(OPP, np.int8))
print('saved', sum(len(x) for x in C))
