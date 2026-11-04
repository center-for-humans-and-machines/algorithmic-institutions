"""v2: v1's sweep as dense arrays [point, episode, player, round] -> /tmp/v2_arrays.npz.
Our group (the rule's) is agent_group 0 in every run ('<rule>_vs_ah')."""
import json, glob
import numpy as np, pyarrow.parquet as pq
s = json.load(open('plots/simulation/policy_finder/pf-20261104-1_sweep/sweep.json'))
names = [p['name'] for p in s['points']]
P, E, N, R = len(names), 500, 8, 24
C = np.zeros((P, E, N, R), np.int8); PU = np.zeros_like(C); G = np.zeros_like(C); V = np.zeros_like(C)
for f in sorted(glob.glob('plots/simulation/policy_finder/pf-20261104-1_sweep/per_round_p*.parquet')):
    pf = pq.ParquetFile(f)
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=['episode', 'participant_code', 'round_number', 'punishment', 'contribution', 'agent_group', 'contribution_valid', 'run'])
        run = t.column('run').dictionary_encode().combine_chunks()
        dic = [x.as_py().split('managed by ')[1].split('_vs_')[0] for x in run.dictionary]
        pidx = np.array([names.index(x) for x in dic])[run.indices.to_numpy()]
        ep = t.column('episode').to_numpy()
        pc = t.column('participant_code').to_numpy(zero_copy_only=False)
        # player index: participant codes within an episode -> 0..7 by sorted order
        key = np.char.add(np.char.add(pidx.astype(str), '_'), ep.astype(str))
        import pandas as pd
        df = pd.DataFrame({'k': key, 'pc': pc})
        pl = df.groupby('k').pc.rank(method='dense').to_numpy().astype(int) - 1
        r = t.column('round_number').to_numpy()
        C[pidx, ep, pl, r] = t.column('contribution').to_numpy()
        PU[pidx, ep, pl, r] = t.column('punishment').to_numpy()
        G[pidx, ep, pl, r] = t.column('agent_group').to_numpy()
        V[pidx, ep, pl, r] = t.column('contribution_valid').to_numpy().astype(np.int8)
        del t, df
    print(f, flush=True)
prm = np.array([[p['params'][k] for k in ['slope', 'tol', 'cap', 't_stop', 'pre_switch']] for p in s['points']])
pool = np.array([p['pool'] for p in s['points']])
np.savez_compressed('/tmp/v2_arrays.npz', C=C, P=PU, G=G, V=V, prm=prm, pool=pool)
print(np.bincount(G.ravel()))
