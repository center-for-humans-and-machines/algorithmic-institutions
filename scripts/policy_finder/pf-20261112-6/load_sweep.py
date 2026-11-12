"""v2: load a sweep's per_round parquet(s) into arrays [N, 8, 24] (N = run x episode), with own-group flags.
The files are stored ordered by (run, episode, player, round), checked in v2 exploration 2."""
import numpy as np, pyarrow.parquet as pq, pyarrow.compute as pc, glob

def load(base, files=None):
    out = {k: [] for k in ['c', 'p', 'g', 'v', 'mine_g', 'name']}
    for f in files or sorted(glob.glob(base + 'per_round*.parquet')):
        t = pq.read_table(f, columns=['run', 'round_number', 'punishment', 'contribution', 'agent_group',
                                     'contribution_valid'], read_dictionary=['run'])
        r = t.column('round_number').to_numpy()
        assert (r.reshape(-1, 24) == np.arange(24)).all()
        n = len(r) // 192; sh = (n, 8, 24)
        out['c'].append(t.column('contribution').to_numpy().reshape(sh).astype(np.float32))
        out['p'].append(t.column('punishment').to_numpy().reshape(sh).astype(np.float32))
        out['g'].append(t.column('agent_group').to_numpy().reshape(sh).astype(np.int8))
        out['v'].append(t.column('contribution_valid').to_numpy().reshape(sh).astype(np.int8))
        run = t.column('run').combine_chunks()
        idx = run.indices.to_numpy()[::192]; dic = run.dictionary.to_pylist()
        tails = [s.split('managed by ')[-1] for s in dic]
        o0 = np.array([not s.startswith('ah_vs_') for s in tails])
        nm = np.array([s.split('_vs_')[0] if not s.startswith('ah_vs_') else s.split('_vs_')[-1] for s in tails])
        out['mine_g'].append(np.where(o0[idx], 0, 1).astype(np.int8))
        out['name'].append(nm[idx])
        del t
    return {k: np.concatenate(v) for k, v in out.items()}
