"""v3: load sim per_round parquet files into dense arrays [game, round, player], memory-light.
Returns dict with c, p (float32), g (group_id int8), v (valid bool), cg (common_good float32),
run (name per game), episode."""
import pyarrow.parquet as pq, pyarrow.compute as pc, numpy as np


def load(path, runs_filter=None):
    f = pq.ParquetFile(path)
    cols = {k: [] for k in ['run', 'episode', 'pid', 'round', 'p', 'cg', 'c', 'v', 'g']}
    names = {}
    for i in range(f.metadata.num_row_groups):
        t = f.read_row_group(i, columns=['run', 'episode', 'participant_code', 'round_number', 'punishment',
                                         'common_good', 'contribution', 'contribution_valid', 'group_id'])
        run = t.column('run').to_numpy(zero_copy_only=False)
        u, inv = np.unique(run, return_inverse=True)
        ids = np.array([names.setdefault(x, len(names)) for x in u])[inv].astype(np.int16)
        pcode = t.column('participant_code').to_numpy(zero_copy_only=False)
        uu, ii = np.unique(pcode, return_inverse=True)
        pid = np.array([int(x.split('_')[0]) for x in uu], dtype=np.int8)[ii]
        keep = np.ones(len(ids), bool)
        if runs_filter is not None:
            okn = {names[x] for x in u if runs_filter in x}
            keep = np.isin(ids, list(okn))
        cols['run'].append(ids[keep]); cols['pid'].append(pid[keep])
        cols['episode'].append(t.column('episode').to_numpy()[keep].astype(np.int32))
        cols['round'].append(t.column('round_number').to_numpy()[keep].astype(np.int8))
        cols['p'].append(t.column('punishment').to_numpy()[keep].astype(np.float32))
        cols['cg'].append(t.column('common_good').to_numpy()[keep].astype(np.float32))
        cols['c'].append(t.column('contribution').to_numpy()[keep].astype(np.float32))
        cols['v'].append(t.column('contribution_valid').to_numpy(zero_copy_only=False)[keep])
        cols['g'].append(t.column('group_id').to_numpy()[keep].astype(np.int8))
        del t
    C = {k: np.concatenate(v) for k, v in cols.items()}
    o = np.lexsort((C['pid'], C['round'], C['episode'], C['run']))
    n = len(o) // 192
    sh = (n, 24, 8)
    inv_names = {v: k for k, v in names.items()}
    out = {k: C[k][o].reshape(sh) for k in ['c', 'p', 'g', 'v', 'cg']}
    out['run'] = np.array([inv_names[x] for x in C['run'][o].reshape(sh)[:, 0, 0]])
    out['episode'] = C['episode'][o].reshape(sh)[:, 0, 0]
    return out
