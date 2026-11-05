# v2: turn v1's sweep per-round parquets into compact arrays [game, player, round]
# meta per game: point index, opponent, seat of my rule (0/1)
import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa, numpy as np, glob, sys
base = sys.argv[1] if len(sys.argv) > 1 else '/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-7_sweep/'
dest = sys.argv[2] if len(sys.argv) > 2 else '/tmp/v1sweep.npz'
out = {k: [] for k in ['C','P','G','V','point','opp','seat']}
for f in sorted(glob.glob(base + 'per_round*.parquet')):
    t = pq.read_table(f, columns=['episode','participant_code','round_number','punishment','contribution','contribution_valid','group_id','run'], read_dictionary=['run','participant_code'])
    run = t.column('run').combine_chunks(); pc_ = t.column('participant_code').combine_chunks()
    pl_map = np.array([int(s.split('_')[0]) for s in pc_.dictionary.to_pylist()])
    pl = pl_map[pc_.indices.to_numpy()].astype(np.int64)
    ri = run.indices.to_numpy().astype(np.int64)
    ep = t.column('episode').to_numpy().astype(np.int64); rd = t.column('round_number').to_numpy().astype(np.int64)
    names = run.dictionary.to_pylist()
    key = ((ri * 10000 + ep) * 8 + pl) * 24 + rd
    order = np.argsort(key, kind='stable'); del key
    n = len(order) // 192
    for k, col in [('C','contribution'),('P','punishment'),('G','group_id'),('V','contribution_valid')]:
        out[k].append(t.column(col).to_numpy()[order].reshape(n, 8, 24).astype(np.int8))
    rig = ri[order].reshape(n, 192)[:, 0]
    meta = []
    for name in names:
        a, b = name.split('managed by ')[1].split('_vs_')
        seat = 0 if a.startswith('pf-') else 1
        mine, opp = (a, b) if seat == 0 else (b, a)
        meta.append((int(mine.split('_s')[-1]) if '_s' in mine else -1, int(opp.split('_')[1]), seat))
    meta = np.array(meta)
    out['point'].append(meta[rig, 0]); out['opp'].append(meta[rig, 1]); out['seat'].append(meta[rig, 2])
    del t, order
np.savez_compressed(dest, **{k: np.concatenate(v) for k, v in out.items()})
print({k: np.concatenate(v).shape for k, v in out.items()})
