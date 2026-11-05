# v2: turn the v1 sweep's per-round parquet into arrays [game, player, round] (cached in /tmp)
import pyarrow.parquet as pq, numpy as np, pandas as pd, json, os, re
D = '/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-11_sweep/'
OUT = '/tmp/pf11_sweep.npz'
def build():
    C=[];P=[];G=[];V=[];R=[];E=[]
    for part in ['per_round_p1of2.parquet','per_round_p2of2.parquet']:
        f = pq.ParquetFile(D+part)
        # the file is ordered by run, episode, player, round: 192 rows per game
        for b in f.iter_batches(batch_size=192*2000, columns=['episode','round_number','punishment','contribution','agent_group','contribution_valid','run']):
            n = b.num_rows//192
            col = lambda k: b.column(k).to_numpy(zero_copy_only=False)
            assert (col('round_number').reshape(n,8,24) == np.arange(24)).all()
            C.append(col('contribution').reshape(n,8,24).astype(np.int8))
            P.append(col('punishment').reshape(n,8,24).astype(np.int8))
            G.append(col('agent_group').reshape(n,8,24).astype(np.int8))
            V.append(col('contribution_valid').reshape(n,8,24))
            R.append(np.asarray(b.column('run').take(np.arange(0,b.num_rows,192)).to_pylist()))
            E.append(col('episode')[::192].astype(np.int32))
    np.savez(OUT, c=np.concatenate(C), p=np.concatenate(P), g=np.concatenate(G), v=np.concatenate(V), run=np.concatenate(R), ep=np.concatenate(E))
def load():
    if not os.path.exists(OUT): build()
    z = np.load(OUT, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    run = d['run']
    pair = np.array([r.split('managed by ')[-1] for r in run])
    mine_g0 = np.array([x.startswith('pf-') for x in pair])
    pt = np.array([int(re.search(r'_s(\d+)', x).group(1)) for x in pair])
    opp = np.array([int(re.search(r'opponent_(\d)', x).group(1)) for x in pair])
    d.update(mine_g0=mine_g0, pt=pt, opp=opp)
    S = json.load(open(D+'sweep.json'))
    d['params'] = pd.DataFrame([dict(**x['params'], pool=x['pool']) for x in S['points']])
    d['params']['pt'] = [int(re.search(r'_s(\d+)', x['name']).group(1)) for x in S['points']]
    return d
if __name__ == '__main__':
    d = load(); print({k: getattr(v,'shape',None) for k,v in d.items()})
    # sanity: group sizes
    print(np.bincount(d['opp']), d['mine_g0'].mean())
