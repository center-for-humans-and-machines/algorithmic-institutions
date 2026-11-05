# v3: arrays [game, player, round] from any sweep dir (cached in /tmp), generalising b2_arrays
import pyarrow.parquet as pq, numpy as np, pandas as pd, json, os, re, glob, sys
def build(D, OUT):
    C=[];P=[];G=[];V=[];R=[];E=[]
    for part in sorted(glob.glob(D+'per_round*.parquet')):
        f = pq.ParquetFile(part)
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
def load(D='/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-11-v2_sweep/', tag='v2'):
    OUT = f'/tmp/pf11_{tag}.npz'
    if not os.path.exists(OUT): build(D, OUT)
    z = np.load(OUT, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    pair = np.array([r.split('managed by ')[-1] for r in d['run']])
    d['mine_g0'] = np.array([x.startswith('pf-') for x in pair])
    d['pt'] = np.array([int(m.group(1)) if (m:=re.search(r'_s(\d+)', x)) else -1 for x in pair])
    d['opp'] = np.array([int(re.search(r'opponent_(\d)', x).group(1)) for x in pair])
    # my group's label: agent_group values; group of seat 0 is the smaller label
    labs = np.unique(d['g']); g0 = labs.min()
    d['mine'] = np.where(d['mine_g0'][:,None,None], d['g']==g0, d['g']!=g0)
    S = json.load(open(D+'sweep.json'))
    d['params'] = pd.DataFrame([dict(**x['params'], pool=x['pool'], name=x['name']) for x in S['points']])
    return d
if __name__ == '__main__':
    d = load(*sys.argv[1:]); print({k: getattr(v,'shape',None) for k,v in d.items()}, np.unique(d['g']))
    print(np.bincount(d['opp']), d['mine_g0'].mean(), d['mine'].mean())
