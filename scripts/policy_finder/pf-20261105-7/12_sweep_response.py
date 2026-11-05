"""Response of clones to punishment, from v1's sweep (random deficit rules).
Builds arrays [game, player, round] for c, p, valid and saves to /tmp/pf7_arr.npz."""
import pandas as pd, numpy as np, json
B='plots/simulation/policy_finder/pf-20261105-7_sweep/'
cols=['run','episode','participant_code','round_number','contribution','punishment','contribution_valid']
C=[];P=[];V=[];R=[]
for i in (1,2):
    df=pd.read_parquet(B+f'per_round_p{i}of2.parquet',columns=cols)
    df['run']=df.run.astype('category')
    df=df.sort_values(['run','episode','participant_code','round_number'])
    n=len(df)//96
    C.append(df.contribution.values.reshape(n,4,24).astype(np.float32))
    P.append(df.punishment.values.reshape(n,4,24).astype(np.float32))
    V.append(df.contribution_valid.values.reshape(n,4,24).astype(np.float32))
    R.append(df.run.values[::96].astype(str))
    del df
C=np.concatenate(C);P=np.concatenate(P);V=np.concatenate(V);R=np.concatenate(R)
sw=json.load(open(B+'sweep.json'))
pm={p['name']:p['params'] for p in sw['points']}
names=[r.split('managed by ')[1] for r in R]
par=np.array([[pm[n][k] for k in ('target','slope','window','taper')] for n in names])
np.savez('/tmp/pf7_arr.npz',C=C,P=P,V=V,par=par)
print(C.shape)
