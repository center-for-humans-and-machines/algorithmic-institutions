"""Build [game, player, round] arrays for c, p, valid for the versions given on argv
(sweeps + finals). Saves /tmp/pf7/<version>_<sweep|final>.npz (as 27, generalised)."""
import pandas as pd, numpy as np, glob, sys
cols=['run','episode','participant_code','round_number','contribution','punishment','contribution_valid']
for v in sys.argv[1:]:
    B=f'plots/simulation/policy_finder/{v}_sweep/'
    for tag,files in [('sweep',sorted(glob.glob(B+'per_round_p*.parquet'))+sorted(glob.glob(B+'per_round.parquet'))),('final',[B+'final/per_round.parquet'])]:
        C=[];P=[];V=[];R=[]
        for f in files:
            df=pd.read_parquet(f,columns=cols)
            df['run']=df.run.astype('category')
            df=df.sort_values(['run','episode','participant_code','round_number'])
            n=len(df)//96
            C.append(df.contribution.values.reshape(n,4,24).astype(np.int8))
            P.append(df.punishment.values.reshape(n,4,24).astype(np.int8))
            V.append(df.contribution_valid.values.reshape(n,4,24).astype(np.int8))
            R.append(np.array([r.split('managed by ')[1] for r in df.run.values[::96].astype(str)]))
            del df
        np.savez(f'/tmp/pf7/{v}_{tag}.npz',C=np.concatenate(C),P=np.concatenate(P),V=np.concatenate(V),R=np.concatenate(R))
        print(v,tag,sum(len(x) for x in C))
