# v7: zero-twin index (copy of v6_twins.py, rebuilt after /tmp was cleared). Saved to /tmp/v6_Z.parquet.
# (key, seat, ver, part, row, lead, po, pt, mem). Saved to /tmp/v6_Z.parquet.
import numpy as np, pandas as pd
fr=[]
for k in ['v1','v2','v3','v4']:
    for part in '12':
        d=np.load(f'/tmp/v6_{k}_{part}.npz',allow_pickle=True)
        f={a:d[a] for a in ['key','seat','pun','lead','po','pt','mem']}
        m=~f['pun']
        fr.append(pd.DataFrame(dict(key=f['key'][m],seat=f['seat'][m],ver=k,part=part,row=np.where(m)[0],lead=f['lead'][m],po=f['po'][m],pt=f['pt'][m],mem=f['mem'][m])))
Z=pd.concat(fr).drop_duplicates(['key','seat']); Z.to_parquet('/tmp/v6_Z.parquet'); print(len(Z))
