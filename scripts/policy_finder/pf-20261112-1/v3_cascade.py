# v3: in v2's sweep, players our rule punished in a decision round and who then moved to ah's group:
# what does ah charge them afterwards, how much do they contribute, and do they come back?
# Compared with movers our rule did not punish, at the same decision-round contribution.
import pandas as pd, numpy as np, glob, sys
sys.path.insert(0,'scripts/policy_finder/pf-20261112-1'); from v3_load import load
fs=sorted(glob.glob('plots/simulation/policy_finder/pf-20261112-1-v2_sweep/per_round_p*.parquet'))
rows=[]
for f in fs:
    D=load(f); P,C,G=D['P'],D['C'],D['G']; M=D['mine'][:,None]
    for dec in [3,7,11,15]:
        sel=(G[:,:,dec]==M)&(G[:,:,dec+1]!=M)
        back=(G[:,:,dec+5]==M)
        rows.append(pd.DataFrame({'dec':dec,'pdec':P[:,:,dec][sel],'cdec':C[:,:,dec][sel],
            'ah_p_first':P[:,:,dec+1][sel],'ah_p_next4':P[:,:,dec+1:dec+5].sum(-1)[sel],
            'c_next4':C[:,:,dec+1:dec+5].mean(-1)[sel],'back':back[sel]}))
    del D,P,C,G
r=pd.concat(rows)
r['pbin']=pd.cut(r.pdec,[-1,0,5,12,20,30])
r['cbin']=pd.cut(r.cdec,[-1,0,2,5,20])
print(len(r))
g=r.groupby(['cbin','pbin'],observed=True)
print(g[['ah_p_first','ah_p_next4','c_next4','back']].mean().round(2).assign(n=g.size()))
r.drop(columns=["pbin","cbin"]).to_parquet("/tmp/v3_movers.parquet")
