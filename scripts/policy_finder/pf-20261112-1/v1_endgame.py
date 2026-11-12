# v2: does punishment after the last switch decision (rounds 20-22, groups fixed) pay in v1's sweep?
# Uses design points whose t_start >= 20 (identical to zero before) and per-episode group data.
import json, pandas as pd, numpy as np, glob
import pyarrow.parquet as pq
pts=pd.read_csv('/tmp/v1pts.csv')
late=set(pts.name[pts.t_start>=17])
base='plots/simulation/policy_finder/pf-20261112-1_sweep/'
out=[]
for f in sorted(glob.glob(base+'per_round_p*.parquet')):
    d=pq.read_table(f,columns=['episode','participant_code','round_number','punishment','contribution','agent_group','run'],read_dictionary=['run']).to_pandas()
    cats=pd.Series(d.run.cat.categories)
    pair=cats.str.split('by ').str[1]
    seat0=~pair.str.startswith('ah_vs')
    pt=pd.Series(np.where(seat0,pair.str.split('_vs_').str[0],pair.str.split('_vs_').str[1]))
    codes=d.run.cat.codes.values
    keep=pt.isin(late).values[codes]
    d=d[keep].copy(); codes=d.run.cat.codes.values
    d['pt']=pt.values[codes]; d['seat']=np.where(seat0.values[codes],0,1)
    d['mine']=(d.agent_group==d.seat)
    d=d[d.round_number>=16]
    d['pool']=1.6*d.contribution-d.punishment
    g=d.groupby(['pt','seat','episode','round_number','mine']).agg(pool=('pool','sum'),c=('contribution','mean'),p=('punishment','sum'),n=('pool','size')).reset_index()
    out.append(g)
g=pd.concat(out)
g=g.merge(pts[['name','t_start','thr','k','cap','dec_w']],left_on='pt',right_on='name')
m=g[g.mine]
# per point: punishment per member-round in rounds t_start..22, own pool per member in rounds 20-23
def summ(x):
    r=x.round_number
    return pd.Series(dict(p20=x.p[(r>=20)&(r<=22)].sum()/x.n[(r>=20)&(r<=22)].sum(),
        cpm20=(x.c*x.n)[r==20].sum()/x.n[r==20].sum(),
        cpm23=(x.c*x.n)[r==23].sum()/x.n[r==23].sum(),
        pool2023=x.pool[r>=20].sum()/x.n[r>=20].sum(),
        n20=x.n[r==20].mean()))
s=m.groupby(['pt','t_start']).apply(summ).reset_index()
s['gain_c']=s.cpm23-s.cpm20
print(s.sort_values('p20').round(2).to_string())
print(s[['p20','gain_c','pool2023']].corr().round(2))
import statsmodels.formula.api as smf
print(smf.ols('gain_c ~ p20',s).fit().summary2().tables[1].round(3))
print(smf.ols('pool2023 ~ p20',s).fit().summary2().tables[1].round(3))
