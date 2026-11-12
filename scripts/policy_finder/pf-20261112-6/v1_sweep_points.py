"""v2: per design point of the v1 sweep, where its punishment went (phase, target) and how that relates to score."""
import pandas as pd, numpy as np, json, glob
B='plots/simulation/policy_finder/pf-20261112-6_sweep/'
pts={p['name']:p for p in json.load(open(B+'sweep.json'))['points']}
rows=[]
for f in sorted(glob.glob(B+'per_round_p*.parquet')):
    import pyarrow.parquet as pq
    d=pq.read_table(f,columns=['episode','round_number','punishment','contribution','agent_group','run'],read_dictionary=['run']).to_pandas()
    cats=pd.Series(d.run.cat.categories)
    tail=cats.str.split('managed by ').str[-1]
    o0=~tail.str.startswith('ah_vs_')
    nm=np.where(o0,tail.str.split('_vs_').str[0],tail.str.split('_vs_').str[-1])
    codes=d.run.cat.codes.values
    own0=o0.values[codes]
    d['name']=pd.Categorical.from_codes(pd.Categorical(nm).codes[codes],pd.Categorical(nm).categories)
    d['mine']=np.where(own0,d.agent_group==0,d.agent_group==1)
    m=d[d.mine]
    ph=np.select([m.round_number>=20, m.round_number%4==0, m.round_number%4==1],['late','b0','b1'],'b23')
    m=m.assign(ph=ph, low=(m.contribution<5))
    # punishment per (episode*seat) summed over rounds, by phase
    neps=m.groupby('name').apply(lambda x: x[['run','episode']].drop_duplicates().shape[0])
    P=m.groupby(['name','ph']).punishment.sum().unstack(fill_value=0).div(neps,axis=0)
    PL=m[~m.low].groupby('name').punishment.sum().reindex(P.index,fill_value=0)/neps
    P['on_coop']=PL
    rows.append(P)
P=pd.concat(rows).fillna(0)
P['pool']=[pts[n]['pool'] for n in P.index]; P['members']=[pts[n]['members'] for n in P.index]
P['own']=[pts[n]['own'] for n in P.index]; P['other']=[pts[n]['other'] for n in P.index]
print(P.describe().round(2))
import numpy.linalg as la
X=np.column_stack([np.ones(len(P)),P[['b0','b1','b23','late','on_coop']].values])
for y in ['pool','members','own','other']:
    b=la.lstsq(X,P[y].values,rcond=None)[0]; print(y,'const,b0,b1,b23,late,on_coop per pt-sum:',b.round(4))
z=P[(P[['b0','b1','b23','late']].sum(1)==0)]; print('zero points',len(z),z.pool.mean().round(2))
lo=P[(P[['b0','b1','b23']].sum(1)==0)&(P.late>0)]; print('late-only points',len(lo),lo.pool.mean().round(2), lo.late.mean().round(1))
P.to_csv('scripts/policy_finder/pf-20261112-6/v1_points.csv')
