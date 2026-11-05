# All 36 finals: block-adjusted score and per-phase punishment / contribution profile.
import pandas as pd, numpy as np, json
base='plots/simulation/policy_finder/'
vs=['pf-20261105-7','pf-20261105-7-v2','pf-20261105-7-v3','pf-20261105-7-v4','pf-20261105-7-v5','pf-20261105-7-v6']
rows=[];per={}
for vi,v in enumerate(vs):
    d=pd.read_parquet(base+v+'_sweep/final/per_round.parquet')
    d['run']=d.run.str.replace('ah exp1 managed by ','')
    js=json.load(open(base+v+'_sweep/final/sweep.json'))
    prm={p['name']:p['params'] for p in js['points']}
    for k,r in enumerate(d.run.unique()):
        x=d[(d.run==r)&d.contribution_valid]
        ep=x.groupby(['episode','round_number']).payoff.mean().groupby('episode').sum()
        per[r]=ep
        c=x.groupby('round_number').contribution.mean(); p=x.groupby('round_number').punishment.mean()
        rows.append(dict(run=r,ver=vi+1,block=k,score=ep.mean(),p0=p[0],p1=p[1],p2_6=p[2:7].sum(),p7_11=p[7:12].sum(),p12_15=p[12:16].sum(),p16=p[16:].sum(),ptot=p.sum(),
                         c1=c[1],c3=c[3],c7=c[7],c12=c[12],c16=c[16],c23=c[23],params=prm[r]))
T=pd.DataFrame(rows)
T['adj']=T.score-T.groupby('block').score.transform('mean')
pd.set_option('display.width',250)
print(T.drop(columns='params').sort_values(['block','adj'],ascending=[True,False]).round(2).to_string())
T.to_pickle('/tmp/pf7_finals.pkl')
pd.DataFrame(per).to_pickle('/tmp/pf7_final_eps.pkl')
