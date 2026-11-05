"""Per random_1 episode: fit punishment = a + b*c (linear) on valid rows; then relate b,a to the episode's contribution trend and payoff."""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
p=pd.read_csv('experiments/pilot_random1_player_round_slim.csv'); p=p[p.player_no_input==0]
r=p[p.experiment_name=='random_1']
rows=[]
for e,g in r.groupby('episode_id'):
    x=g.contribution.values; y=g.punishment.values
    if x.std()>0: b,a=np.polyfit(x,y,1)
    else: b,a=np.nan,y.mean()
    # zero-crossing: contribution where punishment stops
    pos=g[g.punishment>0]
    rows.append(dict(ep=e,a=a,b=b,cmax_pun=pos.contribution.max() if len(pos) else np.nan,
        pun=y.mean(), c_first=g[g.round_number<3].contribution.mean(), c_last=g[g.round_number>=g.round_number.max()-2].contribution.mean(),
        nr=g.round_number.max()+1, payoff=g.payoff.mean(), cg=g.contribution.mean()))
d=pd.DataFrame(rows); pd.set_option('display.width',200)
print(d.round(2).sort_values('pun').to_string())
print(d[['a','b','pun','c_first','c_last','payoff','cg']].corr().round(2))
# show example episode
print(r[r.episode_id==d.ep.iloc[0]][['round_number','contribution','punishment']].head(16).T)
