# v2: how often does the free-rider dump fire? Window-mean contribution of members of zero-managed groups
# at decision rounds (reference sims), and a unit check of the rule's code on a hand-made game.
import pandas as pd, numpy as np, torch as th
from aimanager.manager.rule import load_rule
d=pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
pair=d.run.str.split('by ').str[1]
mgr=np.where(d.agent_group==0,pair.str.split('_vs_').str[0],pair.str.split('_vs_').str[1])
d=d[(mgr=='zero')&(d.round_number<20)].copy()
d['win']=d.round_number//4
d=d.sort_values(['run','episode','participant_code','round_number'])
w=d.groupby(['run','episode','participant_code','win']).agg(cm=('contribution','mean'),g=('agent_group','last'),gfirst=('agent_group','first')).reset_index()
w=w[w.g==w.gfirst]
for b in range(0,9):
    print('bar',b,'share of members marked: %.3f'%(w.cm<=b).mean(), ' by window', (w.cm<=b).groupby(w.win).mean().round(3).tolist())

# unit check: 1 game, round t=7 (decision of window 1); player 0 gives 0,0 in window 0 then 1,1,2,0 in window 1
import yaml
cfg=yaml.safe_load(open('configs/managers/rule_based/pf-20261112-1-v2.yml'))
def run(params,t,c,ing):
    sc={'th':th,'c':c[...,:t+1],'valid':th.ones_like(c[...,:t+1]),'in_group':ing[...,:t+1],'p':th.zeros_like(c[...,:t]),'t':th.tensor(float(t))}
    sc.update({k:th.tensor(float(v)) for k,v in params.items()})
    exec(cfg['code'],sc); return sc['punishment']
c=th.full((1,8,24),10.); c[0,0,4:8]=th.tensor([1.,1.,2.,0.]); c[0,0,:4]=0; c[0,1,4:8]=3.
ing=th.zeros(1,8,24); ing[0,:4]=1
for t in [3,6,7,19,23]:
    print(t, run({'bar':1,'amount':12,'first_dec':0},t,c,ing)[0].tolist(), run({'bar':1,'amount':12,'first_dec':2},t,c,ing)[0].tolist())
