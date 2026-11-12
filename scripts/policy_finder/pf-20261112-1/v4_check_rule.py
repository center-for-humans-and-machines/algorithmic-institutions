# v4 (copy of v3_check_rule.py on the v4 config): check the rule on a hand-made game, then count how often it would fire on zero-managed groups of the
# reference sims (rounds 7, 11, 15, 19; my group = the zero-managed one).
import torch as th, numpy as np, pandas as pd
from aimanager.manager.rule import load_rule, rule_inputs, run_rule
R='configs/managers/rule_based/pf-20261112-1-v4.yml'
def run(params, C, V, G, P, group, t):
    code,pr=load_rule(R,params,'exp2_diff')
    inp=rule_inputs(C[...,:t+1],V[...,:t+1],P[...,:t],G[...,:t+1],group,'exp2_diff')
    return run_rule(code,pr,inp)
# hand-made: 8 players in group 0 from round 4 on; player 2 gives 0,1,0,0 in window 1, player 5 gives 2 each round
C=th.full((1,8,24),10.); V=th.ones(1,8,24); P=th.zeros(1,8,24); G=th.zeros(1,8,24,dtype=th.long); G[:,4:,:4]=1
C[0,2,4:8]=th.tensor([0.,1,0,0]); C[0,5,4:8]=2.
base=dict(n_full=8,bar=2,amount=20,n_push=1,last_dec=3)
print('t7 push 1:',run(base,C,V,G,P,0,7)[0].tolist())
print('t7 push 2:',run({**base,'n_push':2},C,V,G,P,0,7)[0].tolist())
print('t6 (no decision):',run(base,C,V,G,P,0,6)[0].tolist())
print('t7 bar 0:',run({**base,'bar':0},C,V,G,P,0,7)[0].tolist())
print('t7 n_full 9:',run({**base,'n_full':9},C,V,G,P,0,7)[0].tolist())
print('t7 group 1 (empty, nobody):',run(base,C,V,G,P,1,7)[0].tolist())
print('t3:',run(base,C,V,G,P,0,3)[0].tolist())
G2=G.clone(); G2[0,0,4:]=1
print('t7 with 7 members, n_full 8:',run(base,C,V,G2,P,0,7)[0].tolist(),' n_full 7:',run({**base,'n_full':7},C,V,G2,P,0,7)[0].tolist())
print('t19 last_dec 3 / 4:',run(base,C,V,G,P,0,19)[0].tolist(), run({**base,'last_dec':4},C,V,G,P,0,19)[0].tolist())
# frequency on the reference sims' zero-managed groups
d=pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet',columns=['run','episode','participant_code','round_number','contribution','group_id','contribution_valid'])
d=d[d.run.str.contains('zero_vs_ah|ah_vs_zero')]
d=d.sort_values(['run','episode','participant_code','round_number'])
for run_name,x in d.groupby('run'):
    mine=0 if 'zero_vs_ah' in run_name else 1
    n=len(x)//(24*8)
    Cc=th.tensor(x.contribution.values.reshape(n,8,24),dtype=th.float); Vv=th.tensor(x.contribution_valid.values.reshape(n,8,24),dtype=th.float)
    Gg=th.tensor(x.group_id.values.reshape(n,8,24)); Pp=th.zeros(n,8,24)
    for prm in [dict(n_full=8,bar=1,amount=20,n_push=1,last_dec=3),dict(n_full=7,bar=3,amount=20,n_push=2,last_dec=4)]:
        out=[]
        for t in [7,11,15,19]:
            pu=run(prm,Cc,Vv,Gg,Pp,mine,t)
            out.append(f't{t}: games with a push {(pu.sum(1)>0).float().mean():.3f}, n8 {((Gg[:,:,t]==mine).sum(1)==8).float().mean():.3f}')
        print(run_name.split('by ')[-1], prm, '; '.join(out))
