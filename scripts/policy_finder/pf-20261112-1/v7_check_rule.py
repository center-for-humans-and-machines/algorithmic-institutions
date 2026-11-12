# v7: hand-made checks of the v7 rule; on v5's games at round 7 (zero manager's state) gap_more 14 reproduces v6, and
# how often a second / third target is punished for several gap_more values.
import torch as th, numpy as np
from aimanager.manager.rule import load_rule, rule_inputs, run_rule
R='configs/managers/rule_based/pf-20261112-1-v7.yml'; R6='configs/managers/rule_based/pf-20261112-1-v6.yml'
def run(params, C, V, G, P, group, t, r=R):
    code,pr=load_rule(r,params,'exp2_diff')
    inp=rule_inputs(C[...,:t+1],V[...,:t+1],P[...,:t],G[...,:t+1],group,'exp2_diff')
    return run_rule(code,pr,inp)
base=dict(n_full=7,gap_full=3.0,gap_six=10.0,gap_more=5.0,bar=12,amount=20,n_push=3,last_dec=3)
C=th.full((1,8,24),10.); V=th.ones(1,8,24); P=th.zeros(1,8,24); G=th.zeros(1,8,24,dtype=th.long); G[:,4:,:4]=1
C[0,2,4:8]=0.; C[0,5,4:8]=4.; C[0,6,4:8]=8.
# gaps: p2 (50+4+8)/7=8.86; p5 (50+0+8)/7-4=4.29; p6 (50+0+4)/7-8=-0.29
print('8 members: p2 (gap 8.9) and p5 (gap 4.3) at gap_more 4:',run({**base,'gap_more':4.0},C,V,G,P,0,7)[0].tolist())
print('gap_more 5 -> p2 only:',run({**base,'gap_more':5.0},C,V,G,P,0,7)[0].tolist())
print('n_push 1, gap_more 4 -> p2 only:',run({**base,'n_push':1,'gap_more':4.0},C,V,G,P,0,7)[0].tolist())
print('t6, t3, t19:',run(base,C,V,G,P,0,6)[0].sum().item(),run(base,C,V,G,P,0,3)[0].sum().item(),run(base,C,V,G,P,0,19)[0].sum().item())
G6=G.clone(); G6[0,0,4:]=1; G6[0,1,4:]=1
# 6 members p2..p7: p2 (30+4+8)/5=8.4; p5 (30+0+8)/5-4=3.6
print('6 members gap_six 10 -> none (p2 gap 8.4):',run(base,C,V,G6,P,0,7)[0].tolist())
print('6 members gap_six 3 gap_more 3 -> p2, p5:',run({**base,'gap_six':3.0,'gap_more':3.0},C,V,G6,P,0,7)[0].tolist())
print('empty group 1:',run(base,C,V,G,P,1,7)[0].tolist())
d=np.load('/tmp/v6_v5_1.npz',allow_pickle=True)
Cc=th.tensor(d['C'][:20000].astype(np.float32)); Vv=th.tensor(d['V'][:20000].astype(np.float32)); Gg=th.tensor(d['G'][:20000].astype(np.int64)); s=d['seat'][:20000]
Pp=th.zeros_like(Cc)
for seat in (0,1):
    m=s==seat; a=Cc[m];b=Vv[m];g=Gg[m];p=Pp[m]
    p6=dict(n_full=7,gap_full=3.83,gap_six=13.35,bar=12,amount=20,n_push=1,last_dec=3)
    u6=run(p6,a,b,g,p,seat,7,R6); u7=run({**p6,'n_push':3,'gap_more':14.0},a,b,g,p,seat,7)
    print('seat',seat,'v7 gap_more 14 == v6 final:',bool((u6==u7).all()),'differing games',int((u6!=u7).any(1).sum()),'of',len(a),'push share %.3f'%(u6.sum(1)>0).float().mean())
    for gm in [3.,4.,5.,6.,8.,10.]:
        u=run({**p6,'n_push':3,'gap_more':gm},a,b,g,p,seat,7); k=(u>0).sum(1)
        print('  gap_more',gm,'games with 1/2/3 targets: %.3f %.3f %.3f'%((k==1).float().mean(),(k==2).float().mean(),(k==3).float().mean()))
