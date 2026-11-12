# v6: hand-made game checks of the v6 rule, and how often it fires on v5's games at round 7 (the zero manager's state
# there), compared with the v6_freq7.py shares; also checks gap_full 0 reproduces v5 (bar 12, amount 20) at round 7.
import torch as th, numpy as np
from aimanager.manager.rule import load_rule, rule_inputs, run_rule
R='configs/managers/rule_based/pf-20261112-1-v6.yml'; R5='configs/managers/rule_based/pf-20261112-1-v5.yml'
def run(params, C, V, G, P, group, t, r=R):
    code,pr=load_rule(r,params,'exp2_diff')
    inp=rule_inputs(C[...,:t+1],V[...,:t+1],P[...,:t],G[...,:t+1],group,'exp2_diff')
    return run_rule(code,pr,inp)
base=dict(n_full=7,gap_full=4.0,gap_six=9.0,bar=12,amount=20,n_push=1,last_dec=3)
C=th.full((1,8,24),10.); V=th.ones(1,8,24); P=th.zeros(1,8,24); G=th.zeros(1,8,24,dtype=th.long); G[:,4:,:4]=1
C[0,2,4:8]=th.tensor([0.,1,0,0]); C[0,5,4:8]=4.
print('8 members, p2 mean .25 gap 8.9 -> p2:',run(base,C,V,G,P,0,7)[0].tolist())
print('t6, t3, t19:',run(base,C,V,G,P,0,6)[0].sum().item(),run(base,C,V,G,P,0,3)[0].sum().item(),run(base,C,V,G,P,0,19)[0].sum().item())
print('gap_full 9.5 -> none:',run({**base,'gap_full':9.5},C,V,G,P,0,7)[0].tolist())
G6=G.clone(); G6[0,0,4:]=1; G6[0,1,4:]=1
print('6 members, gap ~8.6 < gap_six 9 -> none:',run(base,C,V,G6,P,0,7)[0].tolist(),' gap_six 8 -> p2:',run({**base,'gap_six':8.0},C,V,G6,P,0,7)[0].tolist())
G5=G6.clone(); G5[0,3,4:]=1
print('5 members -> none:',run({**base,'gap_six':0.0,'gap_full':0.0},C,V,G5,P,0,7)[0].tolist())
print('empty group 1:',run(base,C,V,G,P,1,7)[0].tolist())
V2=V.clone(); V2[0,2,7]=0
print('target timed out this round -> next lowest p5 (gap ~6):',run(base,C,V2,G,P,0,7)[0].tolist())
# v5 data at round 7
for part in '1':
    d=np.load(f'/tmp/v6_v5_{part}.npz',allow_pickle=True)
    Cc=th.tensor(d['C'][:20000].astype(np.float32)); Vv=th.tensor(d['V'][:20000].astype(np.float32)); Gg=th.tensor(d['G'][:20000].astype(np.int64)); s=d['seat'][:20000]
    Pp=th.zeros_like(Cc)
    for seat in (0,1):
        m=s==seat; a=Cc[m];b=Vv[m];g=Gg[m];p=Pp[m]
        for prm in [dict(base,gap_full=0.0,gap_six=20.0),base,dict(base,gap_six=6.0)]:
            u=run(prm,a,b,g,p,seat,7)
            print('seat',seat,prm['gap_full'],prm['gap_six'],'share with a push at round 7 %.3f'%(u.sum(1)>0).float().mean())
        u5=run(dict(n_full=7,bar=12,amount=20,n_push=1,last_dec=3),a,b,g,p,seat,7,R5)
        u6=run(dict(base,gap_full=0.0,gap_six=20.0),a,b,g,p,seat,7)
        print('  v6(gap 0, six off) == v5(bar12, amount20):',bool((u5==u6).all()), 'differing games', int((u5!=u6).any(1).sum()))
