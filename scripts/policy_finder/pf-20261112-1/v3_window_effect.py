# v3: per (episode, window w = rounds 4w..4w+3) of v1's sweep: does the punishment the rule dealt in the window
# (split: non-decision rounds / decision round) change the lead over the rest of the game, and does it depend on
# the group's size at the window's start? Controls: the state at the window's first round before punishment.
import pandas as pd, numpy as np, glob, sys
sys.path.insert(0,'scripts/policy_finder/pf-20261112-1'); from v3_load import load
rows=[]
for f in sorted(glob.glob('plots/simulation/policy_finder/pf-20261112-1_sweep/per_round_p*.parquet')):
    D=load(f); P,C,G=(D[k].astype(np.float32) for k in 'PCG'); M=D['mine'][:,None,None]; pt=D['point']; del D
    mine=(G==M)
    pool_m=(1.6*C*mine-P*mine).sum(1); pool_o=(1.6*C*~mine-P*~mine).sum(1); lead=pool_m-pool_o
    pm=(P*mine).sum(1)  # my punishment per round [E,24]
    for w in range(1,6):
        s=4*w; e=min(s+4,24)
        n=mine[:,:,s].sum(1)
        c0=C[:,:,s]
        rows.append(pd.DataFrame({'pt':pt,'w':w,'Tnd':pm[:,s:e-1].sum(1) if w<5 else pm[:,s:23].sum(1),'Td':pm[:,e-1] if w<5 else pm[:,23],
            'n':n,'cm0':(c0*mine[:,:,s]).sum(1)/np.maximum(n,1),'co0':(c0*~mine[:,:,s]).sum(1)/np.maximum(8-n,1),
            'low0':((c0<=3)&mine[:,:,s]).sum(1),'lead_prev':lead[:,s-4:s].mean(1),'pm_prev':pm[:,s-4:s].sum(1),
            'lead_rest':lead[:,s:].sum(1)+pm[:,s:].sum(1),  # lead from the window on, with my own punishments added back
            'rest':24-s}))
    del P,C,G,mine
r=pd.concat(rows)
import statsmodels.formula.api as smf
r['ng']=pd.cut(r.n,[-1,3,5,6,8]).astype(str)
for w in range(1,6):
    x=r[r.w==w]
    m=smf.ols('lead_rest ~ Tnd:ng + Td:ng + C(n) + C(low0) + cm0 + co0 + lead_prev + pm_prev',x).fit()
    # net effect per point on the score's sum over rounds: coef - 1 (the point itself)
    print('window',w,'n rows',len(x),'mean Tnd %.1f Td %.1f'%(x.Tnd.mean(),x.Td.mean()))
    print('  gross return per point (net = coef - 1):',{k.replace('ng[','').replace(']',''):f'{v:.2f}({m.bse[k]:.2f})' for k,v in m.params.filter(like='T').items()})
