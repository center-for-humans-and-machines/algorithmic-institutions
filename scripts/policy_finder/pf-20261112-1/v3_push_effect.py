# v3: per (episode, decision) of v2's sweep: does the punishment the rule dealt in the decision round
# change the rest-of-game lead (own pool - ah's pool per round, rounds dec+1..23)? Controls: the state at the
# decision (sizes, both groups' window contributions and punishments, the lead so far, the count of low givers).
# The punishment varies with the design point (bar, amount, first_dec), which is drawn independently of the game.
import pandas as pd, numpy as np, glob, sys, json
sys.path.insert(0,'scripts/policy_finder/pf-20261112-1'); from v3_load import load
rows=[]
for f in sorted(glob.glob('plots/simulation/policy_finder/pf-20261112-1-v2_sweep/per_round_p*.parquet')):
    D=load(f); P,C,G=(D[k].astype(np.float32) for k in 'PCG'); M=D['mine'][:,None,None]; pt=D['point']; del D
    mine=(G==M)
    pool_m=(1.6*C*mine-P*mine).sum(1); pool_o=(1.6*C*~mine-P*~mine).sum(1)   # [E,24]
    lead=pool_m-pool_o
    for dec in [3,7,11,15,19]:
        w=slice(dec-3,dec+1); mw=mine[:,:,w]
        n=mw[:,:,-1].sum(1)
        cw=C[:,:,w].mean(-1)
        low=((cw<=1)&mw[:,:,-1]).sum(1); low3=((cw<=3)&mw[:,:,-1]).sum(1); low5=((cw<=5)&mw[:,:,-1]).sum(1)
        Tn=(P[:,:,dec]*mw[:,:,-1]).sum(1)
        rows.append(pd.DataFrame({'point':pt,'dec':dec,'T':(P[:,:,dec]*mw[:,:,-1]).sum(1),'Tearly':(P[:,:,dec-3:dec]*mw[:,:,:-1]).sum((1,2)),
            'n':n,'low':low,'low3':low3,'low5':low5,'cmw':(C[:,:,w]*mw).sum((1,2))/np.maximum(mw.sum((1,2)),1),
            'cow':(C[:,:,w]*~mw).sum((1,2))/np.maximum((~mw).sum((1,2)),1),
            'pow':(P[:,:,w]*~mw).sum((1,2))/np.maximum((~mw).sum((1,2)),1),
            'lead_before':lead[:,:dec+1].mean(1),'lead_win':(lead[:,w].sum(1)+Tn)/4,
            'lead_after':lead[:,dec+1:].mean(1),'lead_next':lead[:,dec+1:dec+5].mean(1),
            'n_after':mine[:,:,dec+1].sum(1),'own_after':pool_m[:,dec+1:].mean(1),'oth_after':pool_o[:,dec+1:].mean(1),
            'n_end':mine[:,:,23].sum(1),'cmin':np.where(mw[:,:,-1],cw,99).min(1),'n_low1':((cw<=1)&mw[:,:,-1]).sum(1)}))
    del P,C,G,mine
r=pd.concat(rows); r['pc']=r.point.astype('category').cat.codes; r.drop(columns='point').to_parquet('/tmp/v3_push.parquet')
import statsmodels.formula.api as smf
for dec in [7,11,15,19]:
    x=r[r.dec==dec]
    for y in ['lead_next','lead_after','n_after']:
        m=smf.ols(f'{y} ~ T + Tearly + C(n) + C(low) + C(low3) + C(low5) + cmw + cow + pow + lead_win + lead_before',x).fit(cov_type='cluster',cov_kwds={'groups':x.pc})
        print(dec,y,'T coef %.3f se %.3f'%(m.params['T'],m.bse['T']), 'mean T %.2f, share T>0 %.3f'%(x['T'].mean(),(x['T']>0).mean()))

# score-scale net effect of one point dealt at decision d: benefit coef*(23-d)/24 minus the point itself /24
print('net per point on the score: coef*(23-d)/24 - 1/24')
