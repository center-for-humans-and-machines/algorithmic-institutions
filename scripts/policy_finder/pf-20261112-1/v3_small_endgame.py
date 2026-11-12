# v3: check exploration-6's surprise without regression: v1 points that never punish before round 17
# (t_start >= 17), episodes by my group's size in rounds 20-23 (fixed: no switch after 19); compare points that
# punish in 20-22 (t_start <= 20) with points that do not (t_start >= 21), by the rule's harshness.
import pandas as pd, numpy as np, glob, sys, json
sys.path.insert(0,'scripts/policy_finder/pf-20261112-1'); from v3_load import load
sw=json.load(open('plots/simulation/policy_finder/pf-20261112-1_sweep/sweep.json'))
par={p['name']:p['params'] for p in sw['points']}
rows=[]
for f in sorted(glob.glob('plots/simulation/policy_finder/pf-20261112-1_sweep/per_round_p*.parquet')):
    D=load(f); P,C,G=(D[k].astype(np.float32) for k in 'PCG'); M=D['mine'][:,None,None]; pt=D['point']; del D
    mine=(G==M)
    n=mine[:,:,20].sum(1)
    pm=(P*mine)[:,:,20:].sum((1,2)); cm=(C*mine)[:,:,20:].sum((1,2)); co=(C*~mine)[:,:,20:].sum((1,2)); po=(P*~mine)[:,:,20:].sum((1,2))
    c19=(C*mine)[:,:,19].sum(1)
    rows.append(pd.DataFrame({'pt':pt,'n':n,'pm':pm,'cm':cm,'co':co,'po':po,'c19':c19}))
r=pd.concat(rows)
for k in ['thr','k','cap','t_start']: r[k]=r.pt.map(lambda x:par[x][k])
r=r[r.t_start>=17]
r['active']=pd.cut(r.t_start,[16,19,20,22]).astype(str)
r['own']=1.6*r.cm-r.pm; r['oth']=1.6*r.co-r.po
r['ng']=pd.cut(r.n,[-1,0,1,2,3,5,8])
g=r.groupby(['ng','active'],observed=True)
pd.set_option('display.width',200); print(g[['pm','cm','c19','own','co','po','oth']].mean().round(1).assign(N=g.size()))
