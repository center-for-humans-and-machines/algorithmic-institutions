import sys; sys.path.insert(0,'scripts/policy_finder/pf-20261106-3')
from load import load
import pandas as pd, statsmodels.formula.api as smf
df = load().sort_values(['game','player_id','round_number'])
g=df.groupby(['game','player_id'])
df['pc']=g.contribution.shift(1); df['pp']=g.punishment.shift(1); df['pg']=g.group_id.shift(1)
d=df.dropna(subset=['pc']); d=d[(d.player_no_input==0)&(d.pg==d.group_id)&(d.pc<10)]
for lo,hi in [(1,8),(8,16),(16,24)]:
    s=d[(d.round_number>=lo)&(d.round_number<hi)]
    m=smf.ols('contribution ~ pc + pp',s).fit()
    print(lo,hi,len(s),m.params.round(3).to_dict(), 'se pp',round(m.bse.pp,3))
