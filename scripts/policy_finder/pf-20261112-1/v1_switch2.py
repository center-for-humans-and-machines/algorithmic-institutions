# v2: follow-up on v1_switch.py's saved decision table (/tmp/v1switch.parquet):
# do members react to the punishment of OTHERS earlier in the window (not only through that round's share)?
import pandas as pd, numpy as np
import statsmodels.formula.api as smf
x=pd.read_parquet('/tmp/v1switch.parquet')
# group punishment over the window per member-round, excluding the decision round and own
x['oth_pearly']=(x.gpwin*x.gn*4 - x.gp - x.pearly)/((x.gn-1).clip(lower=1)*3)
r=x[x.mine==1]
m=smf.logit('sw ~ pdec + pearly + oth_pdec + oth_pearly + share + oshare + gn + contribution + C(win)',r).fit(disp=0)
print(m.summary2().tables[1].round(4))
a=x[x.mine==0]
m2=smf.logit('sw ~ pdec + pearly + share + oshare + gn + on + contribution + C(win)',a).fit(disp=0)
print('ah group members'); print(m2.summary2().tables[1].round(4))
print('mean switch rate by window, rule group / ah group');print(x.groupby(['win','mine']).sw.mean().unstack().round(3))
