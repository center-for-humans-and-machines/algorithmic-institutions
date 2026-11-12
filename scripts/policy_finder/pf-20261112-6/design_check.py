"""Design coverage: how many sweep points punish at all, and a toy run of the rule code."""
import numpy as np, pandas as pd, yaml, torch as th
from aimanager.manager.rule import sobol_design, load_rule
path='configs/managers/rule_based/pf-20261112-6.yml'
rule=yaml.safe_load(open(path))
pts=pd.DataFrame(sobol_design(rule,256))
pts['maxpun']=np.floor(np.minimum(pts.k*pts.thr,30))
print(pts.describe().round(2))
print('points that never punish (max pun 0 or t_start 23):',((pts.maxpun<1)|(pts.t_start>=23)).sum())
print('points punishing only rounds 20-22:',((pts.t_start>=20)&(pts.t_start<23)&(pts.maxpun>=1)).sum())
