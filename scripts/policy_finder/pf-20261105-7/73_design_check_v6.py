"""Score v4's, v5's and v6's Sobol designs on the pooled real-data GP of 72 (mid_fine coordinates)."""
import sys, yaml, numpy as np, runpy
sys.argv=[sys.argv[0]]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    g=runpy.run_path('scripts/policy_finder/pf-20261105-7/72_pooled_gp_mid.py')
pred,conv=g['pred'],g['conv']
sys.path.insert(0,'src')
from aimanager.manager.rule import sobol_design
for v in ['pf-20261105-7-v4','pf-20261105-7-v5','pf-20261105-7-v6']:
    pts=sobol_design(yaml.safe_load(open(f'configs/managers/rule_based/{v}.yml')),256)
    X=np.array([conv({k:float(x) for k,x in p.items()}) for p in pts])
    s=pred(X)
    print(v,'mean %.1f'%s.mean(),'pct 25/50/75/90/max',np.percentile(s,[25,50,75,90,100]).round(1),'share >=654: %.2f'%(s>=654).mean(),'>=655.5: %.2f'%(s>=655.5).mean())
