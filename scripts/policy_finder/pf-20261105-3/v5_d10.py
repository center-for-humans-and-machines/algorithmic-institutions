"""v5: profiles of d10 and pmax in the final form (cref 20), 4 seeds x 3000 games."""
import sys, warnings, numpy as np, torch as th
warnings.filterwarnings('ignore')
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261105-3')
from v5_tail import sc, doses
b = sc(d10=7.0); print('base d10 7 pmax 30', b.mean().round(2), doses(d10=7.0), flush=True)
for kw in [dict(d10=5.5), dict(d10=6.0), dict(d10=8.0), dict(d10=9.5), dict(d10=7.0, pmax=25), dict(d10=7.0, pmax=36), dict(d10=7.0, pmax=42)]:
    d = sc(**kw) - b
    print(kw, f'{d.mean():+.2f}', np.round(d, 1), doses(**kw), flush=True)
