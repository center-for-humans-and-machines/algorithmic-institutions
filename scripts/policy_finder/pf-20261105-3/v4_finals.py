"""v4: surrogate scores of every earlier final point (fresh-seed results known)."""
import sys, json, warnings; warnings.filterwarnings('ignore')
import torch as th
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261105-3')
from v4_sim import simulate, rule_fn, ROOT
from v4_surrogate import VERSIONS
th.set_num_threads(2)
for ver in VERSIONS:
    d = json.load(open(f'{ROOT}plots/simulation/policy_finder/{ver}_sweep/final/sweep.json'))
    for pnt in d['points']:
        s, se = simulate(rule_fn(f'configs/managers/rule_based/{ver}.yml', pnt['params']), n=4000, seed=7)
        print(f"{pnt['name']:28s} final {pnt['pool']:.1f} surrogate {s:.1f} ({se:.1f})", flush=True)
