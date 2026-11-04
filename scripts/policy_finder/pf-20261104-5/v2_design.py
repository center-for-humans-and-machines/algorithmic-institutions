"""v2: how heavy are the design's punishments? Full-strength punishment of a zero
contributor and of a 7-contributor, min(cap, k*(thr-c)), over v1's and v2's Sobol designs."""
import numpy as np, pandas as pd
from aimanager.manager.rule import read_rule, sobol_design
for path in ['configs/managers/rule_based/pf-20261104-5.yml', 'configs/managers/rule_based/pf-20261104-5-v2.yml']:
    rule = read_rule(path)
    rule = rule[0] if isinstance(rule, tuple) else rule
    d = pd.DataFrame(sobol_design(rule, 256))
    p0 = np.minimum(d.cap, d.k * d.thr)
    p7 = np.minimum(d.cap, d.k * np.maximum(d.thr - 7, 0))
    print(path, 'n', len(d))
    print('  punishment of a 0-contributor: quantiles', np.percentile(p0, [10, 25, 50, 75, 90]).round(1))
    print('  punishment of a 7-contributor: quantiles', np.percentile(p7, [10, 25, 50, 75, 90]).round(1))
    print('  share with 0-contributor punishment >= 10:', (p0 >= 10).mean().round(2))
