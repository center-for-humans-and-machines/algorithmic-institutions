"""v2: does the surrogate find v1's landscape peak (target 13.75, slope 2.2, horizon 12, no cap)?"""
import numpy as np
from v2_surrogate import Surrogate, v1_rule
S = Surrogate()
base = dict(target=13.75, slope=2.23, horizon=12, cap=30)
for k, vals in dict(target=[8, 10, 12, 13.75, 15.5, 17], slope=[1, 1.5, 2.23, 3, 5], horizon=[4, 8, 12, 16, 20, 24]).items():
    out = []
    for v in vals:
        pr = dict(base); pr[k] = v
        out.append(round(S.play(v1_rule(**pr), n=6000, seed=1)[0], 1))
    print(k, list(zip(vals, out)))
