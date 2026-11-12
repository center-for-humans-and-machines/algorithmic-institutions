"""Rough per-capita punishment spend of the rule over its sweep ranges, applied to the
contributions of zero-managed groups in the reference sims (static, no response)."""
import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d[(d.run.str.contains('zero_vs_zero')) & (d.contribution_valid == 1)]
c = d.contribution.values; t = d.round_number.values
def spend(thr, base, slope, t_stop):
    s = np.clip(thr - c, 0, None)
    p = np.floor(np.clip((base + slope * s) * (s > 0) * (t <= t_stop), 0, 30))
    return p.mean(), (p > 0).mean()
for thr in [3, 6, 9, 12, 16]:
    for base, slope in [(0, 0.25), (1, 0), (1, 0.25), (2, 0.5), (4, 1.5)]:
        m, f = spend(thr, base, slope, 23)
        print(f'thr={thr:2d} base={base} slope={slope:4}: spend/capita {m:5.2f}, share punished {f:.2f}')
