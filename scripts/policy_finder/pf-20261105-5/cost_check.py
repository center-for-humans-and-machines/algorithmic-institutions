"""Rough cost of the rule if contributions followed the zero / humanlike sims (no feedback)."""
import pandas as pd, numpy as np
df = pd.read_parquet("plots/simulation/29_EXP1_reference/per_round.parquet")
df = df[df.contribution_valid]
for mgr in ("zero", "humanlike"):
    s = df[df.run.str.endswith(mgr)]
    for T, b, k, stop in [(12, 2, 0.3, 18), (15, 1, 0.2, 16), (8, 4, 0.5, 20), (20, 3, 1.0, 24)]:
        c = s.contribution.values
        p = np.where((c < T) & (s.round_number.values < stop), np.floor(np.clip(b + k * (T - c), 0, 30)), 0)
        print(mgr, (T, b, k, stop), "mean p per player-round", round(p.mean(), 2), "cost per game", round(2 * p.mean() * 24, 1))
