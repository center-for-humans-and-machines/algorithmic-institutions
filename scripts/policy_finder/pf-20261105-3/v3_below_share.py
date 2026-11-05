"""v3: under v2's final rule, share of valid players below each maintenance standard in rounds 9-19 (what a nudge would cost)."""
import pandas as pd
f = pd.read_parquet('plots/simulation/policy_finder/pf-20261105-3-v2_sweep/final/per_round.parquet')
f = f[f.run.str.endswith('v2_gp') & f.contribution_valid & f.round_number.between(9, 19)]
for m in range(13, 21):
    s = (f.contribution < m).mean()
    print(f'mthr {m}: share below {s:.2f}; a 2-point nudge costs {2 * 2 * s:.2f} mean payoff per round')
