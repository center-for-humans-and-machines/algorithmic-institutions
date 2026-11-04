"""v1 sweep: regress c_late, members, m_r4 and pool on the phase doses (per group-round punishment)."""
import pandas as pd, numpy as np, statsmodels.formula.api as smf
F = pd.read_csv('scripts/policy_finder/pf-20261104-11/v1_sweep_features.csv', index_col=0)
F.columns = [c.replace('-', '_') for c in F.columns]
F['P_early'] = (F.P_r0 + 2 * F.P_r1_2) / 3
F['P_mid'] = (4 * F.P_r4_7 + 4 * F.P_r8_11) / 8
F['P_late'] = (8 * F.P_r12_19 + 4 * F.P_r20_23) / 12
for y in ['c_late', 'members', 'm_r4', 'pool']:
    m = smf.ols(f'{y} ~ np.sqrt(P_early) + P_early + P_r3 + P_mid + P_late', F).fit()
    print(y, m.params.round(3).to_dict(), 'R2', round(m.rsquared, 2))
# binned by early dose
F['eb'] = pd.cut(F.P_early, [-1, 1, 3, 6, 10, 15, 30])
print(F.groupby('eb', observed=True)[['pool', 'c_late', 'members', 'm_r4', 'P_mid', 'P_late']].agg(['mean', 'count']).round(2).to_string())
F['lb'] = pd.cut(F.P_late, [-1, 0.05, 0.5, 1.5, 5])
print(F[F.P_early > 3].groupby('lb', observed=True)[['pool', 'c_late', 'members', 'P_early']].agg(['mean', 'count']).round(2).to_string())
