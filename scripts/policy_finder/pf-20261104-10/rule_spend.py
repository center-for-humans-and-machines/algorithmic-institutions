"""Static check: the rule's punishment per member-round and share punished, applied to the
contributions of the zero-managed groups in the reference sims (no feedback), at a few param sets."""
import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d[d.run.str.contains('zero_vs_zero') & (d.contribution_valid==1)]
c = d.contribution.values; t = d.round_number.values
def rule(c0, slope, cap, t_stop, w_dec):
    p = np.minimum(slope*np.clip(c0-c,0,None), cap)
    p = p*np.where(t%4==3, w_dec, 1)*(t<=t_stop)
    return np.floor(np.clip(p,0,30))
for ps in [(9.5,0.55,6.5,13,0.5),(6,0.3,3,8,0),(12,1,4,19,0),(16,3,12,23,1),(3,0.1,1,4,0),(8,0.5,3,15,0)]:
    p = rule(*ps); act = t<=ps[3]
    print(ps, 'mean p/member-round %.2f' % p.mean(), 'in active rounds %.2f' % p[act].mean(), 'share punished %.2f' % (p>0).mean())
