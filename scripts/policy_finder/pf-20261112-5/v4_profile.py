"""Per-round profile of the v3 final peak vs `zero` (reference sims, both seats): my members, my members' mean
contribution, punishment dealt per member, the other group's members and contribution, and per-round lead."""
import numpy as np, pandas as pd
d = pd.read_parquet('/opt/algorithmic-institutions/plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet',
                    columns=['run', 'episode', 'participant_code', 'round_number', 'punishment', 'contribution', 'agent_group', 'contribution_valid'])
d = d[d.run.str.contains('zero_vs_ah|ah_vs_zero')]
def prof(c, p, m, v):
    pool = 1.6 * c * v - p
    out = pd.DataFrame(dict(
        mem=m.sum(1).mean(0), c_mine=(c * m).sum((0, 1)) / m.sum((0, 1)), p_mine=(p * m).sum((0, 1)) / m.sum((0, 1)),
        c_oth=(c * (1 - m)).sum((0, 1)) / (1 - m).sum((0, 1)), p_oth=(p * (1 - m)).sum((0, 1)) / (1 - m).sum((0, 1)),
        lead=((pool * m).sum(1) - (pool * (1 - m)).sum(1)).mean(0)))
    return out
arrs = {k: [] for k in 'cpmv'}
for r, x in d.groupby('run'):
    seat = 0 if 'by zero_vs' in r else 1
    x = x.sort_values(['episode', 'participant_code', 'round_number'])
    E = x.episode.nunique(); sh = (E, 8, 24)
    arrs['c'].append(x.contribution.values.reshape(sh)); arrs['p'].append(x.punishment.values.reshape(sh))
    arrs['v'].append(x.contribution_valid.values.reshape(sh)); arrs['m'].append((x.agent_group.values.reshape(sh) == seat))
Z = prof(*[np.concatenate(arrs[k]).astype(np.float32) for k in 'cpmv'])
z = np.load('/tmp/pf/v3f.npz'); s = z['point'] == -1
G = prof(*[z[k][s].astype(np.float32) for k in ['c', 'p', 'mine', 'v']])
pd.set_option('display.width', 200)
print(pd.concat({'zero': Z, 'v3gp': G}, axis=1).round(2))
print('means', Z.mean().round(2).to_dict(), G.mean().round(2).to_dict())
