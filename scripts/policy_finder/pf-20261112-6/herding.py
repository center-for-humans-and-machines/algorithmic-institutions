"""v3: herding and the fate of low contributors in the reference zero-vs-zero sims (no punishment anywhere).
(a) next-round contribution of a member on its own contribution, its group mates' mean and the other group's mean;
(b) what a low contributor (c <= 3 for 2 rounds) does after a decision: stays vs moves, contributions after."""
import numpy as np, pandas as pd
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet',
                    columns=['run', 'episode', 'round_number', 'participant_code', 'group_id', 'agent_group', 'contribution', 'punishment'])
print(d.run.unique())
for runkey in ['zero_vs_zero', 'zero_vs_ah']:
    s = d[d.run.str.contains(runkey)].sort_values(['run', 'episode', 'participant_code', 'round_number'])
    n = len(s) // 192
    c = s.contribution.to_numpy().reshape(n, 8, 24).astype(float)
    g = s.group_id.to_numpy().reshape(n, 8, 24)
    p = s.punishment.to_numpy().reshape(n, 8, 24).astype(float)
    X, y = [], []
    for t in range(23):
        if t % 4 == 3: continue
        same = g[:, :, t][:, :, None] == g[:, :, t][:, None, :]
        eye = np.eye(8, dtype=bool)[None]
        mates = same & ~eye
        nm = mates.sum(2)
        mm = np.where(nm > 0, (mates * c[:, None, :, t]).sum(2) / np.maximum(nm, 1), np.nan)
        oth = ~same
        no = oth.sum(2)
        om = np.where(no > 0, (oth * c[:, None, :, t]).sum(2) / np.maximum(no, 1), np.nan)
        ok = ~np.isnan(mm) & ~np.isnan(om) & (p[:, :, t] == 0)
        X.append(np.column_stack([np.ones(ok.sum()), c[:, :, t][ok], mm[ok], om[ok], nm[ok] + 1]))
        y.append(c[:, :, t + 1][ok])
    X = np.concatenate(X); y = np.concatenate(y)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    print(runkey, 'c_next = %.2f + %.3f own + %.3f mates_mean + %.3f other_mean + %.3f size' % tuple(b), 'n', len(y))
    # low contributors at decisions
    for dec in [3, 7, 11, 15]:
        low = (c[:, :, dec] <= 3) & (c[:, :, dec - 1] <= 3)
        moved = g[:, :, dec + 1] != g[:, :, dec]
        after = c[:, :, dec + 1:dec + 5].mean(2)
        print(f'  dec {dec}: low share {low.mean():.3f}; P(move|low) {moved[low].mean():.2f} vs {moved[~low].mean():.2f};'
              f' next-block c: low stayed {after[low & ~moved].mean():.2f}, low moved {after[low & moved].mean():.2f}, others {after[~low].mean():.2f}')
