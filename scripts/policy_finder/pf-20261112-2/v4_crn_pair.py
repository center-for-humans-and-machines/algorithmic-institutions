"""v4: paired per-game comparison of fining points against never-punish on the same seeds
(slots 1 and 2: v1 = never punish)."""
import pandas as pd, numpy as np
G = pd.read_parquet('/tmp/v4_crn.parquet')
for slot in [1, 2]:
    a = G[(G.slot == slot) & (G.ver == 'pf-20261112-2')].set_index(['seat', 'episode'])
    b = G[(G.slot == slot) & (G.ver == 'pf-20261112-2-v2')].set_index(['seat', 'episode'])
    j = a[['lead', 'n_end']].join(b[['lead', 'spend', 't_first', 'n_end']], lsuffix='_0', rsuffix='_1')
    j['d'] = j.lead_1 - j.lead_0
    nf = j[j.t_first.isna()]
    print(f'slot {slot}: {b.pt.iloc[0]}; not fired {len(nf)}, max |d| there {nf.d.abs().max():.4f}')
    f = j[j.t_first.notna()]
    print(f'  fired {len(f)}: mean d {f.d.mean():.2f} +- {f.d.std()/np.sqrt(len(f)):.2f}; spend {f.spend.mean():.1f}; overall {j.d.mean():.2f} +- {j.d.std()/np.sqrt(len(j)):.2f}')
    f = f.assign(tb=pd.cut(f.t_first, [-1, 3, 7, 11, 15, 19, 23]))
    print(f.groupby('tb', observed=True).d.agg(['mean', 'sem', 'size']).round(2))
