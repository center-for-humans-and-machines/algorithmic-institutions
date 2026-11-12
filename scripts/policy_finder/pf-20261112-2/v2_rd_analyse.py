import pandas as pd, numpy as np
x = pd.read_parquet('/tmp/v2_rd.parquet')
x['tot'] = x.V - x.p_now + x.oV  # value to my pool from round t on: this player's net future + fine now + others' future
x['fb'] = pd.cut(x.fine, [0, 1, 2, 4, 8, 30]); x['tb'] = pd.cut(x.t, [-1, 3, 7, 11, 15, 19, 22]); x['cb'] = pd.cut(x.cstar, [-1, 2, 5, 8, 11, 14, 20])
x['phase'] = x.t % 4
def rd(g):
    m = g.groupby('off')[['p_now', 'V', 'c1', 'stay', 'oV', 'odc', 'nfut', 'tot']].mean()
    if len(m) < 3: return None
    jump = (m.loc[0] - m.loc[1]) - (m.loc[1] - m.loc[2])
    jump['n'] = len(g)
    return jump
for by in ['fb', 'tb', 'cb', 'phase']:
    print('by', by)
    print(x.groupby(by, observed=True).apply(rd).round(3))
print('all'); print(rd(x).round(3))
