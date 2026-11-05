"""v2: the clones' response to punishment, estimated on version 1's sweep (256 rules,
so p varies across rules for the same contribution). Change of contribution k rounds
later by contribution bin x punishment bin, and by round."""
import numpy as np, pandas as pd
from v2_load import load
a = load(); c, p, v = a['c'], a['p'], a['v']

def table(k, rounds=range(0, 23), cond=None, cb=(-1, 2, 6, 10, 14, 17, 20), pb=(-1, 0, 3, 6, 10, 15, 20, 25, 30)):
    rows = []
    for t in rounds:
        if t + k > 23: continue
        m = v[..., t] & v[..., t + k]
        if cond is not None: m &= cond[..., t]
        rows.append(pd.DataFrame({'c': c[..., t][m], 'p': p[..., t][m], 'd': (c[..., t + k] - c[..., t])[m]}))
    df = pd.concat(rows)
    df['cb'] = pd.cut(df.c, cb); df['pb'] = pd.cut(df.p, pb)
    m = df.pivot_table(index='cb', columns='pb', values='d', aggfunc='mean', observed=True).round(2)
    n = df.pivot_table(index='cb', columns='pb', values='d', aggfunc='size', observed=True)
    return m.where(n >= 200)

if __name__ == '__main__':
    for k in [1, 3, 6]:
        print(f'change after {k} round(s), all rounds'); print(table(k).to_string())
    print('after 1, round 0 only'); print(table(1, [0]).to_string())
    print('after 1, rounds 5-15'); print(table(1, range(5, 16)).to_string())
    print('after 3, round 0 only'); print(table(3, [0]).to_string())
