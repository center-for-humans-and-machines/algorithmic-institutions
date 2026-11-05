# v5: v4 sweep, my group's starting members: value to my pool in rounds 2-11 and staying, by round-0 contribution x round-0 hit
import pandas as pd, numpy as np
X = pd.read_parquet('/tmp/v5/r1.parquet')  # written by v5_s3_round1.py
pd.set_option('display.width', 250)
X['c0b'] = pd.cut(X.c0, [-1, 2, 5, 8, 10, 12, 20])
X['p0b'] = pd.cut(X.p0, [-1, 0, 10, 20, 29, 30])
X['val01'] = X.val + 1.6 * (X.c0 + X.c1) - X.p0 - X.p1
for v in ['stay4', 'val', 'val01']:
    print(v); print(X.pivot_table(index='c0b', columns='p0b', values=v, aggfunc='mean', observed=True).round(2))
print(X.pivot_table(index='c0b', columns='p0b', values='val', aggfunc='size', observed=True))
# same for opponent_4 only
Y = X[X.opp == 4]
print('opp4 val01'); print(Y.pivot_table(index='c0b', columns='p0b', values='val01', aggfunc='mean', observed=True).round(2))
