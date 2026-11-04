"""v1 sweep as a natural experiment: in the rule's group (agent_group 0), how the
next rounds' contribution and switching answer the punishment a member got,
at fixed own contribution. Punishment varies across design points at fixed c."""
import pandas as pd, numpy as np, glob, warnings; warnings.filterwarnings('ignore')
fs = sorted(glob.glob('plots/simulation/policy_finder/pf-20261104-12_sweep/per_round_p*of2.parquet'))
out = []
for i, f in enumerate(fs):
    d = pd.read_parquet(f, columns=['episode','participant_code','round_number','punishment','contribution','agent_group','contribution_valid','run'])
    d['ep'] = d.episode + 10000*i
    d = d.sort_values(['run','ep','participant_code','round_number'])
    g = d.groupby(['run','ep','participant_code'])
    for k in (1, 2, 4):
        d[f'c{k}'] = g.contribution.shift(-k); d[f'g{k}'] = g.agent_group.shift(-k)
    out.append(d)
d = pd.concat(out); del out
d.to_parquet('/tmp/v1sweep.parquet')
x = d[(d.agent_group == 0) & d.contribution_valid.astype(bool)]
# contribution response, rounds where staying for 2 rounds guaranteed (pos 0 in block -> t+1 same block)
y = x[(x.round_number % 4 == 0) & (x.round_number <= 16) & (x.g1 == 0)]
y['cb'] = pd.cut(y.contribution, [-1, 0, 3, 6, 9, 12, 15, 20])
y['pb'] = pd.cut(y.punishment, [-1, 0, 5, 10, 15, 20, 25, 30])
print('block-first rounds, c_{t+1}-c_t by own c and punishment (n>200)')
t = y.groupby(['cb','pb']).apply(lambda z: pd.Series({'dc1': (z.c1-z.contribution).mean(), 'n': len(z)}))
print(t[t.n > 200].dc1.unstack().round(2))
# switching at the decision after a block vs total punishment in the block
x = x.copy()
x['blk'] = x.round_number // 4
bl = x[x.round_number <= 19].groupby(['run','ep','participant_code','blk']).agg(P=('punishment','sum'), c=('contribution','mean'), n=('round_number','size'), last=('round_number','max'))
last = x[x.round_number % 4 == 3][['run','ep','participant_code','blk','g1']].set_index(['run','ep','participant_code','blk'])
bl = bl.join(last, how='inner'); bl = bl[bl.n == 4]
bl['left'] = (bl.g1 != 0).astype(float)
bl['Pb'] = pd.cut(bl.P, [-1, 0, 10, 20, 40, 60, 90, 120])
bl['cb'] = pd.cut(bl.c, [-1, 5, 10, 15, 20])
print('leave rate by block punishment (rows) and mean contribution (cols)')
print(bl.groupby(['Pb','cb']).left.mean().unstack().round(3))
print(bl.groupby(['Pb','cb']).left.size().unstack())
