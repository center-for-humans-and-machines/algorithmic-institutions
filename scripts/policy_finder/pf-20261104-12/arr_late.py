"""Does punishment late in the game pay? For members of the rule's group at a block's
first round t, by own contribution and punishment received at t: their mean contribution
in each later round while in the group, and the share still in the group, and a rough
pool account: sum over later rounds of 1.6*c*in_group minus the punishment (own
member only; ignores spill-overs onto others and later punishment)."""
import sys, numpy as np, pandas as pd
z = np.load(sys.argv[1]); c, p, g, v = (z[k].astype(np.int16) for k in 'cpgv')
for t in [int(a) for a in sys.argv[2].split(',')]:
    m = (g[..., t] == 0) & (v[..., t] == 1)
    cc = c[..., t][m]; pp = p[..., t][m]
    fut_in = (g[..., t+1:][m] == 0)
    fut_c = c[..., t+1:][m] * fut_in
    fut_p = p[..., t+1:][m] * fut_in
    d = pd.DataFrame({'c': cc, 'p': pp, 'val': 1.6*fut_c.sum(-1) - fut_p.sum(-1) - pp, 'gain_c': fut_c.sum(-1), 'in_end': fut_in[:, -1], 'futp': fut_p.sum(-1)})
    d['cb'] = pd.cut(d.c, [-1, 0, 4, 8, 12, 20]); d['pb'] = pd.cut(d.p, [-1, 0, 8, 16, 24, 30])
    tab = d.groupby(['cb', 'pb'], observed=True).agg(val=('val', 'mean'), in_end=('in_end', 'mean'), futp=('futp', 'mean'), n=('val', 'size'))
    tab = tab[tab.n > 300]
    print(f'--- t={t}: own pool account over rounds t..23 (val), share in group at 23, later punishment')
    print(tab.val.unstack().round(1)); print(tab.in_end.unstack().round(2))
