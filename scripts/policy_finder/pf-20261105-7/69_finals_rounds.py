"""Per-round mean c, p and payoff contribution for the best final points of v3/v4/v5."""
import numpy as np, sys
for v,names in [('pf-20261105-7-v3',['pf-20261105-7-v3_gp']),('pf-20261105-7-v4',['pf-20261105-7-v4_s044']),('pf-20261105-7-v5',['pf-20261105-7-v5_gp','pf-20261105-7-v5_s201'])]:
    d=np.load(f'/tmp/pf7/{v}_final.npz')
    for n in names:
        m=d['R']==n; C=d['C'][m].astype(float); P=d['P'][m].astype(float); V=d['V'][m].astype(float)
        nv=V.sum(1); pay=(V*(20-C-P)).sum(1)/nv + (1.6*(C*V).sum(1)-(P*V).sum(1))/nv
        print(n, m.sum(), 'pay %.1f'%pay.sum(1).mean(), 'P/player %.1f'%(P*V).sum((1,2)).mean().__truediv__(4))
        print('  c', ' '.join('%4.1f'%x for x in (C*V).sum((0,1))/V.sum((0,1))))
        print('  p', ' '.join('%4.1f'%x for x in (P*V).sum((0,1))/V.sum((0,1))))
