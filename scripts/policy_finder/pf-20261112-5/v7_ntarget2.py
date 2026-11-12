"""v7: shock vs no shock at round R (the lower-dose game deals 0 at R): d lead R..23 and d mem by members hit x
dose per member hit. Mean diffs (late points diff reported, not controlled), plus version pair mix."""
import numpy as np, pandas as pd
z = np.load('/tmp/pf/pairs.npz')
for R0 in [0, 4]:
    i = (z['R'] == R0) & (z['dd'] != 0)
    s = np.sign(z['dd'][i]).astype(float)
    plo = np.where(s > 0, z['pb'][i], z['pa'][i]); nhit = np.where(s > 0, z['na'][i], z['nb'][i])
    vhi = np.where(s > 0, z['va'][i], z['vb'][i]); vlo = np.where(s > 0, z['vb'][i], z['va'][i])
    d = pd.DataFrame(dict(dd=z['dd'][i] * s, plo=plo, nhit=np.minimum(nhit, 4), dlead=z['dlead'][i] * s, down=z['down'][i] * s,
                          dmem=z['dmem'][i] * s, late=z['late'][i].sum(1) * s, nlow=z['nlow'][i], vhi=vhi, vlo=vlo))
    d = d[d.plo == 0]
    d['dose'] = pd.cut(d.dd / d.nhit, [0, 5, 10, 15, 20, 25, 31])
    print(f'R{R0}: shock vs none, pairs', len(d))
    g = d.groupby(['nhit', 'dose'], observed=True).agg(n=('dlead', 'size'), dlead=('dlead', 'mean'), down=('down', 'mean'),
                                                        dmem=('dmem', 'mean'), late=('late', 'mean'))
    print(g[g.n > 300].round(2).to_string())
    print(pd.crosstab(d.nhit, d.vhi).to_string())
