"""v7: round-0 shock vs none, dose per member hit >= 12, by members hit x members giving <=5, with the later
points difference limited (|late| <= LIM). d lead R..23, d own, d mem (mean over rounds 0..23 from R)."""
import numpy as np, pandas as pd, sys
LIM = float(sys.argv[1]) if len(sys.argv) > 1 else 10
z = np.load('/tmp/pf/pairs.npz')
for R0 in [0, 4]:
    i = (z['R'] == R0) & (z['dd'] != 0)
    s = np.sign(z['dd'][i]).astype(float)
    plo = np.where(s > 0, z['pb'][i], z['pa'][i]); nhit = np.where(s > 0, z['na'][i], z['nb'][i])
    d = pd.DataFrame(dict(dd=z['dd'][i] * s, plo=plo, nhit=np.minimum(nhit, 4), dlead=z['dlead'][i] * s, down=z['down'][i] * s,
                          dmem=z['dmem'][i] * s, late=z['late'][i].sum(1) * s, nlow=np.minimum(z['nlow'][i], 4), n=z['n'][i],
                          cm=z['cm'][i]))
    d = d[(d.plo == 0) & (d.late.abs() <= LIM) & (d.dd / d.nhit >= 12)]
    print(f'R{R0}: shock (>=12 per member hit) vs none, |late|<={LIM}: pairs {len(d)}')
    g = d.groupby(['nlow', 'nhit'], observed=True).agg(n=('dlead', 'size'), dd=('dd', 'mean'), dlead=('dlead', 'mean'),
                                                       se=('dlead', 'sem'), down=('down', 'mean'), dmem=('dmem', 'mean'), late=('late', 'mean'))
    print(g[g.n > 100].round(2).to_string())
