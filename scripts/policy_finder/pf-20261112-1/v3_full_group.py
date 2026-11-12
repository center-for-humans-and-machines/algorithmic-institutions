# v3: the push when my group holds all 8 (ah's group empty) or 7, at decisions 7-19 of v2's sweep:
# effect of the decision-round punishment (in bins) on the rest-of-game lead, own and ah's pool, members.
# Effects are per later round; "sum" multiplies by the rounds left. Reads /tmp/v3_push.parquet (v3_push_effect.py).
import pandas as pd, numpy as np, statsmodels.formula.api as smf
r=pd.read_parquet('/tmp/v3_push.parquet')
r['Tb']=pd.cut(r['T'],[-1,0,9,30,60,240],labels=['0','1-9','10-30','31-60','61+']).astype(str)
r['left']=23-r.dec
for nn in [8,7,6]:
    x=r[(r.n==nn)&(r.dec>=7)&(r.dec<=19)].copy()
    print(f'=== my group {nn} members at the decision; rows {len(x)}; share with a member of window mean <=1: {(x.n_low1>0).mean():.2f}')
    for y in ['lead_after','own_after','oth_after','n_end']:
        m=smf.ols(f'{y} ~ C(Tb, Treatment("0")):C(dec) + Tearly + C(dec)*(C(low) + C(low3) + C(low5) + cmw + cow + lead_win + lead_before)',x).fit()
        out=[]
        for tb in ['1-9','10-30','31-60','61+']:
            ks=[k for k in m.params.index if f'[T.{tb}]' in k]
            # pool decisions 7, 11, 15 (weights equal) as sums over the rounds left
            v=[(m.params[k]*(23-int(k.split('[')[-1][:-1])) if y!='n_end' else m.params[k]) for k in ks if not k.endswith('[19]')]
            out.append(f'{tb}: '+' '.join(f'{a:.1f}' for a in v))
        print(f'  {y} (dec 7/11/15, summed over later rounds):',' | '.join(out))
    x10=x[x.Tb.isin(['0','10-30'])]
    print('  mean T in 10-30 bin %.1f'%x[x.Tb=='10-30']['T'].mean())

# pooled test: decisions 7, 11, 15; outcome the lead summed over the rest of the game; one dummy per T bin
print('pooled over decisions 7/11/15, lead summed over later rounds, effect (se) by T bin:')
for nn in [8,7,6,5]:
    x=r[(r.n==nn)&r.dec.isin([7,11,15])].copy(); x['lead_sum']=x.lead_after*x.left; x['own_sum']=x.own_after*x.left; x['oth_sum']=x.oth_after*x.left
    res=[]
    for y in ['lead_sum','own_sum','oth_sum']:
        m=smf.ols(f'{y} ~ C(Tb, Treatment("0")) + Tearly + C(dec)*(C(low) + C(low3) + C(low5) + cmw + cow + lead_win + lead_before)',x).fit()
        key=lambda tb: 'C(Tb, Treatment("0"))[T.'+tb+']'
        res.append(y+': '+' '.join('%s %.1f(%.1f)'%(tb,m.params[key(tb)],m.bse[key(tb)]) for tb in ['1-9','10-30','31-60','61+']))
    print(f' n={nn}:', ' || '.join(res))
